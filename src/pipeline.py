"""End-to-end surveillance pipeline orchestration.

Wires: cameras → detection → tracking → Re-ID → homography mapping →
path aggregation → rule engine → alerts → rendering.

Every stage is optional at runtime so the pipeline degrades gracefully on
machines without GPU/models/calibration.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .alerts.alert_manager import AlertManager
from .core.anomaly.anomaly_detector import DwellAnomalyDetector, SpeedAnomalyDetector
from .core.camera.camera_manager import CameraManager
from .core.detection.detector import create_detector
from .core.mapping.coordinate_mapper import CoordinateMapper
from .core.mapping.floor_plan import FloorPlan
from .core.mapping.heatmap import Heatmap
from .core.path.database import AnalyticsDB, AnalyticsWriter
from .core.path.path_tracker import PathTracker
from .core.path.predictor import VelocityPredictor, ZoneMarkov
from .core.reid.embedder import create_embedder
from .core.reid.global_id_manager import GlobalIDManager
from .core.rules.rule_engine import RuleEngine
from .core.state_hub import StateHub
from .core.tracking.tracker_manager import TrackerManager
from .utils.config import load_zones_config
from .utils.logger import get_logger
from .utils.visualization import draw_tracks

logger = get_logger(__name__)


class SurveillancePipeline:
    def __init__(
        self,
        cameras_config: str | Path = "config/cameras.yaml",
        zones_config: str | Path | None = "config/zones.yaml",
        model_config: dict | None = None,
        alerts_config_path: str | Path | None = None,
        device: str | None = None,
        enable_reid: bool = True,
        enable_display: bool = False,
        state_hub: StateHub | None = None,
    ) -> None:
        self.enable_display = bool(enable_display)
        self.hub = state_hub
        cfg = dict(model_config or {})

        # ---- Cameras -------------------------------------------------- #
        self.camera_manager = CameraManager(cameras_config)
        if self.hub is not None:
            for cam in self.camera_manager.cameras.values():
                self.hub.register_camera(
                    cam.id, name=getattr(cam, "name", ""), source=str(cam.source)
                )

        # ---- Detection ------------------------------------------------ #
        det_cfg = dict(cfg.get("detection", {}))
        backend = det_cfg.get("model_type", "auto")
        if backend in ("yolov8", "yolov11", "yolo"):
            backend = "yolo"
        self.detector = create_detector(
            backend="auto" if backend == "auto" else backend,
            model_path=det_cfg.get("model_path", "yolo11m.pt"),
            confidence_threshold=float(det_cfg.get("confidence_threshold", 0.5)),
            device=device or det_cfg.get("device") or None,
            input_size=int(det_cfg.get("input_size", 640)),
        )

        # ---- Tracking + Re-ID ------------------------------------------ #
        embedder = None
        if enable_reid:
            try:
                embedder = create_embedder(str(cfg.get("reid", {}).get("backend", "auto")))
            except Exception as exc:
                logger.warning("Embedder init failed (%s); Re-ID disabled", exc)

        reid_cfg = dict(cfg.get("reid", {}))
        track_cfg = dict(cfg.get("tracking", {}))
        self.global_ids = GlobalIDManager(
            similarity_threshold=float(reid_cfg.get("similarity_threshold", 0.6)),
            gallery_size=int(reid_cfg.get("gallery_size", 50)),
            max_time_gap=float(reid_cfg.get("max_time_gap", 60)),
            max_distance_between_cameras=float(reid_cfg.get("max_distance_between_cameras", 50.0)),
        )
        tracker_params = {
            k: float(track_cfg[k])
            for k in (
                "track_high_thresh",
                "track_low_thresh",
                "new_track_thresh",
                "match_thresh",
                "appearance_weight",
            )
            if k in track_cfg
        }
        if not tracker_params:
            tracker_params = {"track_high_thresh": 0.5, "new_track_thresh": 0.6}

        self.tracker_manager = TrackerManager(
            list(self.camera_manager.cameras.keys()),
            tracker_config=tracker_params,
            embedder=embedder.embed if embedder is not None else None,
        )

        # ---- Mapping / floor plan -------------------------------------- #
        self.mapper = CoordinateMapper()
        # Calibration files may sit beside cameras.yaml OR in project config/.
        calib_dirs = [Path(cameras_config).parent, Path(__file__).resolve().parents[1] / "config"]
        for cam_id in self.camera_manager.cameras:
            for calib_dir in calib_dirs:
                calib_file = calib_dir / f"calibration_{cam_id}.yaml"
                if calib_file.exists():
                    try:
                        self.mapper.load_calibration(cam_id, calib_file)
                        break
                    except Exception as exc:
                        logger.warning("Calibration %s failed: %s", cam_id, exc)

        first_cam = next(iter(self.camera_manager.cameras), None)
        floor_id = self.mapper.floor_of(first_cam) if first_cam else "floor_0"

        zones = []
        if zones_config and Path(zones_config).exists():
            zones = [z for z in load_zones_config(zones_config) if z["floor_id"] == floor_id]

        floor_image = cfg.get("floor_plan_path")
        if not floor_image:
            floor_image = next((z["floor_plan"] for z in zones if z.get("floor_plan")), None)
        if floor_image and not Path(floor_image).exists():
            logger.warning("Floor plan image missing: %s — using blank canvas", floor_image)
            floor_image = None

        self.floor_plan = FloorPlan(image_path=floor_image, floor_id=floor_id)
        for z in zones:
            self.floor_plan.add_zone(
                name=z["name"],
                polygon=z["polygon"],
                zone_type=z["type"],
                rule=z["rule"],
                params=z["params"],
                color=z.get("color"),
            )
        logger.info("Floor plan ready with %d zone(s)", len(self.floor_plan.zones))

        self.global_ids.camera_positions.update(self._camera_anchors())

        # ---- Paths / rules / alerts ------------------------------------ #
        self.path_tracker = PathTracker()

        # ---- Phase 11 identity (opt-in, hardware-agnostic) -------------- #
        identity_cfg_path = Path(cameras_config).parent / "identity.yaml"
        if not identity_cfg_path.exists():
            identity_cfg_path = Path(__file__).resolve().parents[1] / "config" / "identity.yaml"
        identity_cfg: dict = {}
        if identity_cfg_path.exists():
            from .utils.config import load_yaml

            identity_cfg = load_yaml(identity_cfg_path)
        fr_cfg = dict(identity_cfg.get("face_recognition", {}))
        self.face_recognition_enabled = bool(fr_cfg.get("enabled", False))

        self.face_engine = None
        self.identity_registry = None
        self._face_sample_counter = 0
        self._face_sample_interval = 10
        # gid -> (name, roles) cache; refreshed on each successful face match.
        self.person_identities: dict[int, tuple[str | None, tuple[str, ...]]] = {}

        if self.face_recognition_enabled:
            try:
                from .core.identity.face_engine import create_face_engine
                from .core.identity.registry import IdentityRegistry

                self.face_engine = create_face_engine(identity_cfg)
                self.identity_registry = IdentityRegistry(
                    store_dir=fr_cfg.get("store_dir", "data/identities"),
                    match_distance_threshold=float(fr_cfg.get("match_threshold", 0.363)),
                )
                self._face_sample_interval = max(1, int(fr_cfg.get("sample_every_n_frames", 10)))
                logger.info(
                    "Face recognition ON (available=%s backend=%s/%s)",
                    self.face_engine.available,
                    self.face_engine.detector_backend,
                    self.face_engine.embedder_backend,
                )
                if not self.face_engine.available:
                    logger.warning(
                        "Face recognition enabled but models unavailable — "
                        "running WITHOUT identities (Tier 0)"
                    )
            except Exception as exc:
                logger.warning("Face engine init failed (%s); FR disabled", exc)
                self.face_recognition_enabled = False
        else:
            logger.info("Face recognition disabled (opt-in via config/identity.yaml)")

        self.rule_engine = RuleEngine(
            self.floor_plan,
            strict_unknown_roles=bool(fr_cfg.get("strict_unknown_roles", False)),
        )
        alert_cfg = {}
        if alerts_config_path and Path(alerts_config_path).exists():
            from .utils.config import load_yaml

            alert_cfg = load_yaml(alerts_config_path)
        self.alerts = AlertManager(alert_cfg)

        # ---- Phase 9 analytics (all optional, all graceful) ------------- #
        analytics_cfg = dict(cfg.get("analytics", {}))
        db_path = analytics_cfg.get("db_path", "data/output/analytics.db")
        self.analytics_enabled = bool(analytics_cfg.get("enabled", True))
        self.db: AnalyticsDB | None = None
        self.db_writer: AnalyticsWriter | None = None
        if self.analytics_enabled:
            try:
                self.db = AnalyticsDB(db_path)
                self.db_writer = AnalyticsWriter(
                    self.db,
                    flush_interval_s=float(analytics_cfg.get("db_flush_interval_s", 2.0)),
                )
            except Exception as exc:
                logger.warning("Analytics DB unavailable (%s); continuing without history", exc)

        h, w = self.floor_plan.image.shape[:2]
        self.heatmap = Heatmap(
            w,
            h,
            cell_size=int(analytics_cfg.get("heatmap_cell_size", 12)),
            decay_per_minute=float(analytics_cfg.get("heatmap_decay_per_minute", 0.5)),
        )
        self.predictor = VelocityPredictor(
            horizon_s=float(analytics_cfg.get("prediction_horizon_s", 3.0)),
        )
        self.zone_markov = ZoneMarkov()
        self.speed_anomaly = SpeedAnomalyDetector(
            z_threshold=float(analytics_cfg.get("anomaly_speed_z", 3.5)),
            absolute_floor=float(analytics_cfg.get("anomaly_speed_floor", 2.2)),
        )
        self.dwell_anomaly = DwellAnomalyDetector()
        self._zone_presence: dict[tuple[int, str], bool] = {}
        self._last_zone: dict[int, str | None] = {}

        self.frame_counter = 0

    # ------------------------------------------------------------------ #
    def _camera_anchors(self) -> dict[str, tuple[float, float]]:
        """Best-effort world anchor per camera (plan center of its view)."""
        anchors: dict[str, tuple[float, float]] = {}
        for cam_id in self.mapper.calibrations:
            try:
                h, w = 480.0, 640.0
                anchors[cam_id] = self.mapper.pixel_to_world(cam_id, w / 2.0, h * 0.75)
            except Exception:  # pragma: no cover - degenerate homography
                continue
        return anchors

    # ------------------------------------------------------------------ #
    def _resolve_identity(self, trk, frame: np.ndarray) -> tuple[str | None, tuple[str, ...]]:
        """Detect a face in the track's head region and match the registry."""
        assert self.face_engine is not None and self.identity_registry is not None
        try:
            hx, hy, hw, hh = self.face_engine.head_region_from_person_bbox(trk.bbox, frame.shape)
            head_crop = frame[hy : hy + hh, hx : hx + hw]
            faces = self.face_engine.detect(head_crop)
            if not faces:
                return None, ()
            best = max(faces, key=lambda f: f.score)
            face_img = self.face_engine.align_crop(head_crop, best)
            embedding = self.face_engine.embed(face_img)
            if embedding is None:
                return None, ()
            name, _dist = self.identity_registry.match(embedding)
            if name is None:
                return None, ()
            return name, tuple(self.identity_registry.roles_of(name))
        except Exception as exc:  # never let FR break tracking
            logger.debug("identity resolution failed: %s", exc)
            return None, ()

    def process_frame(self, camera_id: str, timestamp: float, frame: np.ndarray) -> dict:
        """Run all stages for one incoming frame; returns per-frame results."""
        self.frame_counter += 1
        detections = self.detector.detect(frame)

        tracks = self.tracker_manager.update(
            camera_id, detections, frame=frame, timestamp=timestamp
        )

        positions_world: dict[int, tuple[float, float]] = {}
        violations = []
        calibrated = self.mapper.is_calibrated(camera_id)

        # Phase 11: sample faces every N frames (CPU budget).
        self._face_sample_counter += 1
        do_face_sampling = (
            self.face_recognition_enabled
            and self.face_engine is not None
            and self.face_engine.available
            and self.identity_registry is not None
            and self._face_sample_counter % max(self._face_sample_interval, 1) == 0
        )

        for trk in tracks:
            if trk.feature is not None:
                trk.global_id = self.global_ids.match_or_create(
                    camera_id, trk.track_id, trk.feature, timestamp
                )
            else:
                gid = self.global_ids.get_global_id(camera_id, trk.track_id)
                trk.global_id = (
                    gid
                    if gid is not None
                    else self.global_ids.match_or_create(camera_id, trk.track_id, None, timestamp)
                )

            if do_face_sampling:
                name, roles = self._resolve_identity(trk, frame)
                if name is not None:
                    trk.name = name
                    trk.roles = roles
                    self.person_identities[trk.global_id] = (name, roles)

            cached_name, cached_roles = self.person_identities.get(trk.global_id, (None, ()))
            trk.name = trk.name or cached_name
            trk.roles = trk.roles or cached_roles

            if calibrated:
                wx, wy = self.mapper.bbox_to_world(camera_id, trk.bbox)
                positions_world[trk.global_id] = (wx, wy)

                path = self.path_tracker.update(trk.global_id, wx, wy, camera_id, timestamp)

                # Analytics: heatmap + DB + velocity model.
                self.heatmap.add_point(wx, wy, timestamp)
                self.predictor.observe(trk.global_id, wx, wy, timestamp)
                if self.db_writer is not None:
                    self.db_writer.add_position(
                        trk.global_id, timestamp, wx, wy, camera_id, self.mapper.floor_of(camera_id)
                    )

                # Zone entry/exit bookkeeping (dwell anomalies + Markov).
                zones_here = {z["name"] for z in self.floor_plan.check_point_in_zone(wx, wy)}
                for zname in zones_here:
                    path.record_zone_visit(zname)
                    key = (trk.global_id, zname)
                    if not self._zone_presence.get(key):
                        self._zone_presence[key] = True
                        self.dwell_anomaly.enter(trk.global_id, zname, timestamp)
                for key, present in list(self._zone_presence.items()):
                    gid_z, zname = key
                    if gid_z == trk.global_id and present and zname not in zones_here:
                        self._zone_presence[key] = False
                        self.dwell_anomaly.exit(gid_z, zname, timestamp)
                    elif gid_z != trk.global_id and not present:
                        del self._zone_presence[key]

                prev_zone = self._last_zone.get(trk.global_id)
                primary = next(iter(sorted(zones_here)), None)
                if primary != prev_zone:
                    if prev_zone is not None:
                        self.zone_markov.observe_sequence([prev_zone, primary or "outside"])
                    self._last_zone[trk.global_id] = primary

                # Rule violations + statistical anomalies.
                violations.extend(
                    self.rule_engine.evaluate(
                        path,
                        now=timestamp,
                        person_name=trk.name,
                        person_roles=trk.roles,
                    )
                )
                violations.extend(
                    self.rule_engine.evaluate_speed(path, max_speed=3.0, now=timestamp)
                )
                violations.extend(self.speed_anomaly.observe(path, now=timestamp))
                for zname in zones_here:
                    violations.extend(
                        self.dwell_anomaly.check_active(trk.global_id, zname, timestamp)
                    )

        for violation in violations:
            self.alerts.dispatch(violation.to_dict())
            if self.hub is not None:
                self.hub.add_event_from_violation(violation)
            if self.db is not None:
                try:
                    self.db.insert_event(violation.to_dict())
                except Exception as exc:  # pragma: no cover
                    logger.debug("event insert failed: %s", exc)

        # Publish to API hub: annotated frame + tracks + world state.
        if self.hub is not None:
            self.hub.bump_frames(camera_id)
            self.hub.update_tracks(camera_id, tracks)
            annotated = draw_tracks(frame.copy(), tracks)
            self.hub.set_frame(camera_id, frame, annotated)

            if calibrated and positions_world:
                paths = {gid: p.path_points() for gid, p in self.path_tracker.active_paths.items()}
                predictions = {}
                for gid in positions_world:
                    pred = self.predictor.predict(gid)
                    if pred is not None:
                        predictions[gid] = pred.to_dict()
                flagged = {v.global_id for v in violations}
                self.hub.set_world_state(
                    positions_world, paths, alerts=flagged, predictions=predictions
                )

        return {
            "detections": len(detections),
            "tracks": tracks,
            "positions": positions_world,
            "violations": violations,
        }

    # ------------------------------------------------------------------ #
    def render_floor_plan(self, results: list[dict]) -> np.ndarray:
        positions: dict[int, tuple[float, float]] = {}
        paths: dict[int, list[tuple[float, float]]] = {}
        flagged: dict[int, bool] = {v.global_id: True for r in results for v in r["violations"]}

        for gid, path in self.path_tracker.active_paths.items():
            pos = path.current_position
            if pos is not None:
                positions[gid] = pos[0], pos[1]
            pts = path.path_points()
            if len(pts) >= 2:
                paths[gid] = pts

        return self.floor_plan.render(tracks_world=positions, paths=paths, alerts=flagged)

    def render_heatmap(self) -> np.ndarray:
        """Plan blended with recent-traffic heat overlay."""
        return self.heatmap.render_overlay(self.floor_plan.image.copy())

    def run(
        self,
        max_frames: int | None = None,
        display: bool | None = None,
        show_plan_window: bool = True,
    ) -> int:
        """Blocking main loop. Returns number of frames processed."""
        use_display = self.enable_display if display is None else display
        started = self.camera_manager.start_all()
        if started == 0:
            raise RuntimeError("No camera could be opened — check config/sources")

        if self.hub is not None:
            self.hub.pipeline_running = True
            for cam_id, cam in self.camera_manager.cameras.items():
                self.hub.set_camera_status(cam_id, online=cam.is_running)

        logger.info(
            "Pipeline running (%d/%d cameras). Ctrl+C to stop.",
            started,
            len(self.camera_manager.cameras),
        )

        processed = 0
        last_reap = time.time()
        last_stats = 0.0

        try:
            for camera_id, ts, frame in self.camera_manager.iter_frames():
                result = self.process_frame(camera_id, ts, frame)
                processed += 1

                if use_display:
                    vis = draw_tracks(frame.copy(), result["tracks"])
                    cv2.putText(
                        vis,
                        f"{camera_id}  det={result['detections']} trk={len(result['tracks'])}",
                        (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(f"Camera {camera_id}", vis)

                    if show_plan_window:
                        cv2.imshow("Floor Plan", self.render_floor_plan([result]))

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if time.time() - last_reap > 30.0:
                    reaped = self.path_tracker.reap_stale()
                    if reaped:
                        logger.info("Archived %d stale path(s)", len(reaped))
                    last_reap = time.time()

                if self.hub is not None and (now := time.time()) - last_stats > 2.0:
                    for cam_id, cam in self.camera_manager.cameras.items():
                        self.hub.set_camera_status(
                            cam_id,
                            online=cam.is_running,
                            fps=cam.actual_fps,
                            dropped=cam.dropped_frames,
                        )
                    last_stats = now

                if self.db_writer is not None:
                    self.db_writer.maybe_flush()

                if max_frames is not None and processed >= max_frames:
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            if self.db_writer is not None:
                self.db_writer.close()
            if self.hub is not None:
                self.hub.pipeline_running = False
                for cam_id in self.camera_manager.cameras:
                    self.hub.set_camera_status(cam_id, online=False)
            self.camera_manager.stop_all()
            if use_display:
                cv2.destroyAllWindows()

        return processed

    # ------------------------------------------------------------------ #
    def run_in_thread(self, max_frames: int | None = None) -> threading.Thread:
        """Run the blocking loop on a daemon thread (for API-server hosting)."""

        def _target() -> None:
            try:
                self.run(max_frames=max_frames, display=self.enable_display)
            except Exception as exc:  # keep server alive if cameras fail
                logger.error("Pipeline thread crashed: %s", exc)
                if self.hub is not None:
                    self.hub.pipeline_running = False

        thread = threading.Thread(target=_target, name="pipeline", daemon=True)
        thread.start()
        return thread
