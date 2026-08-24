"""Thread-safe runtime state shared between the pipeline and the API.

The pipeline (background thread) publishes:
  * latest JPEG-encoded frames per camera
  * confirmed tracks with global IDs
  * world positions + path trails on the floor plan
  * camera stats
  * rule violations / events

The FastAPI layer reads snapshots for REST responses, MJPEG streams, and the
WebSocket broadcast. All reads/writes are guarded by a reentrant lock; frames
are encoded to JPEG once by the producer so N stream clients cost nothing.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CameraStatus:
    camera_id: str
    name: str = ""
    source: str = ""
    online: bool = False
    fps: float = 0.0
    dropped: int = 0
    last_frame_ts: float = 0.0


@dataclass
class HubEvent:
    ts: float
    rule_type: str
    zone_name: str
    global_id: int | None
    reason: str
    snapshot_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "rule_type": self.rule_type,
            "zone_name": self.zone_name,
            "global_id": self.global_id,
            "reason": self.reason,
            "snapshot": self.snapshot_available,
        }


@dataclass
class _CamState:
    frame_jpeg: bytes | None = None
    annotated_jpeg: bytes | None = None
    frame_ts: float = 0.0
    status: CameraStatus = field(default_factory=lambda: CameraStatus(camera_id=""))


class StateHub:
    """Central mutable state; safe for cross-thread use."""

    def __init__(self, event_buffer: int = 300) -> None:
        self._lock = threading.RLock()
        self._cameras: dict[str, _CamState] = {}
        # camera_id -> list of track dicts
        self._tracks: dict[str, list[dict[str, Any]]] = {}
        # gid -> (x, y)
        self._positions: dict[int, tuple[float, float]] = {}
        # gid -> trail points (world coords)
        self._paths: dict[int, list[tuple[float, float]]] = {}
        # gid -> prediction dicts {points, next_zone}
        self._predictions: dict[int, dict] = {}
        self._events: deque[HubEvent] = deque(maxlen=event_buffer)
        self._alerts: set[int] = set()
        # Cumulative processed-frame count per camera (for monitoring).
        self.frames_processed: dict[str, int] = {}
        # Monotonic event counter (buffer may trim; this never resets).
        self.events_total = 0
        self.started_at = time.time()
        self.pipeline_running = False

    def bump_frames(self, camera_id: str, n: int = 1) -> None:
        with self._lock:
            self.frames_processed[camera_id] = self.frames_processed.get(camera_id, 0) + n

    def frames_totals(self) -> dict[str, int]:
        with self._lock:
            return dict(self.frames_processed)

    # ------------------------------------------------------------------ #
    # Cameras & frames
    # ------------------------------------------------------------------ #
    def register_camera(self, camera_id: str, name: str = "", source: str = "") -> None:
        with self._lock:
            self._cameras[camera_id] = _CamState(
                status=CameraStatus(camera_id=camera_id, name=name, source=str(source))
            )

    def set_camera_status(
        self,
        camera_id: str,
        *,
        online: bool | None = None,
        fps: float | None = None,
        dropped: int | None = None,
        name: str | None = None,
    ) -> None:
        with self._lock:
            cam = self._cameras.setdefault(
                camera_id, _CamState(status=CameraStatus(camera_id=camera_id))
            )
            st = cam.status
            if online is not None:
                st.online = online
            if fps is not None:
                st.fps = fps
            if dropped is not None:
                st.dropped = dropped
            if name is not None:
                st.name = name

    def set_frame(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        annotated_bgr: np.ndarray | None = None,
        quality: int = 80,
    ) -> None:
        """Encode + store the latest frame (called from the pipeline thread)."""
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return
        ok_a, buf_a = (
            cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if annotated_bgr is not None
            else (False, None)
        )
        ts = time.time()
        with self._lock:
            cam = self._cameras.setdefault(
                camera_id, _CamState(status=CameraStatus(camera_id=camera_id))
            )
            cam.frame_jpeg = buf.tobytes()
            if ok_a and buf_a is not None:
                cam.annotated_jpeg = buf_a.tobytes()
            cam.frame_ts = ts
            cam.status.last_frame_ts = ts
            cam.status.online = True

    def get_frame(self, camera_id: str, annotated: bool = True) -> bytes | None:
        with self._lock:
            cam = self._cameras.get(camera_id)
            if cam is None:
                return None
            if annotated:
                return cam.annotated_jpeg or cam.frame_jpeg
            return cam.frame_jpeg

    # ------------------------------------------------------------------ #
    # Tracks / world state
    # ------------------------------------------------------------------ #
    def update_tracks(self, camera_id: str, tracks: list[dict[str, Any]]) -> None:
        serialized = [
            {
                "camera_id": t.camera_id,
                "track_id": t.track_id,
                "global_id": t.global_id,
                "bbox": [round(v, 1) for v in t.bbox],
                "confidence": round(float(t.confidence), 3),
                "name": t.name,
                "roles": list(t.roles),
            }
            for t in tracks
        ]
        with self._lock:
            self._tracks[camera_id] = serialized

    def set_world_state(
        self,
        positions: dict[int, tuple[float, float]],
        paths: dict[int, list[tuple[float, float]]],
        alerts: set[int] | None = None,
        predictions: dict[int, dict] | None = None,
    ) -> None:
        with self._lock:
            self._positions = dict(positions)
            self._paths = {gid: pts[-120:] for gid, pts in paths.items()}
            if alerts is not None:
                self._alerts = set(alerts)
            if predictions is not None:
                self._predictions = dict(predictions)

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def add_event_from_violation(self, violation) -> None:
        event = HubEvent(
            ts=float(getattr(violation, "timestamp", time.time())),
            rule_type=violation.rule_type,
            zone_name=violation.zone_name,
            global_id=violation.global_id,
            reason=violation.reason,
        )
        with self._lock:
            self._events.appendleft(event)
            self.events_total += 1

    def recent_events(self, limit: int = 50) -> list[HubEvent]:
        with self._lock:
            return list(self._events)[:limit]

    # ------------------------------------------------------------------ #
    # Snapshots
    # ------------------------------------------------------------------ #
    def camera_statuses(self) -> list[CameraStatus]:
        with self._lock:
            # Copies so callers can't mutate hub state.
            return [replace(c.status) for c in self._cameras.values()]

    def tracks_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for per_cam in self._tracks.values():
                out.extend(per_cam)
            return out

    def live_snapshot(self) -> dict[str, Any]:
        """Compact JSON-able state for WebSocket pushes."""
        with self._lock:
            return {
                "type": "state",
                "ts": time.time(),
                "uptime_s": round(time.time() - self.started_at, 1),
                "running": self.pipeline_running,
                "tracks": self.tracks_snapshot(),
                "positions": {
                    str(g): [round(x, 1), round(y, 1)] for g, (x, y) in self._positions.items()
                },
                "paths": {
                    str(g): [[round(px, 1), round(py, 1)] for px, py in pts[-60:]]
                    for g, pts in self._paths.items()
                },
                "alerts": sorted(self._alerts),
                "predictions": self._predictions,
                "events": [e.to_dict() for e in list(self._events)[:20]],
                "cameras": {
                    cid: {
                        "online": c.status.online,
                        "fps": round(c.status.fps, 1),
                        "dropped": c.status.dropped,
                    }
                    for cid, c in self._cameras.items()
                },
            }
