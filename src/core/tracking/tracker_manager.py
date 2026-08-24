"""One tracker instance per camera, managed centrally."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ...utils.logger import get_logger
from ..types import Detection, Track
from .byte_tracker import BYTETracker

logger = get_logger(__name__)

# Signature of an embedder callback: crops -> (N, D) L2-normalized features.
EmbedderFn = Callable[[list[np.ndarray]], np.ndarray | None]


class TrackerManager:
    """Runs a BYTETracker per camera and exposes typed Track outputs."""

    def __init__(
        self,
        camera_ids: Sequence[str],
        tracker_config: dict | None = None,
        embedder: EmbedderFn | None = None,
    ) -> None:
        cfg = dict(tracker_config or {})
        self.embedder = embedder
        self.trackers: dict[str, BYTETracker] = {
            cam_id: BYTETracker(camera_id=cam_id, **cfg) for cam_id in camera_ids
        }
        self._last_frame_ts: dict[str, float] = {}

    def update(
        self,
        camera_id: str,
        detections: Sequence[Detection],
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> list[Track]:
        """Update the given camera's tracker and return confirmed Tracks."""
        if camera_id not in self.trackers:
            raise KeyError(f"Unknown camera: {camera_id}")

        ts = timestamp if timestamp is not None else self._last_frame_ts.get(camera_id, 0.0)
        self._last_frame_ts[camera_id] = ts

        # Extract embeddings per-detection BEFORE association so the tracker
        # can blend appearance cost and seed new tracks with features.
        features: np.ndarray | None = None
        if self.embedder is not None and frame is not None and len(detections) > 0:
            h, w = frame.shape[:2]
            crops = []
            for det in detections:
                x1, y1, x2, y2 = (int(v) for v in det.bbox)
                crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
                if crop.size == 0:
                    crop = None
                crops.append(crop)
            valid = [c for c in crops if c is not None]
            try:
                feats = self.embedder(valid) if valid else None
            except Exception as exc:  # pragma: no cover - model dependent
                logger.warning("Embedder failed (%s); continuing without features", exc)
                feats = None
            if feats is not None:
                features = np.zeros((len(crops), feats.shape[1]), dtype=feats.dtype)
                fill = 0
                for idx, crop in enumerate(crops):
                    if crop is not None:
                        features[idx] = feats[fill]
                        fill += 1

        raw_tracks = self.trackers[camera_id].update(detections, features=features, timestamp=ts)

        result = [
            Track(
                track_id=rt["track_id"],
                camera_id=camera_id,
                bbox=rt["bbox"],
                confidence=rt["confidence"],
                is_confirmed=True,
                hits=rt["hits"],
                feature=rt.get("feature"),
            )
            for rt in raw_tracks
        ]
        return result

    def update_all(
        self,
        camera_detections: dict[str, Sequence[Detection]],
        frames: dict[str, np.ndarray] | None = None,
        timestamps: dict[str, float] | None = None,
    ) -> dict[str, list[Track]]:
        out: dict[str, list[Track]] = {}
        for cam_id, dets in camera_detections.items():
            frame = frames.get(cam_id) if frames else None
            ts = timestamps.get(cam_id) if timestamps else None
            out[cam_id] = self.update(cam_id, dets, frame=frame, timestamp=ts)
        return out
