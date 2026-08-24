"""Cross-camera global identity management.

Maintains a feature gallery per global ID and matches new local tracks using
cosine similarity plus optional spatio-temporal gating:

* temporal — same person seen on camera B within `max_time_gap` seconds of
  disappearing from camera A;
* spatial — cameras far apart in world coordinates can only match within a
  plausible time window (travel-time feasibility).

The feasibility check uses per-camera anchor points when available
(`camera_positions`: cam_id -> (x, y) meters); without calibration data it
degrades to pure appearance matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...utils.geometry import euclidean
from ...utils.logger import get_logger

logger = get_logger(__name__)

WALK_SPEED_MPS = 1.6  # conservative human walking speed for travel-time gating


@dataclass
class IdentityGallery:
    global_id: int
    features: list[np.ndarray] = field(default_factory=list)
    last_seen_ts: float = -np.inf
    cameras: set[str] = field(default_factory=set)


class GlobalIDManager:
    """Assigns persistent global IDs across cameras."""

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        gallery_size: int = 50,
        max_time_gap: float = 60.0,
        max_distance_between_cameras: float = 50.0,
        camera_positions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.threshold = float(similarity_threshold)
        self.gallery_size = int(gallery_size)
        self.max_time_gap = float(max_time_gap)
        self.max_cam_distance = float(max_distance_between_cameras)
        self.camera_positions = dict(camera_positions or {})

        self.galleries: dict[int, IdentityGallery] = {}
        self.camera_to_global: dict[tuple[str, int], int] = {}
        self.next_global_id = 1

    # ------------------------------------------------------------------ #
    def match_or_create(
        self,
        camera_id: str,
        track_id: int,
        feature: np.ndarray | None,
        timestamp: float = 0.0,
    ) -> int:
        """Map ``(camera_id, track_id)`` to a global ID, matching by appearance."""
        key = (str(camera_id), int(track_id))
        if key in self.camera_to_global:
            gid = self.camera_to_global[key]
            if feature is not None:
                self._update_gallery(gid, feature, timestamp, camera_id)
            return gid

        gid: int | None = None
        if feature is not None:
            best_sim, best_gid = self._best_match(feature, timestamp, camera_id)
            if best_gid is not None and best_sim >= self.threshold:
                gid = best_gid

        if gid is None:
            gid = self.next_global_id
            self.next_global_id += 1
            self.galleries[gid] = IdentityGallery(global_id=gid)

        if feature is not None:
            self._update_gallery(gid, feature, timestamp, camera_id)
        self.galleries[gid].cameras.add(str(camera_id))
        self.camera_to_global[key] = gid
        return gid

    # ------------------------------------------------------------------ #
    def _best_match(
        self, feature: np.ndarray, timestamp: float, camera_id: str
    ) -> tuple[float, int | None]:
        best_sim = -1.0
        best_gid: int | None = None

        feat = feature / (np.linalg.norm(feature) + 1e-9)
        for gallery in self.galleries.values():
            if not self._feasible(gallery, timestamp, camera_id):
                continue
            if len(gallery.features) == 0:
                continue
            mat = np.stack(gallery.features)
            sims = mat @ feat
            sim = float(np.max(sims))
            if sim > best_sim:
                best_sim, best_gid = sim, gallery.global_id

        return best_sim, best_gid

    def _feasible(self, gallery: IdentityGallery, timestamp: float, camera_id: str) -> bool:
        """Spatio-temporal feasibility of matching this gallery right now."""
        dt = timestamp - gallery.last_seen_ts
        if dt > self.max_time_gap:
            return False
        if dt < 0:
            return True  # out-of-order frames: don't gate

        pos_b = self.camera_positions.get(camera_id)
        if pos_b is None:
            return True  # no calibration → appearance-only

        feasible = False
        for other_cam in gallery.cameras:
            pos_a = self.camera_positions.get(other_cam)
            if pos_a is None:
                continue
            dist = euclidean(pos_a, pos_b)
            required_time = dist / WALK_SPEED_MPS + 0.75  # small processing slack
            if dt >= min(required_time, self.max_time_gap):
                feasible = True
        # If no calibrated partner camera contributed yet, allow the match.
        return feasible or all(c not in self.camera_positions for c in gallery.cameras)

    def _update_gallery(
        self, gid: int, feature: np.ndarray, timestamp: float, camera_id: str
    ) -> None:
        gallery = self.galleries.setdefault(gid, IdentityGallery(global_id=gid))
        normed = feature / (np.linalg.norm(feature) + 1e-9)
        gallery.features.append(normed.astype(np.float32))
        if len(gallery.features) > self.gallery_size:
            gallery.features = gallery.features[-self.gallery_size :]
        gallery.last_seen_ts = max(gallery.last_seen_ts, float(timestamp))
        gallery.cameras.add(str(camera_id))

    # ------------------------------------------------------------------ #
    def get_global_id(self, camera_id: str, track_id: int) -> int | None:
        return self.camera_to_global.get((str(camera_id), int(track_id)))

    def forget_track(self, camera_id: str, track_id: int) -> None:
        """Drop the (camera, local-track) binding once its tracker loses it."""
        self.camera_to_global.pop((str(camera_id), int(track_id)), None)
