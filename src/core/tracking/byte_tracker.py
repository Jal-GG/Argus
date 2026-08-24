"""BYTE two-stage multi-object tracker.

Implements the ByteTrack association scheme (Zhang et al., 2022) with an
optional appearance-embedding term, i.e. the practical BoT-SORT recipe:

  Stage 1: high-confidence detections <-> active tracks
           cost = λ * IoU-distance + (1-λ) * cosine-distance (when features given)
  Stage 2: remaining tracks <-> low-confidence detections (IoU only)
           — recovers occluded people that YOLO scores poorly.

This replaces 2017-era DeepSORT with the current standard used by
Ultralytics' built-in trackers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...utils.logger import get_logger
from ..types import Detection
from .track import SingleTrack

logger = get_logger(__name__)


def iou_distance(tracks: Sequence[SingleTrack], detections: Sequence[Detection]) -> np.ndarray:
    """Cost matrix ``1 - IoU`` between track and detection boxes."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))

    t_boxes = np.array([t.bbox for t in tracks])
    d_boxes = np.array([d.bbox for d in detections])

    ix1 = np.maximum(t_boxes[:, None, 0], d_boxes[None, :, 0])
    iy1 = np.maximum(t_boxes[:, None, 1], d_boxes[None, :, 1])
    ix2 = np.minimum(t_boxes[:, None, 2], d_boxes[None, :, 2])
    iy2 = np.minimum(t_boxes[:, None, 3], d_boxes[None, :, 3])

    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih

    areas_t = (t_boxes[:, 2] - t_boxes[:, 0]) * (t_boxes[:, 3] - t_boxes[:, 1])
    areas_d = (d_boxes[:, 2] - d_boxes[:, 0]) * (d_boxes[:, 3] - d_boxes[:, 1])
    union = areas_t[:, None] + areas_d[None, :] - inter

    return 1.0 - np.where(union > 0, inter / union, 0.0)


def embedding_distance(tracks: Sequence[SingleTrack], features: np.ndarray) -> np.ndarray:
    """Cosine distance between track EMA features and detection features."""
    if len(tracks) == 0 or len(features) == 0:
        return np.empty((len(tracks), len(features)))
    track_feats = np.stack([t.feature for t in tracks])  # (M, D), L2-normalized
    sims = track_feats @ features.T  # (M, N)
    return 1.0 - sims


class BYTETracker:
    """Per-camera tracker instance."""

    def __init__(
        self,
        camera_id: str,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        appearance_weight: float = 0.0,
        max_age: int = 30,
        n_init: int = 3,
    ) -> None:
        self.camera_id = str(camera_id)
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.match_thresh = float(match_thresh)
        # 0.0 => pure ByteTrack; >0 blends appearance cost into stage 1 (BoT-SORT).
        self.appearance_weight = float(appearance_weight)
        self.max_age = int(max_age)
        self.n_init = int(n_init)

        self.tracks: list[SingleTrack] = []
        self._next_id = 1
        self._frame_id = 0

    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: Sequence[Detection],
        features: np.ndarray | None = None,
        timestamp: float = 0.0,
    ) -> list[dict]:
        """Advance one frame; returns confirmed tracks as plain dicts."""
        self._frame_id += 1

        # Predict every existing track forward.
        for trk in self.tracks:
            trk.predict()

        dets = list(detections)
        det_features = features
        if det_features is not None and len(det_features) != len(dets):
            logger.warning("Feature count mismatch; ignoring embeddings this frame")
            det_features = None

        highs = [i for i, d in enumerate(dets) if d.confidence >= self.track_high_thresh]
        lows = [
            i
            for i, d in enumerate(dets)
            if self.track_low_thresh <= d.confidence < self.track_high_thresh
        ]

        # Stage 1 pool: ALL live tracks (confirmed AND tentative), per ByteTrack.
        pool = list(self.tracks)

        # ---------------- Stage 1: high-confidence association ---------- #
        matches_1, unmatched_pool, unmatched_dets_high = self._associate(
            pool, highs, dets, det_features, use_appearance=True
        )
        for trk_idx, det_idx in matches_1:
            feat = det_features[det_idx] if det_features is not None else None
            pool[trk_idx].update(dets[det_idx].bbox, dets[det_idx].confidence, feat, timestamp)

        # ---------------- Stage 2: low-confidence rescue ---------------- #
        # Only CONFIRMED tracks may be rescued; tentative ones die immediately.
        rescue_candidates = [pool[i] for i in unmatched_pool if pool[i].is_confirmed()]
        matches_2, _, _ = self._associate(rescue_candidates, lows, dets, None, use_appearance=False)
        for trk_idx, det_idx in matches_2:
            rescue_candidates[trk_idx].update(
                dets[det_idx].bbox, dets[det_idx].confidence, None, timestamp
            )

        matched_stage2 = {id(rescue_candidates[i]) for i, _ in matches_2}
        matched_stage1 = {id(pool[r]) for r, _ in matches_1}

        # Age out: unmatched tentative tracks are dropped at once; unmatched
        # confirmed tracks survive until max_age frames without a hit.
        alive: list[SingleTrack] = []
        for trk in pool:
            if id(trk) not in matched_stage1 and id(trk) not in matched_stage2:
                trk.mark_missed()
            if not trk.is_deleted():
                alive.append(trk)

        # New tentative tracks from confident unmatched detections only.
        for det_idx in unmatched_dets_high:
            if dets[det_idx].confidence >= self.new_track_thresh:
                feat = det_features[det_idx] if det_features is not None else None
                trk = SingleTrack(
                    track_id=self._next_id,
                    bbox=dets[det_idx].bbox,
                    confidence=dets[det_idx].confidence,
                    n_init=self.n_init,
                    max_age=self.max_age,
                )
                if feat is not None:
                    trk._update_feature(feat)
                trk.history.append((timestamp, dets[det_idx].bbox))
                self._next_id += 1
                alive.append(trk)

        self.tracks = alive

        return self._collect_output()

    # ------------------------------------------------------------------ #
    def _associate(
        self,
        tracks: list[SingleTrack],
        det_indices: list[int],
        detections: Sequence[Detection],
        features: np.ndarray | None,
        use_appearance: bool,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian assignment; returns (matches, unmatched_track_idx, unmatched_det_idx)."""
        if len(tracks) == 0:
            return [], [], list(range(len(det_indices)))
        if len(det_indices) == 0:
            return [], list(range(len(tracks))), []

        sub_dets = [detections[i] for i in det_indices]
        cost = iou_distance(tracks, sub_dets)

        if (
            use_appearance
            and self.appearance_weight > 0.0
            and features is not None
            and all(t.feature is not None for t in tracks)
        ):
            app_cost = embedding_distance(tracks, features)
            lam = self.appearance_weight
            cost = lam * cost + (1.0 - lam) * app_cost

        row, col = linear_sum_assignment(cost)

        matches: list[tuple[int, int]] = []
        matched_rows: set[int] = set()
        matched_cols: set[int] = set()
        for r, c in zip(row, col, strict=False):
            if cost[r, c] <= self.match_thresh:
                matches.append((r, det_indices[c]))
                matched_rows.add(r)
                matched_cols.add(c)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_rows]
        unmatched_dets = [j for j in range(len(det_indices)) if j not in matched_cols]
        return matches, unmatched_tracks, unmatched_dets

    def _collect_output(self) -> list[dict]:
        out = []
        for trk in self.tracks:
            if trk.is_confirmed() and trk.time_since_update == 0:
                x1, y1, x2, y2 = trk.bbox
                out.append(
                    {
                        "camera_id": self.camera_id,
                        "track_id": trk.track_id,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": trk.confidence,
                        "feature": trk.feature,
                        "hits": trk.hits,
                    }
                )
        return out
