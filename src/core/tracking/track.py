"""Single-object track with lifecycle management."""

from __future__ import annotations

from collections import deque

import numpy as np

from .kalman_filter import KalmanFilterXYAH

# Track states
TENTATIVE = 1  # not yet enough consecutive hits to be confirmed
CONFIRMED = 2  # stable, reported downstream
DELETED = 3  # removed


class SingleTrack:
    """One tracked person inside one camera view."""

    _kf = KalmanFilterXYAH()

    def __init__(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        confidence: float,
        n_init: int = 3,
        max_age: int = 30,
        ema_alpha: float = 0.9,
    ) -> None:
        self.track_id = int(track_id)
        self.mean, self.covariance = self._kf.initiate(KalmanFilterXYAH.from_bbox(bbox))
        self.confidence = float(confidence)

        self.n_init = int(n_init)
        self.max_age = int(max_age)
        self.ema_alpha = float(ema_alpha)

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = TENTATIVE

        self.feature: np.ndarray | None = None
        self.history: deque[tuple[float, tuple[float, float, float, float]]] = deque(maxlen=64)

    # ------------------------------------------------------------------ #
    def predict(self) -> None:
        """Advance motion model one frame."""
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        bbox: tuple[float, float, float, float],
        confidence: float,
        feature: np.ndarray | None = None,
        timestamp: float = 0.0,
    ) -> None:
        """Associate a detection with this track."""
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, KalmanFilterXYAH.from_bbox(bbox)
        )
        self.hits += 1
        self.time_since_update = 0
        self.confidence = 0.5 * self.confidence + 0.5 * float(confidence)

        if feature is not None:
            self._update_feature(feature)

        if self.state == TENTATIVE and self.hits >= self.n_init:
            self.state = CONFIRMED

        self.history.append((timestamp, bbox))

    def mark_missed(self) -> None:
        """No detection matched this frame."""
        if self.state == TENTATIVE or self.time_since_update > self.max_age:
            self.state = DELETED

    def is_tentative(self) -> bool:
        return self.state == TENTATIVE

    def is_confirmed(self) -> bool:
        return self.state == CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == DELETED

    # ------------------------------------------------------------------ #
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return KalmanFilterXYAH.to_bbox(self.mean)

    @property
    def tlwh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1

    # ------------------------------------------------------------------ #
    def _update_feature(self, feature: np.ndarray) -> None:
        """Exponentially-average appearance embedding for stability."""
        feature = feature / (np.linalg.norm(feature) + 1e-9)
        if self.feature is None:
            self.feature = feature
        else:
            blended = self.ema_alpha * self.feature + (1.0 - self.ema_alpha) * feature
            norm = np.linalg.norm(blended)
            self.feature = blended / norm if norm > 0 else feature
