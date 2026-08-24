"""Constant-velocity Kalman filter on bounding boxes.

State vector (8-dim), following the DeepSORT/ByteTrack formulation::

    [cx, cy, a, h, vcx, vcy, va, vh]

where (cx, cy) is box center, ``a`` is aspect ratio (w/h), ``h`` is height,
and the trailing four entries are their velocities.
"""

from __future__ import annotations

import numpy as np


class KalmanFilterXYAH:
    """Kalman filter for ``(center-x, center-y, aspect, height)`` boxes."""

    def __init__(self) -> None:
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a new track state from an unassociated measurement."""
        mean_pos = measurement.astype(float)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        h = float(measurement[3])
        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                1e-2,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                1e-5,
                10 * self._std_weight_velocity * h,
            ]
        )
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the prediction step (constant-velocity model)."""
        h = float(mean[3])

        motion_std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ]
        )
        motion_cov = np.diag(np.square(motion_std))

        mean = mean.copy()
        mean[:4] += mean[4:]  # x' = F x with identity F over one frame

        return mean, covariance + motion_cov

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space."""
        h = float(mean[3])
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ]
        )
        innovation_cov = np.diag(np.square(std))

        mean_proj = mean[:4].copy()
        cov_proj = covariance[:4, :4].copy()
        return mean_proj, cov_proj + innovation_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct the predicted state with a new measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol = np.linalg.cholesky(projected_cov)
        # K = P[:, :4] @ S^{-1}; solved via Cholesky of S (projected covariance).
        kalman_gain = np.linalg.solve(chol.T, np.linalg.solve(chol, covariance[:, :4].T)).T

        innovation = measurement.astype(float) - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    # ------------------------------------------------------------------ #
    @staticmethod
    def to_bbox(mean: np.ndarray) -> tuple[float, float, float, float]:
        """Convert state mean to an ``(x1, y1, x2, y2)`` bbox."""
        cx, cy, a, h = mean[:4]
        w = a * h
        return (
            float(cx - w / 2.0),
            float(cy - h / 2.0),
            float(cx + w / 2.0),
            float(cy + h / 2.0),
        )

    @staticmethod
    def from_bbox(bbox: tuple[float, float, float, float]) -> np.ndarray:
        """Convert an ``(x1, y1, x2, y2)`` bbox to a measurement vector."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        return np.array([cx, cy, w / h, h], dtype=float)

    @staticmethod
    def gating_distance(
        mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        """Squared Mahalanobis distance from state to each measurement."""
        projected_mean, projected_cov = KalmanFilterXYAH.project(mean, covariance)
        diff = measurements - projected_mean

        chol = np.linalg.cholesky(projected_cov)
        white = np.linalg.solve(chol, diff.T).T
        return np.sum(np.square(white), axis=1)
