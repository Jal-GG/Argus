"""Pixel → floor-plan world coordinate mapping via per-camera homography."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from ...utils.config import load_yaml
from ...utils.geometry import bbox_bottom_center
from ...utils.logger import get_logger

logger = get_logger(__name__)


class CoordinateMapper:
    """Holds one 3x3 homography per camera and converts coordinates.

    Calibration YAML format::

        camera_id: cam_001
        floor_id: floor_0
        homography_matrix:
          - [1.0, 0.0, 0.0]
          - [0.0, 1.0, 0.0]
          - [0.0, 0.0, 1.0]
        scale_meters_per_pixel: 0.05   # optional, for metric speeds
    """

    def __init__(self) -> None:
        self.calibrations: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    def load_calibration(self, camera_id: str, calibration_path: str | Path) -> None:
        data = load_yaml(calibration_path)
        H = np.asarray(data["homography_matrix"], dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"Invalid homography shape {H.shape} in {calibration_path}")
        self.calibrations[str(camera_id)] = {
            "H": H,
            "floor_id": str(data.get("floor_id", "floor_0")),
            "scale": float(data.get("scale_meters_per_pixel", 1.0)),
        }
        logger.info(
            "Loaded calibration for %s (floor=%s)",
            camera_id,
            self.calibrations[str(camera_id)]["floor_id"],
        )

    def register_homography(
        self, camera_id: str, H: np.ndarray, floor_id: str = "floor_0", scale: float = 1.0
    ) -> None:
        self.calibrations[str(camera_id)] = {
            "H": np.asarray(H, dtype=np.float64),
            "floor_id": floor_id,
            "scale": float(scale),
        }

    def is_calibrated(self, camera_id: str) -> bool:
        return str(camera_id) in self.calibrations

    @property
    def camera_positions(self) -> dict[str, tuple[float, float]]:
        """Approximate each camera's position on the plan via image-center projection."""
        positions: dict[str, tuple[float, float]] = {}
        for cam_id in self.calibrations:
            try:
                positions[cam_id] = tuple(self.pixel_to_world(cam_id, 320.0, 480.0))
            except Exception:  # pragma: no cover - degenerate H
                continue
        return positions

    # ------------------------------------------------------------------ #
    def pixel_to_world(self, camera_id: str, x: float, y: float) -> tuple[float, float]:
        H = self._require_H(camera_id)
        pt = np.array([[[float(x), float(y)]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, H)
        wx, wy = out[0, 0]
        return float(wx), float(wy)

    def world_to_pixel(self, camera_id: str, X: float, Y: float) -> tuple[float, float]:
        H = self._require_H(camera_id)
        H_inv = np.linalg.inv(H)
        pt = np.array([[[float(X), float(Y)]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, H_inv)
        px, py = out[0, 0]
        return float(px), float(py)

    def bbox_to_world(self, camera_id: str, bbox: Sequence[float]) -> tuple[float, float]:
        """Project a person's ground-contact point onto the floor plan."""
        gx, gy = bbox_bottom_center(bbox)
        return self.pixel_to_world(camera_id, gx, gy)

    def floor_of(self, camera_id: str) -> str:
        return self.calibrations.get(str(camera_id), {}).get("floor_id", "floor_0")

    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_homography(
        image_points: Sequence[Sequence[float]], world_points: Sequence[Sequence[float]]
    ) -> np.ndarray:
        """DLT with RANSAC from >=4 point correspondences."""
        if len(image_points) < 4 or len(image_points) != len(world_points):
            raise ValueError("Need at least 4 corresponding points")
        H, _ = cv2.findHomography(
            np.asarray(image_points, dtype=np.float64),
            np.asarray(world_points, dtype=np.float64),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if H is None:
            raise ValueError("Homography estimation failed (degenerate points?)")
        return H

    def _require_H(self, camera_id: str) -> np.ndarray:
        calib = self.calibrations.get(str(camera_id))
        if calib is None:
            raise KeyError(f"Camera '{camera_id}' is not calibrated")
        return calib["H"]
