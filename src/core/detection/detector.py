"""Detector protocol + factory.

Selects the best available backend:
  1. YOLO11/12 via ``ultralytics`` (GPU or CPU)
  2. OpenCV HOG person descriptor (dependency-light fallback)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ...utils.logger import get_logger
from ..types import Detection

logger = get_logger(__name__)


@runtime_checkable
class PersonDetector(Protocol):
    """Any backend able to detect people in BGR frames."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...

    def close(self) -> None: ...


def create_detector(
    backend: str = "auto",
    model_path: str = "yolo11m.pt",
    confidence_threshold: float = 0.5,
    device: str | None = None,
    input_size: int = 640,
) -> PersonDetector:
    """Build the highest-quality detector available for this machine.

    Args:
        backend: ``"auto"`` | ``"yolo"`` | ``"hog"``.
        model_path: YOLO weights name/path (e.g. ``yolo11m.pt``, ``yolov8n.pt``).
        device: ``"cuda"``, ``"cpu"``, ``"mps"`` or ``None`` for auto.
    """
    backend = backend.lower()
    if backend in ("auto", "yolo"):
        try:
            from .yolo_detector import YOLODetector

            det = YOLODetector(
                model_path=model_path,
                conf_threshold=confidence_threshold,
                device=device,
                input_size=input_size,
            )
            logger.info("Detector backend: YOLO (%s)", model_path)
            return det
        except ImportError as exc:
            if backend == "yolo":
                raise
            logger.warning("ultralytics unavailable (%s); falling back to HOG", exc)

    from .hog_detector import HOGPersonDetector

    det = HOGPersonDetector(conf_threshold=confidence_threshold)
    logger.info("Detector backend: OpenCV HOG (fallback)")
    return det
