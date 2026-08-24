"""YOLO11/YOLOv8 person detection via ultralytics."""

from __future__ import annotations

import numpy as np

from ...utils.logger import get_logger
from ..types import Detection

logger = get_logger(__name__)


class YOLODetector:
    """Ultralytics-based person detector (COCO class 0)."""

    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str | None = None,
        input_size: int = 640,
        half_precision: bool | None = None,
    ) -> None:
        from ultralytics import YOLO  # lazy import; may be absent on CPU-only boxes

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = int(input_size)

        import torch

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ultralytics accepts 'cpu'/'cuda'/'mps' directly.
        self.device: str | int = resolved_device

        self.model = YOLO(model_path)
        # Note: ultralytics applies AMP/precision handling automatically.

        # Warm up so first real frame doesn't pay compilation cost.
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        try:
            self.model.predict(
                dummy,
                classes=[0],
                conf=0.01,
                device=self.device,
                verbose=False,
                imgsz=self.input_size,
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("YOLO warmup failed (%s); continuing anyway", exc)

        logger.info("YOLO loaded: %s on %s", model_path, resolved_device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            classes=[0],  # person only
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.input_size,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs, strict=False):
                x1, y1, x2, y2 = (float(v) for v in box)
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=float(conf)))
        return detections

    def close(self) -> None:
        del self.model
