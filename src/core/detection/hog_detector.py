"""OpenCV HOG person detector — zero-dependency fallback backend.

Accuracy is far below YOLO, but it keeps the whole pipeline runnable on
machines without torch/ultralytics and in CPU-only CI.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...utils.logger import get_logger
from ..types import Detection

logger = get_logger(__name__)


class HOGPersonDetector:
    """Histogram-of-Oriented-Gradients + linear-SVM person detector."""

    def __init__(
        self,
        conf_threshold: float = 0.5,
        win_stride: tuple[int, int] = (4, 4),
        scale: float = 1.03,
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.win_stride = win_stride
        self.scale = scale

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # HOG is slow on large frames; cap working resolution.
        h, w = frame.shape[:2]
        max_side = 960
        resize = max(h, w) > max_side
        if resize:
            factor = max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * factor), int(h * factor)))

        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            padding=(8, 8),
            scale=self.scale,
        )

        detections: list[Detection] = []
        if len(rects) == 0:
            return detections

        # Non-max suppression to merge overlapping windows.
        indices = cv2.dnn.NMSBoxes(
            [[x, y, bw, bh] for x, y, bw, bh in rects],
            [float(s) for s in weights],
            score_threshold=0.0,
            nms_threshold=0.5,
        )
        keep = np.array(indices).flatten() if len(indices) else range(len(rects))

        inv = 1.0 / factor if resize else 1.0
        for i in keep:
            x, y, bw, bh = rects[i]
            score = float(weights[i])
            if score < self.conf_threshold:
                continue
            detections.append(
                Detection(
                    bbox=(
                        float(x) * inv,
                        float(y) * inv,
                        float(x + bw) * inv,
                        float(y + bh) * inv,
                    ),
                    confidence=score,
                )
            )
        return detections

    def close(self) -> None:
        del self.hog
