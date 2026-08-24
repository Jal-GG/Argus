"""OpenCV drawing helpers for frames and floor-plan overlays."""

from __future__ import annotations

import cv2
import numpy as np


def _id_color(gid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng((int(gid) * 2654435761) % (2**32))
    r, g, b = (int(v) for v in rng.integers(60, 255, 3))
    return b, g, r


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            frame,
            f"{det.confidence:.2f}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_tracks(frame: np.ndarray, tracks: list, show_global_ids: bool = True) -> np.ndarray:
    """Draw confirmed tracks with local and global IDs."""
    for trk in tracks:
        x1, y1, x2, y2 = (int(v) for v in trk.bbox)
        color = _id_color(trk.global_id) if show_global_ids and trk.global_id else (0, 220, 220)

        label = f"T{trk.track_id}"
        if show_global_ids and trk.global_id is not None:
            label = f"G{trk.global_id} ({label})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 22), (x1 + 10 * len(label), y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame
