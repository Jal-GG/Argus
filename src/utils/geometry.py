"""Geometry helpers shared by tracking, mapping, and rule modules."""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two boxes given as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def bbox_center(bbox: Sequence[float]) -> Point:
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_bottom_center(bbox: Sequence[float]) -> Point:
    """Ground-contact point of a person box — best anchor for homography."""
    x1, _, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return ((x1 + x2) / 2.0, float(y2))


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test (works without OpenCV)."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def path_length(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        total += euclidean(points[i - 1], points[i])
    return total
