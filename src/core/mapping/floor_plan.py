"""Floor-plan model with zone management, queries, and rendering."""

from __future__ import annotations

import time
from collections.abc import Sequence

import cv2
import numpy as np

from ...utils.geometry import point_in_polygon
from ...utils.logger import get_logger

logger = get_logger(__name__)

ZONE_COLORS: dict[str, tuple[int, int, int]] = {
    "required": (0, 200, 0),  # green
    "restricted": (0, 0, 220),  # red
    "safe": (220, 200, 0),  # cyan-ish
    "custom": (140, 140, 140),  # gray
}


class FloorPlan:
    """A building floor plan image plus its annotated zones."""

    def __init__(
        self,
        image_path: str | None = None,
        size: tuple[int, int] = (800, 600),
        floor_id: str = "floor_0",
    ) -> None:
        self.floor_id = floor_id
        if image_path is not None:
            self.image = cv2.imread(str(image_path))
            if self.image is None:
                raise FileNotFoundError(f"Floor plan not found: {image_path}")
        else:
            # Blank canvas so demos run without real estate imagery.
            self.image = np.full((size[1], size[0], 3), 245, dtype=np.uint8)

        self.zones: list[dict] = []

    # ------------------------------------------------------------------ #
    def add_zone(
        self,
        name: str,
        polygon: Sequence[Sequence[float]],
        zone_type: str = "custom",
        rule: str | None = None,
        params: dict | None = None,
        alert_channels: list | None = None,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        pts = [(float(x), float(y)) for x, y in polygon]
        self.zones.append(
            {
                "name": name,
                "type": zone_type,
                "polygon": np.array(pts, dtype=np.float32),
                "rule": rule,
                "params": params or {},
                "alert_channels": alert_channels,
                "color": color or ZONE_COLORS.get(zone_type, ZONE_COLORS["custom"]),
            }
        )

    def zones_by_type(self, zone_type: str) -> list[dict]:
        return [z for z in self.zones if z["type"] == zone_type]

    def find_zone(self, name: str) -> dict | None:
        for z in self.zones:
            if z["name"] == name:
                return z
        return None

    # ------------------------------------------------------------------ #
    def check_point_in_zone(self, x: float, y: float) -> list[dict]:
        return [z for z in self.zones if point_in_polygon((x, y), z["polygon"])]

    def dwell_time_in_zone(
        self,
        name: str,
        positions: Sequence[tuple[float, float, float]],
        current_time: float | None = None,
    ) -> float:
        """Accumulated seconds inside zone `name` given (t, x, y) samples."""
        zone = self.find_zone(name)
        if zone is None or len(positions) < 2:
            return 0.0

        total, enter_ts = 0.0, None
        for ts, x, y in positions:
            inside = point_in_polygon((x, y), zone["polygon"])
            if inside and enter_ts is None:
                enter_ts = ts
            elif not inside and enter_ts is not None:
                total += ts - enter_ts
                enter_ts = None
        if enter_ts is not None:
            end = current_time if current_time is not None else positions[-1][0]
            total += end - enter_ts
        return max(total, 0.0)

    # ------------------------------------------------------------------ #
    def render(
        self,
        tracks_world: dict[int, tuple[float, float]] | None = None,
        paths: dict[int, list[tuple[float, float]]] | None = None,
        alerts: dict[int, bool] | None = None,
    ) -> np.ndarray:
        """Draw zones, paths, and live positions onto the plan."""
        canvas = self.image.copy()

        overlay = canvas.copy()
        for zone in self.zones:
            pts = zone["polygon"].astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], zone["color"])
            cv2.polylines(canvas, [pts], True, zone["color"], 2)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        for zone in self.zones:
            centroid = zone["polygon"].mean(axis=0).astype(int)
            cv2.putText(
                canvas,
                f"{zone['name']} [{zone['type']}]",
                (int(centroid[0]) - 30, int(centroid[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )

        alerts = alerts or {}
        for gid, points in (paths or {}).items():
            if len(points) >= 2:
                arr = np.array(points[-120:], dtype=np.int32).reshape(-1, 1, 2)
                color = (0, 0, 255) if alerts.get(gid) else _id_color(gid)
                cv2.polylines(canvas, [arr], False, color, 2, cv2.LINE_AA)

        for gid, (x, y) in (tracks_world or {}).items():
            color = (0, 0, 255) if alerts.get(gid) else _id_color(gid)
            cv2.circle(canvas, (int(x), int(y)), 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"#{gid}",
                (int(x) + 8, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        stamp = time.strftime("%H:%M:%S")
        cv2.putText(
            canvas,
            f"Live · {len(tracks_world or {})} tracked · {stamp}",
            (12, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        return canvas


def _id_color(gid: int) -> tuple[int, int, int]:
    # Deterministic per-ID color (no hash randomization surprises).
    rng = np.random.default_rng((int(gid) * 2654435761) % (2**32))
    r, g, b = (int(v) for v in rng.integers(60, 255, 3))
    return b, g, r  # BGR
