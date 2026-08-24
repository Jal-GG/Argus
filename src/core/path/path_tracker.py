"""Per-person trajectory aggregation on the floor plan."""

from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass, field

from ...utils.geometry import euclidean
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PersonPath:
    """Trajectory of one global identity in world (floor-plan) coordinates."""

    global_id: int
    # (timestamp, x, y, camera_id)
    positions: deque[tuple[float, float, float, str]] = field(
        default_factory=lambda: deque(maxlen=2000)
    )
    zone_visits: dict[str, float] = field(default_factory=dict)  # zone -> seconds
    first_seen: float | None = None
    last_seen: float | None = None
    total_distance: float = 0.0
    status: str = "active"  # active | completed | suspicious

    def add_position(
        self, x: float, y: float, camera_id: str, timestamp: float | None = None
    ) -> None:
        ts = float(timestamp if timestamp is not None else time.time())
        if self.first_seen is None:
            self.first_seen = ts
        self.last_seen = ts

        if len(self.positions) > 0:
            last = self.positions[-1]
            # Guard against homography jitter across camera hand-offs.
            if last[3] == camera_id or euclidean((last[1], last[2]), (x, y)) < 50.0:
                self.total_distance += euclidean((last[1], last[2]), (x, y))

        self.positions.append((ts, float(x), float(y), str(camera_id)))

    def record_zone_visit(self, zone_name: str, dwell_seconds: float = 0.0) -> None:
        self.zone_visits[zone_name] = self.zone_visits.get(zone_name, 0.0) + dwell_seconds

    @property
    def current_position(self) -> tuple[float, float, str] | None:
        if not self.positions:
            return None
        _, x, y, cam = self.positions[-1]
        return x, y, cam

    def speed(self, window: int = 8) -> float:
        """Recent speed in world units/second over the last `window` samples."""
        recent = list(self.positions)[-max(window, 2) :]
        if len(recent) < 2:
            return 0.0
        dt = recent[-1][0] - recent[0][0]
        if dt <= 0:
            return 0.0
        dist = sum(euclidean((a[1], a[2]), (b[1], b[2])) for a, b in itertools.pairwise(recent))
        return dist / dt

    def path_points(self, max_points: int = 120) -> list[tuple[float, float]]:
        return [(x, y) for _, x, y, _ in list(self.positions)[-max_points:]]


class PathTracker:
    """Aggregates WorldObservations into PersonPath objects."""

    def __init__(self, stale_timeout: float = 300.0) -> None:
        self.active_paths: dict[int, PersonPath] = {}
        self.completed_paths: list[PersonPath] = []
        self.stale_timeout = float(stale_timeout)

    def update(
        self,
        global_id: int,
        world_x: float,
        world_y: float,
        camera_id: str,
        timestamp: float | None = None,
    ) -> PersonPath:
        path = self.active_paths.get(global_id)
        if path is None:
            path = PersonPath(global_id=global_id)
            self.active_paths[global_id] = path
        path.add_position(world_x, world_y, camera_id, timestamp)
        return path

    def get_path(self, global_id: int) -> PersonPath | None:
        return self.active_paths.get(global_id)

    def reap_stale(self, now: float | None = None) -> list[int]:
        """Archive paths idle longer than `stale_timeout`; returns reaped IDs."""
        now = now if now is not None else time.time()
        reaped: list[int] = []
        for gid, path in list(self.active_paths.items()):
            if path.last_seen is not None and (now - path.last_seen) > self.stale_timeout:
                path.status = "completed"
                self.completed_paths.append(self.active_paths.pop(gid))
                reaped.append(gid)
        return reaped
