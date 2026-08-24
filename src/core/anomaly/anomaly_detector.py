"""Statistical movement-anomaly detection.

Two detectors, both online (no training data needed):

* :class:`SpeedAnomalyDetector` — rolling z-score on each person's own speed
  history; flags sudden bursts far outside their personal baseline
  (running, teleport-glitches, homography errors).
* :class:`DwellAnomalyDetector` — flags people whose current dwell time in a
  zone exceeds the population's learned mean by k·sigma (unusual lingering).

Violations reuse :class:`~src.core.rules.rule_engine.RuleViolation` so they
flow through the existing alert/event plumbing.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..rules.rule_engine import RuleViolation


@dataclass
class _Rolling:
    values: deque[float] = field(default_factory=lambda: deque(maxlen=60))

    def push(self, v: float) -> None:
        self.values.append(float(v))

    def stats(self) -> tuple[float, float]:
        n = len(self.values)
        if n < 5:
            return 0.0, 0.0
        mean = sum(self.values) / n
        var = sum((v - mean) ** 2 for v in self.values) / n
        return mean, math.sqrt(var)


class SpeedAnomalyDetector:
    """Per-person adaptive speed baseline."""

    def __init__(
        self,
        z_threshold: float = 3.5,
        min_baseline_s: float = 8.0,
        cooldown_s: float = 20.0,
        absolute_floor: float = 2.2,
        window: int = 60,
    ) -> None:
        """
        ``absolute_floor``: never flag below this speed regardless of z-score —
        avoids flagging calm people whose variance is tiny.
        """
        self.z_threshold = float(z_threshold)
        self.min_baseline_s = float(min_baseline_s)
        self.cooldown_s = float(cooldown_s)
        self.absolute_floor = float(absolute_floor)
        self._history: dict[int, _Rolling] = defaultdict(lambda: _Rolling(deque(maxlen=window)))
        self._first_seen: dict[int, float] = {}
        self._last_alert: dict[int, float] = {}

    def observe(self, path, now: float | None = None) -> list[RuleViolation]:
        """Feed the person's current speed; returns violations if any."""
        now = now if now is not None else time.time()
        gid = path.global_id
        speed = path.speed()

        if gid not in self._first_seen:
            self._first_seen[gid] = now

        rolling = self._history[gid]
        mean, std = rolling.stats()
        baseline_ready = (
            len(rolling.values) >= 6
            and (now - self._first_seen[gid]) >= self.min_baseline_s
            and std > 1e-3
        )

        violations: list[RuleViolation] = []
        if baseline_ready:
            z = (speed - mean) / std
            last_alert = self._last_alert.get(gid, float("-inf"))
            if (
                z >= self.z_threshold
                and speed >= self.absolute_floor
                and (now - last_alert) >= self.cooldown_s
            ):
                pos = path.current_position or (0.0, 0.0, "")
                violations.append(
                    RuleViolation(
                        rule_type="anomaly_speed",
                        zone_name="*",
                        global_id=gid,
                        reason=(
                            f"Speed burst {speed:.2f} ({z:.1f}sigma above personal mean {mean:.2f})"
                        ),
                        timestamp=now,
                        position=(pos[0], pos[1]),
                    )
                )
                self._last_alert[gid] = now

        rolling.push(speed)
        return violations


class DwellAnomalyDetector:
    """Flags dwell times far above what a zone usually sees."""

    def __init__(
        self, z_threshold: float = 3.0, min_samples: int = 8, cooldown_s: float = 120.0
    ) -> None:
        self.z_threshold = float(z_threshold)
        self.min_samples = int(min_samples)
        self.cooldown_s = float(cooldown_s)
        self._zone_stats: dict[str, _Rolling] = defaultdict(_Rolling)
        self._zone_first_seen: dict[str, float] = {}
        self._active_dwell_start: dict[tuple[int, str], float] = {}
        self._last_alert: dict[tuple[int, str], float] = {}

    def enter(self, global_id: int, zone_name: str, ts: float) -> None:
        self._active_dwell_start.setdefault((global_id, zone_name), ts)
        self._zone_first_seen.setdefault(zone_name, ts)

    def exit(self, global_id: int, zone_name: str, ts: float) -> None:
        start = self._active_dwell_start.pop((global_id, zone_name), None)
        if start is not None:
            self._zone_stats[zone_name].push(ts - start)

    def check_active(self, global_id: int, zone_name: str, ts: float) -> list[RuleViolation]:
        """Evaluate the *ongoing* dwell against the zone's history."""
        start = self._active_dwell_start.get((global_id, zone_name))
        if start is None:
            return []
        rolling = self._zone_stats[zone_name]
        if len(rolling.values) < self.min_samples:
            return []

        mean, std = rolling.stats()
        if std <= 1e-3:
            return []

        current = ts - start
        z = (current - mean) / std
        key = (global_id, zone_name)
        last_alert = self._last_alert.get(key, float("-inf"))
        if z >= self.z_threshold and ts - last_alert >= self.cooldown_s:
            self._last_alert[key] = ts
            return [
                RuleViolation(
                    rule_type="anomaly_dwell",
                    zone_name=zone_name,
                    global_id=global_id,
                    reason=(
                        f"Dwelling {current:.0f}s "
                        f"({z:.1f}sigma above typical {mean:.0f}s for '{zone_name}')"
                    ),
                    timestamp=ts,
                )
            ]
        return []
