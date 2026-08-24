"""Declarative zone rules evaluated against live person paths.

Rule types (configured per zone in ``config/zones.yaml``):
  - ``required_zone``   : person must visit within `timeout` seconds
  - ``restricted_zone`` : entry forbidden (optionally role-gated)
  - ``loitering``       : dwell time exceeds `max_dwell_time`
  - ``speed_anomaly``   : instantaneous speed exceeds `max_speed`
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ...utils.logger import get_logger
from ..mapping.floor_plan import FloorPlan
from ..path.path_tracker import PersonPath

logger = get_logger(__name__)


@dataclass(slots=True)
class RuleViolation:
    rule_type: str
    zone_name: str
    global_id: int
    reason: str
    timestamp: float
    position: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "rule_type": self.rule_type,
            "zone_name": self.zone_name,
            "global_id": self.global_id,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "position": self.position,
        }


class RuleEngine:
    """Evaluates all enabled zone rules each tick."""

    def __init__(
        self,
        floor_plan: FloorPlan,
        default_channels: list[str] | None = None,
        strict_unknown_roles: bool = False,
    ) -> None:
        self.floor_plan = floor_plan
        self.default_channels = default_channels or ["console"]
        self.strict_unknown_roles = bool(strict_unknown_roles)
        # Per-(person, rule) state for one-shot alerting / cooldowns.
        self._last_violation_ts: dict[tuple[int, str], float] = {}
        self.cooldown = 30.0

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        path: PersonPath,
        now: float | None = None,
        person_name: str | None = None,
        person_roles: tuple[str, ...] = (),
    ) -> list[RuleViolation]:
        """Evaluate zone rules.

        ``person_name``/``person_roles`` come from the identity stage (Phase
        11). When face recognition is off they stay empty and role-gated
        zones fall back to plain restricted-zone semantics unless
        ``strict_unknown_roles`` is set.
        """
        now = now if now is not None else time.time()
        violations: list[RuleViolation] = []

        pos = path.current_position
        if pos is None:
            return violations
        x, y, _cam = pos

        zones_here = self.floor_plan.check_point_in_zone(x, y)
        visited_names = {z["name"] for z in zones_here}

        for zone in self.floor_plan.zones:
            rule_type = zone.get("rule") or _implicit_rule(zone["type"])
            params = zone.get("params", {}) or {}
            key = (path.global_id, f"{rule_type}:{zone['name']}")

            if rule_type == "restricted_zone" and zone["name"] in visited_names:
                allowed_roles = [str(r).lower() for r in params.get("allowed_roles", [])]
                reason: str | None = None

                if not allowed_roles:
                    reason = f"Entered restricted zone '{zone['name']}'"
                else:
                    role_set = {r.lower() for r in person_roles}
                    known = bool(person_name) and (bool(role_set) or not self.strict_unknown_roles)
                    if known and (role_set & set(allowed_roles)):
                        pass  # authorized staff — no alert
                    else:
                        who = person_name or f"unrecognized #{path.global_id}"
                        reason = (
                            f"{who} entered role-gated zone "
                            f"'{zone['name']}' (requires {allowed_roles})"
                        )

                if reason and self._cooldown_ok(key, now):
                    violations.append(
                        RuleViolation(
                            "restricted_zone",
                            zone["name"],
                            path.global_id,
                            reason,
                            now,
                            (x, y),
                        )
                    )

            elif rule_type == "required_zone":
                timeout = float(params.get("timeout", 300))
                elapsed = (now - path.first_seen) if path.first_seen is not None else 0.0
                if (
                    zone["name"] not in path.zone_visits
                    and elapsed > timeout
                    and self._cooldown_ok(key, now, cooldown=max(timeout, self.cooldown))
                ):
                    violations.append(
                        RuleViolation(
                            "required_zone",
                            zone["name"],
                            path.global_id,
                            f"Did not reach required zone '{zone['name']}' within {timeout:.0f}s",
                            now,
                            (x, y),
                        )
                    )

            elif rule_type == "loitering" and zone["name"] in visited_names:
                max_dwell = float(params.get("max_dwell_time", 600))
                dwell = self.floor_plan.dwell_time_in_zone(
                    zone["name"],
                    [(t, px, py) for t, px, py, _ in path.positions],
                    current_time=now,
                )
                if dwell > max_dwell and self._cooldown_ok(key, now, cooldown=max_dwell * 0.5):
                    violations.append(
                        RuleViolation(
                            "loitering",
                            zone["name"],
                            path.global_id,
                            f"Loitering {dwell:.0f}s > {max_dwell:.0f}s",
                            now,
                            (x, y),
                        )
                    )

        return violations

    def evaluate_speed(
        self, path: PersonPath, max_speed: float, now: float | None = None
    ) -> list[RuleViolation]:
        now = now if now is not None else time.time()
        speed = path.speed()
        if speed > max_speed:
            key = (path.global_id, "speed_anomaly:*")
            if self._cooldown_ok(key, now):
                pos = path.current_position or (0.0, 0.0, "")
                return [
                    RuleViolation(
                        "speed_anomaly",
                        "*",
                        path.global_id,
                        f"Speed {speed:.2f} exceeds limit {max_speed:.2f}",
                        now,
                        (pos[0], pos[1]),
                    )
                ]
        return []

    # ------------------------------------------------------------------ #
    def _cooldown_ok(self, key: tuple[int, str], now: float, cooldown: float | None = None) -> bool:
        last = self._last_violation_ts.get(key)
        effective = cooldown if cooldown is not None else self.cooldown
        if last is not None and (now - last) < effective:
            return False
        self._last_violation_ts[key] = now
        return True


def _implicit_rule(zone_type: str) -> str | None:
    """Zones without explicit rules default by their type."""
    mapping = {"restricted": "restricted_zone", "required": "required_zone"}
    return mapping.get(zone_type)
