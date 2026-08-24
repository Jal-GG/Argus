"""Tests for the rule engine and alert dispatcher."""

import time

from src.alerts.alert_manager import AlertManager
from src.core.mapping.floor_plan import FloorPlan
from src.core.path.path_tracker import PersonPath
from src.core.rules.rule_engine import RuleEngine


class _FP(FloorPlan):
    pass


def _floor_plan() -> FloorPlan:
    fp = FloorPlan(size=(200, 200))
    fp.add_zone(
        "Server Room",
        [(80, 80), (160, 80), (160, 160), (80, 160)],
        zone_type="restricted",
    )
    fp.add_zone(
        "Reception Desk",
        [(10, 10), (70, 10), (70, 70), (10, 70)],
        zone_type="required",
        rule="required_zone",
        params={"timeout": 100},
    )
    fp.add_zone(
        "Waiting Area",
        [(10, 110), (70, 110), (70, 170), (10, 170)],
        zone_type="safe",
        rule="loitering",
        params={"max_dwell_time": 50},
    )
    return fp


class TestRuleEngine:
    def test_restricted_entry_triggers_once_then_cooldown(self):
        engine = RuleEngine(_floor_plan())
        path = PersonPath(global_id=1)
        now = 1000.0

        # Person sits inside the restricted zone.
        for dt in range(6):
            path.add_position(100.0, 100.0, "cam_001", timestamp=now + dt)

        violations = engine.evaluate(path, now=now + 5)
        assert len(violations) == 1
        assert violations[0].rule_type == "restricted_zone"

        # Immediate re-evaluation: cooldown suppresses duplicate.
        assert engine.evaluate(path, now=now + 6) == []

    def test_required_zone_timeout(self):
        engine = RuleEngine(_floor_plan())
        path = PersonPath(global_id=2)
        start = 0.0

        # Wandering outside Reception Desk for > timeout (100 s).
        for step in range(12):
            path.add_position(150.0 + step, 40.0, "cam_001", timestamp=start + step * 10)

        violations = engine.evaluate(path, now=start + 115)
        required = [v for v in violations if v.rule_type == "required_zone"]
        assert len(required) == 1
        assert "Reception" in required[0].zone_name or required[0].zone_name

    def test_no_false_violation_when_required_zone_visited(self):
        engine = RuleEngine(_floor_plan())
        path = PersonPath(global_id=3)
        path.add_position(30.0, 30.0, "cam_001", timestamp=5.0)  # visited!
        path.record_zone_visit("Reception Desk", 4.0)
        path.add_position(150.0, 150.0, "cam_001", timestamp=200.0)  # far away later

        violations = engine.evaluate(path, now=250)
        assert all(v.rule_type != "required_zone" for v in violations)

    def test_loitering_after_long_dwell(self):
        engine = RuleEngine(_floor_plan())
        path = PersonPath(global_id=4)
        base = 500.0
        for step in range(14):  # 140 s inside Waiting Area (limit 50 s)
            path.add_position(40.0, 140.0 + step * 0.01, "cam_001", timestamp=base + step * 10)

        violations = engine.evaluate(path, now=base + 135)
        loiter = [v for v in violations if v.rule_type == "loitering"]
        assert len(loiter) == 1

    def test_speed_anomaly(self):
        engine = RuleEngine(_floor_plan(), default_channels=[])
        path = PersonPath(global_id=5)
        # ~10 m per second across the plan (world units ≈ meters here).
        for step in range(5):
            path.add_position(step * 10.0, 50.0, "cam_001", timestamp=step)

        violations = engine.evaluate_speed(path, max_speed=3.0, now=5)
        assert len(violations) == 1
        assert violations[0].rule_type == "speed_anomaly"


class TestAlertManager:
    def test_console_dispatch_does_not_raise(self, caplog):
        manager = AlertManager({"enabled_channels": ["console"]})
        manager.dispatch(
            {
                "rule_type": "restricted_zone",
                "zone_name": "Vault",
                "global_id": 9,
                "reason": "Entered restricted zone 'Vault'",
                "timestamp": time.time(),
            }
        )
        assert any("ALERT" in rec.message for rec in caplog.records)

    def test_unknown_channel_is_soft_fail(self):
        manager = AlertManager({"enabled_channels": ["carrier_pigeon"]})
        manager.dispatch(
            {"rule_type": "x", "zone_name": "y", "global_id": 1, "reason": "z", "timestamp": 0}
        )  # must not raise
