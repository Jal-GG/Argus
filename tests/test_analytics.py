"""Tests for heatmap accumulation/rendering, predictor, and anomaly detectors."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import itertools

from src.core.anomaly.anomaly_detector import (
    DwellAnomalyDetector,
    SpeedAnomalyDetector,
)
from src.core.mapping.heatmap import Heatmap
from src.core.path.predictor import VelocityPredictor, ZoneMarkov


class TestHeatmap:
    def test_accumulation_and_decay(self):
        hm = Heatmap(200, 100, cell_size=10, decay_per_minute=0.5)
        for _ in range(5):
            hm.add_point(105, 55, ts=0.0)
        assert hm.grid[5, 10] == pytest.approx(5.0)

        mass_before = hm.total_mass
        hm.add_point(50, 50, ts=60.0)  # one minute later -> decay applied
        assert hm.total_mass < mass_before + 2.0  # old mass halved-ish

    def test_points_clamped_to_grid(self):
        hm = Heatmap(100, 100, cell_size=20)
        hm.add_point(-500, 99999, ts=0.0)  # out of bounds — must not crash
        assert hm.grid[-1, 0] == pytest.approx(1.0)  # clamped to corner cell

    def test_render_overlay_shape(self):
        base = np.full((120, 160, 3), 240, dtype=np.uint8)
        hm = Heatmap(160, 120, cell_size=8)
        hm.add_points([(80, 60)] * 30)
        out = hm.render_overlay(base)
        assert out.shape == base.shape
        assert not np.array_equal(out, base)  # overlay changed pixels
        # Empty map returns the base untouched.
        empty = Heatmap(160, 120).render_overlay(base.copy())
        assert np.array_equal(empty, base)

    def test_top_cells_returns_hotspots(self):
        hm = Heatmap(200, 200, cell_size=20)
        hm.add_points([(30, 30)] * 10)
        hm.add_points([(170, 170)] * 3)
        hot = hm.top_cells(2)
        assert len(hot) == 2
        assert hot[0] == (30, 30)  # hottest first


class _FakePath:
    """Duck-typed PersonPath for detector tests."""

    def __init__(self, gid, samples):
        self.global_id = gid
        self._samples = samples  # [(t, x, y)]

    @property
    def current_position(self):
        _t, x, y = self._samples[-1]
        return x, y, "cam"

    def speed(self, window=8):
        recent = self._samples[-window:]
        if len(recent) < 2:
            return 0.0
        dt = recent[-1][0] - recent[0][0]
        dist = sum(np.hypot(b[1] - a[1], b[2] - a[2]) for a, b in itertools.pairwise(recent))
        return dist / max(dt, 1e-9)


class TestVelocityPredictor:
    def test_predicts_straight_motion(self):
        pred = VelocityPredictor(horizon_s=2.0, steps=4)
        t = 0.0
        for i in range(20):  # steady 2 units/s to the right
            pred.observe(1, x=10 * i / 10.0, y=0.0, ts=i * 0.5)
            t += 0.5

        result = pred.predict(1)
        assert result is not None
        end = result.points[-1]
        last_x = 19.0
        assert end[0] > last_x + 2.0  # extrapolates beyond last seen
        assert abs(end[1]) < 1.0  # no lateral drift

    def test_stationary_person_no_prediction(self):
        pred = VelocityPredictor()
        for i in range(15):
            pred.observe(2, x=5.0, y=5.0, ts=i * 0.5)
        assert pred.predict(2) is None

    def test_unknown_and_forget(self):
        pred = VelocityPredictor()
        assert pred.predict(99) is None
        pred.observe(3, 0, 0, ts=0)
        pred.observe(3, 3, 0, ts=1)  # 3 u/s — within max_speed
        assert pred.predict(3) is not None
        pred.forget(3)
        assert pred.predict(3) is None

    def test_teleport_guard_ignores_jump(self):
        pred = VelocityPredictor()
        for i in range(10):  # steady ~1 u/s
            pred.observe(4, x=float(i), y=0, ts=float(i))
        pred.observe(4, x=5000, y=0, ts=11)  # impossible jump: v≈454 u/s
        result = pred.predict(4)
        # Anchor moves to last seen point, but velocity must NOT absorb the jump.
        assert result.points[-1][0] < 5010


class TestZoneMarkov:
    def test_learned_transition_prediction(self):
        markov = ZoneMarkov()
        for _ in range(6):
            markov.observe_sequence(["Entrance", "Lobby", "Reception"])
        assert markov.next_zone("Lobby") == "Reception"
        probs = markov.transition_probs("Lobby")
        assert probs["Reception"] == pytest.approx(1.0)

    def test_unknown_zone_returns_none(self):
        assert ZoneMarkov().next_zone("Nowhere") is None


class TestSpeedAnomaly:
    @staticmethod
    def _calm_samples(noise: bool = True) -> list[tuple[float, float, float]]:
        """~15 s of walking at ~1 u/s with mild natural jitter."""
        samples = []
        for i in range(30):
            jitter = (0.25 if (i % 2) else -0.2) if noise else 0.0
            samples.append((i * 0.5, float(i) * 1.0 + jitter, 0.0))
        return samples

    def test_burst_after_calm_baseline_flags(self):
        det = SpeedAnomalyDetector(
            z_threshold=2.5, min_baseline_s=5.0, absolute_floor=1.5, cooldown_s=1
        )
        samples = self._calm_samples()
        for step in range(len(samples)):
            path = _FakePath(gid=1, samples=samples[: step + 1])
            assert det.observe(path, now=samples[step][0]) == []

        # Sudden burst to ~10 u/s.
        burst = [(16.0 + i * 0.5, 20.0 + 10.0 * i * 0.5, 0.0) for i in range(1, 7)]
        flagged = []
        all_pts = samples + burst
        for i in range(len(samples), len(all_pts)):
            fast = _FakePath(gid=1, samples=all_pts[: i + 1])
            flagged.extend(det.observe(fast, now=all_pts[i][0]))
        assert len(flagged) >= 1
        assert flagged[0].rule_type == "anomaly_speed"

    def test_cooldown_suppresses_repeat(self):
        det = SpeedAnomalyDetector(
            min_baseline_s=0.0, cooldown_s=1000, z_threshold=2.0, absolute_floor=0.5
        )
        samples = self._calm_samples()
        for i in range(len(samples)):
            det.observe(_FakePath(1, samples[: i + 1]), now=samples[i][0])
        burst = [(16.0, 60.0, 0.0), (17.0, 70.0, 0.0)]  # far beyond the calm path
        v1 = det.observe(_FakePath(1, samples + burst[:1]), now=16.0)
        v2 = det.observe(_FakePath(1, samples + burst), now=17.0)
        assert v1 and not v2  # second suppressed by cooldown


class TestDwellAnomaly:
    def test_flags_unusual_lingering(self):
        det = DwellAnomalyDetector(min_samples=5, z_threshold=3.0)
        # Population history: visits of ~30 s with realistic variance.
        base_t = 0.0
        for i, dwell in enumerate([28.0, 31.0, 29.0, 33.0, 27.0, 30.0, 32.0, 26.0]):
            det.enter(100 + i, "Vault", base_t)
            det.exit(100 + i, "Vault", base_t + dwell)
            base_t += 100.0

        # New person lingers far longer than any prior visit.
        det.enter(999, "Vault", base_t)
        assert det.check_active(999, "Vault", base_t + 35.0) == []
        hits = det.check_active(999, "Vault", base_t + 300.0)
        assert len(hits) == 1
        assert hits[0].rule_type == "anomaly_dwell"
        assert "Vault" in hits[0].zone_name
