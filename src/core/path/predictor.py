"""Lightweight online path prediction.

Two complementary predictors:

1. :class:`VelocityPredictor` — exponentially-smoothed velocity per global ID
   extrapolated over a short horizon (seconds). Robust, zero-training.
2. :class:`ZoneMarkov` — learns zone→zone transition probabilities online and
   predicts the most likely next zone.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _VelState:
    vx: float = 0.0
    vy: float = 0.0
    last_x: float = 0.0
    last_y: float = 0.0
    last_ts: float = 0.0
    initialized: bool = False


@dataclass
class Prediction:
    global_id: int
    points: list[tuple[float, float]] = field(default_factory=list)
    next_zone: str | None = None

    def to_dict(self) -> dict:
        return {
            "global_id": self.global_id,
            "points": [[round(x, 1), round(y, 1)] for x, y in self.points],
            "next_zone": self.next_zone,
        }


class VelocityPredictor:
    """Short-horizon linear extrapolation with EMA-smoothed velocity."""

    def __init__(
        self,
        horizon_s: float = 3.0,
        steps: int = 6,
        alpha: float = 0.4,
        max_speed: float = 8.0,
        stale_after_s: float = 5.0,
    ) -> None:
        self.horizon_s = float(horizon_s)
        self.steps = int(steps)
        self.alpha = float(alpha)  # EMA weight for newest sample
        self.max_speed = float(max_speed)
        self.stale_after_s = float(stale_after_s)
        self._states: dict[int, _VelState] = {}

    # ------------------------------------------------------------------ #
    def observe(self, global_id: int, x: float, y: float, ts: float) -> None:
        state = self._states.setdefault(global_id, _VelState())
        if not state.initialized:
            state.last_x, state.last_y, state.last_ts = x, y, ts
            state.initialized = True
            return

        dt = ts - state.last_ts
        if dt <= 1e-3:
            return
        raw_vx = (x - state.last_x) / dt
        raw_vy = (y - state.last_y) / dt

        speed = float(np.hypot(raw_vx, raw_vy))
        if speed > self.max_speed:  # teleport/jitter guard
            state.last_x, state.last_y, state.last_ts = x, y, ts
            return

        state.vx = self.alpha * raw_vx + (1 - self.alpha) * state.vx
        state.vy = self.alpha * raw_vy + (1 - self.alpha) * state.vy
        state.last_x, state.last_y, state.last_ts = x, y, ts

    # ------------------------------------------------------------------ #
    def predict(self, global_id: int) -> Prediction | None:
        """Extrapolate future positions; ``None`` if unknown/stale/slow."""
        state = self._states.get(global_id)
        if state is None or not state.initialized:
            return None

        speed = float(np.hypot(state.vx, state.vy))
        if speed < 0.15:  # stationary people need no forecast
            return None

        points = []
        for i in range(1, self.steps + 1):
            t = self.horizon_s * i / self.steps
            points.append((state.last_x + state.vx * t, state.last_y + state.vy * t))

        return Prediction(global_id=global_id, points=points)

    def forget(self, global_id: int) -> None:
        self._states.pop(global_id, None)


class ZoneMarkov:
    """Online first-order Markov model over zone sequences."""

    def __init__(self) -> None:
        self._transitions: dict[str, Counter[str]] = defaultdict(Counter)

    def observe_sequence(self, zones: list[str]) -> None:
        for a, b in itertools.pairwise(zones):
            self._transitions[a][b] += 1

    def next_zone(self, current: str) -> str | None:
        counter = self._transitions.get(current)
        if not counter:
            return None
        total = sum(counter.values())
        best, count = counter.most_common(1)[0]
        confidence = count / total
        return best if confidence >= 0.25 else None

    def transition_probs(self, current: str) -> dict[str, float]:
        counter = self._transitions.get(current, Counter())
        total = sum(counter.values()) or 1
        return {z: round(c / total, 3) for z, c in counter.most_common()}
