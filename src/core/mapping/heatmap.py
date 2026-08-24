"""Occupancy heat-map over the floor plan.

Accumulates world-space samples into a grid, applies exponential time decay
so the map reflects *recent* traffic, and renders a colored overlay blended
onto the plan image.
"""

from __future__ import annotations

import cv2
import numpy as np


class Heatmap:
    def __init__(
        self, width: int, height: int, cell_size: int = 12, decay_per_minute: float = 0.5
    ) -> None:
        """Args:
        width/height: floor-plan pixel size.
        cell_size: aggregation cell in plan pixels.
        decay_per_minute: fraction of mass kept after one minute (0..1].
        """
        self.width = int(width)
        self.height = int(height)
        self.cell = max(4, int(cell_size))
        cols = (self.width + self.cell - 1) // self.cell
        rows_n = (self.height + self.cell - 1) // self.cell
        self.grid = np.zeros((rows_n, cols), dtype=np.float32)
        self.decay_per_minute = float(np.clip(decay_per_minute, 0.01, 1.0))
        self._last_update: float | None = None

    # ------------------------------------------------------------------ #
    def add_point(self, x: float, y: float, ts: float, weight: float = 1.0) -> None:
        """Apply time decay lazily on each insert."""
        if self._last_update is not None and ts > self._last_update:
            minutes = (ts - self._last_update) / 60.0
            factor = float(self.decay_per_minute**minutes)
            if factor < 0.999:
                self.grid *= factor
        self._last_update = ts

        col = min(max(int(x) // self.cell, 0), self.grid.shape[1] - 1)
        row = min(max(int(y) // self.cell, 0), self.grid.shape[0] - 1)
        self.grid[row, col] += weight

    def add_points(self, points, weight: float = 1.0, ts: float | None = None) -> None:
        for x, y in points:
            if ts is not None:
                self.add_point(x, y, ts=ts, weight=weight)
            else:
                # No timestamp: accumulate directly into the cell.
                col = min(max(int(x) // self.cell, 0), self.grid.shape[1] - 1)
                row = min(max(int(y) // self.cell, 0), self.grid.shape[0] - 1)
                self.grid[row, col] += weight

    # ------------------------------------------------------------------ #
    def render_overlay(
        self, base_bgr: np.ndarray, alpha: float = 0.55, blur_ksize: int = 31
    ) -> np.ndarray:
        """Return ``base`` blended with a turbo-coloured density layer."""
        if self.grid.max() <= 0:
            return base_bgr

        norm = self.grid / (self.grid.max() + 1e-9)
        upscaled = cv2.resize(
            norm, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_CUBIC
        )
        if blur_ksize > 1:
            k = blur_ksize | 1
            upscaled = cv2.GaussianBlur(upscaled, (k, k), 0)

        colored = cv2.applyColorMap(
            (np.clip(upscaled, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )

        mask = (upscaled > 0.04).astype(np.float32)[..., None]
        blended = (
            base_bgr.astype(np.float32) * (1 - alpha * mask)
            + colored.astype(np.float32) * alpha * mask
        )
        return blended.astype(np.uint8)

    # ------------------------------------------------------------------ #
    @property
    def total_mass(self) -> float:
        return float(self.grid.sum())

    def top_cells(self, n: int = 5) -> list[tuple[int, int]]:
        flat = self.grid.flatten()
        idx = np.argpartition(flat, -min(n, flat.size))[-n:]
        idx = idx[np.argsort(flat[idx])[::-1]]
        return [
            (
                int(i // self.grid.shape[1] * self.cell + self.cell // 2),
                int(i % self.grid.shape[1] * self.cell + self.cell // 2),
            )
            for i in idx
            if flat[i] > 0
        ]
