"""Appearance embedding protocol + dependency-free fallback embedder.

The Re-ID stage is pluggable:

  1. ``OSNetEmbedder``   — torchreid OSNet, best accuracy (optional install)
  2. ``ColorGridEmbedder`` — HSV histogram over a spatial grid, zero deps

``create_embedder`` picks the strongest backend available so the pipeline
always runs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class Embedder(Protocol):
    feature_dim: int

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return an ``(N, D)`` L2-normalized float32 array for N crops."""
        ...


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-9, None)


class ColorGridEmbedder:
    """HSV color-histogram descriptor over a 2x3 spatial grid.

    Not as discriminative as learned Re-ID, but deterministic, fast, and good
    enough to exercise the full cross-camera matching workflow offline.
    """

    def __init__(self, grid: tuple[int, int] = (3, 2), bins: int = 8) -> None:
        self.grid = grid  # (rows, cols)
        self.bins = bins
        rows, cols = grid
        self.feature_dim = int(rows * cols * bins * 2)  # H + S channels per cell

    def _embed_one(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return np.zeros(self.feature_dim, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_channel, s_channel = hsv[:, :, 0], hsv[:, :, 1]

        rows, cols = self.grid
        cell_h = crop.shape[0] // rows
        cell_w = crop.shape[1] // cols

        desc: list[float] = []
        for r in range(rows):
            for c in range(cols):
                tile_h = h_channel[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                tile_s = s_channel[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                hist_h = cv2.calcHist([tile_h], [0], None, [self.bins], [0, 180])
                hist_s = cv2.calcHist([tile_s], [0], None, [self.bins], [0, 256])
                desc.extend(hist_h.flatten().tolist())
                desc.extend(hist_s.flatten().tolist())

        arr = np.asarray(desc, dtype=np.float32)
        total = arr.sum()
        if total > 0:
            arr /= total
        return arr

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        feats = (
            np.stack([self._embed_one(crop) for crop in crops])
            if crops
            else np.zeros((0, self.feature_dim), dtype=np.float32)
        )
        return l2_normalize(feats)


def create_embedder(backend: str = "auto", device: str | None = None) -> Embedder:
    """Build the best available embedder; never raises for 'auto'."""
    backend = backend.lower()
    if backend in ("auto", "osnet"):
        try:
            from .osnet_embedder import OSNetEmbedder

            emb = OSNetEmbedder(device=device)
            return emb
        except Exception:
            if backend == "osnet":
                raise
            # fall through to color-grid fallback
    return ColorGridEmbedder()
