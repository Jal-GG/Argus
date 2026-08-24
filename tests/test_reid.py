"""Tests for the Re-ID embedder and global identity manager."""

import numpy as np
import pytest

from src.core.reid.embedder import ColorGridEmbedder, l2_normalize
from src.core.reid.global_id_manager import GlobalIDManager


class TestColorGridEmbedder:
    def test_output_shape_and_normalization(self):
        emb = ColorGridEmbedder()
        crops = [
            np.random.default_rng(i).integers(0, 255, (120, 60, 3), dtype=np.uint8)
            for i in range(3)
        ]
        feats = emb.embed(crops)
        assert feats.shape == (3, emb.feature_dim)
        np.testing.assert_allclose(np.linalg.norm(feats, axis=1), 1.0, rtol=1e-5)

    def test_same_crop_similar_different_crop_dissimilar(self):
        np.random.default_rng(0)
        red = np.zeros((100, 50, 3), dtype=np.uint8)
        red[:, :] = (40, 40, 200)  # BGR red-ish person
        blue = np.zeros((100, 50, 3), dtype=np.uint8)
        blue[:, :] = (200, 40, 40)

        emb = ColorGridEmbedder()
        f_red, f_blue = emb.embed([red, blue])
        sim_same = float(f_red @ f_red)
        sim_diff = float(f_red @ f_blue)
        assert sim_same > sim_diff + 0.5

    def test_empty_crop_is_zero_vector(self):
        emb = ColorGridEmbedder()
        (feat,) = emb.embed([np.zeros((0, 0, 3), dtype=np.uint8)])
        assert feat.shape == (emb.feature_dim,)
        assert float(feat.sum()) == pytest.approx(0.0)


def test_l2_normalize_zero_vector_safe():
    out = l2_normalize(np.array([[0.0, 0.0]], dtype=np.float32))
    assert np.isfinite(out).all()


class TestGlobalIDManager:
    def _feat(self, seed: int) -> np.ndarray:
        v = np.random.default_rng(seed).normal(size=32)
        return v / np.linalg.norm(v)

    def test_same_feature_same_global_id(self):
        mgr = GlobalIDManager(similarity_threshold=0.6)
        f = self._feat(1)
        g1 = mgr.match_or_create("cam_001", 10, f, timestamp=1.0)
        g2 = mgr.match_or_create("cam_002", 77, f, timestamp=2.0)
        assert g1 == g2

    def test_orthogonal_features_new_id(self):
        mgr = GlobalIDManager(similarity_threshold=0.9)
        a, b = self._feat(1), self._feat(2)
        # ensure near-orthogonal by construction check
        if abs(float(a @ b)) > 0.5:  # pragma: no cover
            b = -b
            b /= np.linalg.norm(b)
        g_a = mgr.match_or_create("cam_001", 1, a, timestamp=1.0)
        g_b = mgr.match_or_create("cam_002", 2, b, timestamp=2.0)
        assert g_a != g_b

    def test_known_track_binding_persists(self):
        mgr = GlobalIDManager()
        g1 = mgr.match_or_create("cam_001", 5, self._feat(3), timestamp=1.0)
        g2 = mgr.match_or_create("cam_001", 5, None, timestamp=4.0)
        assert g1 == g2

    def test_temporal_gate_blocks_stale_match(self):
        """Feature match should be rejected when last seen too long ago."""
        mgr = GlobalIDManager(similarity_threshold=0.5, max_time_gap=5.0)
        f = self._feat(4)
        g_first = mgr.match_or_create("cam_001", 1, f, timestamp=0.0)
        # New local track on another camera, but 1 hour later.
        g_second = mgr.match_or_create("cam_002", 9, f, timestamp=3600.0)
        assert g_first != g_second

    def test_spatial_gate_blocks_impossible_travel(self):
        """Same second, cameras 500 m apart → physically impossible."""
        positions = {"cam_A": (0.0, 0.0), "cam_B": (500.0, 0.0)}
        mgr = GlobalIDManager(
            similarity_threshold=0.5,
            camera_positions=positions,
            max_distance_between_cameras=50.0,
        )
        f = self._feat(5)
        g_a = mgr.match_or_create("cam_A", 1, f, timestamp=10.0)
        g_b = mgr.match_or_create("cam_B", 2, f, timestamp=11.0)  # 1 s later
        assert g_a != g_b

    def test_spatial_gate_allows_feasible_travel(self):
        positions = {"cam_A": (0.0, 0.0), "cam_B": (8.0, 0.0)}  # ~5 s walk
        mgr = GlobalIDManager(similarity_threshold=0.5, camera_positions=positions)
        f = self._feat(6)
        g_a = mgr.match_or_create("cam_A", 1, f, timestamp=10.0)
        g_b = mgr.match_or_create("cam_B", 2, f, timestamp=20.0)  # 10 s later
        assert g_a == g_b

    def test_gallery_size_capped(self):
        mgr = GlobalIDManager(gallery_size=5)
        for i in range(20):
            mgr._update_gallery(1, self._feat(10 + i), timestamp=i * 0.1, camera_id="c")
        assert len(mgr.galleries[1].features) == 5
