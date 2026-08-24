"""Tests for homography mapping, floor plan, and zone queries."""

import numpy as np
import pytest

from src.core.mapping.coordinate_mapper import CoordinateMapper
from src.core.mapping.floor_plan import FloorPlan


class TestCoordinateMapper:
    def test_identity_homography_roundtrip(self):
        mapper = CoordinateMapper()
        mapper.register_homography("cam_001", np.eye(3))
        x, y = mapper.pixel_to_world("cam_001", 12.0, 34.0)
        assert (x, y) == pytest.approx((12.0, 34.0))

    def test_scale_translation_homography(self):
        """World = 2 * pixel + offset(10, 5)."""
        H = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]])
        mapper = CoordinateMapper()
        mapper.register_homography("cam_002", H)

        wx, wy = mapper.pixel_to_world("cam_002", 5.0, 7.5)
        assert (wx, wy) == pytest.approx((20.0, 20.0))

        px, py = mapper.world_to_pixel("cam_002", 20.0, 20.0)
        assert (px, py) == pytest.approx((5.0, 7.5))

    def test_compute_homography_from_synthetic_correspondences(self):
        rng = np.random.default_rng(42)
        world_pts = rng.uniform(0, 100, size=(6, 2))
        # Affine-ish ground-truth transform: scale 0.05 m/px + shift.
        image_pts = world_pts / 0.05 + [320, 240]

        CoordinateMapper()
        H = CoordinateMapper.compute_homography(image_pts, world_pts)

        test_px = np.float32([[[400, 300]]])
        proj = __import__("cv2").perspectiveTransform(test_px, H)[0, 0]
        expected = (400 - 320) * 0.05, (300 - 240) * 0.05
        assert tuple(proj) == pytest.approx(expected, abs=1e-3)

    def test_uncalibrated_camera_raises(self):
        mapper = CoordinateMapper()
        with pytest.raises(KeyError):
            mapper.pixel_to_world("ghost_cam", 1.0, 1.0)

    def test_bbox_uses_ground_point(self):
        H = np.eye(3)
        mapper = CoordinateMapper()
        mapper.register_homography("cam_001", H)
        # bbox bottom-center is ((10+30)/2, 200) = (20, 200)
        assert mapper.bbox_to_world("cam_001", (10, 100, 30, 200)) == pytest.approx((20.0, 200.0))


class TestFloorPlan:
    PLAN = [(0, 0), (100, 0), (100, 100), (0, 100)]
    LOBBY = [(10, 10), (50, 10), (50, 40), (10, 40)]  # inside plan
    VAULT = [(60, 60), (90, 60), (90, 90), (60, 90)]

    def _plan(self) -> FloorPlan:
        fp = FloorPlan(size=(120, 120))
        fp.add_zone("Lobby", self.LOBBY, "required", rule="required_zone", params={"timeout": 60})
        fp.add_zone("Vault", self.VAULT, "restricted")
        return fp

    def test_point_in_zones(self):
        fp = self._plan()
        names = {z["name"] for z in fp.check_point_in_zone(30, 25)}
        assert names == {"Lobby"}

        names = {z["name"] for z in fp.check_point_in_zone(75, 75)}
        assert names == {"Vault"}

        assert fp.check_point_in_zone(55, 45) == []

    def test_dwell_time_accumulates(self):
        fp = self._plan()
        samples = [
            (0.0, 20, 20),  # enter Lobby
            (5.0, 21, 21),
            (8.0, 70, 70),  # exit -> dwell += 8
            (9.0, 71, 71),
            (12.0, 22, 22),  # re-enter
            None,  # still inside at t=17
        ]
        positions = [(t[0], t[1], t[2]) for t in samples if t is not None]
        dwell = fp.dwell_time_in_zone("Lobby", positions, current_time=17.0)
        assert dwell == pytest.approx(8.0 + 5.0)  # 8s closed visit + 5s open

    def test_render_produces_canvas(self):
        fp = self._plan()
        canvas = fp.render(
            tracks_world={1: (25.0, 26.0)}, paths={1: [(11, 11), (18, 19), (25, 26)]}
        )
        assert canvas.shape == (120, 120, 3)
