"""Unit tests for src/utils/geometry.py."""

import pytest

from src.utils.geometry import (
    bbox_bottom_center,
    bbox_center,
    bbox_iou,
    euclidean,
    path_length,
    point_in_polygon,
)


class TestBBoxIoU:
    def test_identical_boxes(self):
        assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert bbox_iou((0, 0, 5, 5), (10, 10, 20, 20)) == 0.0

    def test_partial_overlap(self):
        # overlap = 5x5=25, union = 100+100-25 = 175
        assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)

    def test_touching_edges_only(self):
        assert bbox_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


class TestCenters:
    def test_center(self):
        assert bbox_center((0, 0, 10, 20)) == (5.0, 10.0)

    def test_bottom_center_is_ground_point(self):
        bc = bbox_bottom_center((2, 3, 12, 43))
        assert bc == (7.0, 43.0)


class TestPointInPolygon:
    SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]

    def test_inside(self):
        assert point_in_polygon((5, 5), self.SQUARE) is True

    def test_outside(self):
        assert point_in_polygon((15, 5), self.SQUARE) is False

    def test_on_edge_counts_as_inside(self):
        assert point_in_polygon((0, 5), self.SQUARE) is True

    def test_concave_polygon(self):
        l_shape = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
        assert point_in_polygon((2, 8), l_shape) is True
        assert point_in_polygon((7, 8), l_shape) is False


def test_euclidean_and_path_length():
    assert euclidean((0, 0), (3, 4)) == pytest.approx(5.0)
    pts = [(0, 0), (3, 4), (6, 8)]
    assert path_length(pts) == pytest.approx(10.0)
