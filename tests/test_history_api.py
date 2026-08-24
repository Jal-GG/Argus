"""Tests for Phase 9 history/analytics API endpoints (with real SQLite)."""

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.mapping.floor_plan import FloorPlan
from src.core.path.database import AnalyticsDB, AnalyticsWriter


class _FakePipeline:
    """Only what the analytics routes need."""

    def __init__(self, tmp_path):
        self.db = AnalyticsDB(tmp_path / "analytics.db")
        self.writer = AnalyticsWriter(self.db, flush_interval_s=999)
        now = time.time()
        for gid in (1, 2):
            for i in range(20):
                self.writer.add_position(
                    gid, ts=now - 100 + i * 5, x=10.0 * i + gid, y=5.0, camera_id="cam_001"
                )
        self.writer.flush()
        self.db.insert_event(
            {
                "global_id": 1,
                "rule_type": "restricted_zone",
                "zone_name": "Vault",
                "timestamp": now - 30.0,
                "reason": "test",
            }
        )
        self.floor_plan = FloorPlan(size=(100, 80))

    def render_heatmap(self):
        from src.core.mapping.heatmap import Heatmap

        h, w = self.floor_plan.image.shape[:2]
        hm = Heatmap(w, h)
        return hm.render_overlay(self.floor_plan.image.copy())


@pytest.fixture()
def client(tmp_path):
    pipeline = _FakePipeline(tmp_path)
    app = create_app(pipeline=pipeline)
    return TestClient(app)


def test_history_positions_trails(client):
    data = client.get("/api/history/positions?minutes=10").json()
    assert set(data["trails"].keys()) == {"1", "2"}
    assert all(len(pts) == 20 for pts in data["trails"].values())
    assert data["count"] == 40


def test_history_positions_person_filter(client):
    data = client.get("/api/history/positions?minutes=10&global_id=2").json()
    assert set(data["trails"].keys()) == {"2"}


def test_history_events_filter(client):
    events = client.get("/api/history/events?minutes=60").json()
    assert len(events) == 1
    assert events[0]["event_type"] == "restricted_zone"

    none = client.get("/api/history/events?minutes=60&event_type=loitering").json()
    assert none == []


def test_history_heatmap_points_count(client):
    data = client.get("/api/history/heatmap-points?minutes=60").json()
    assert data["samples"] == 40


def test_history_persons(client):
    persons = client.get("/api/history/persons").json()
    assert {p["global_id"] for p in persons} == {1, 2}


def test_heatmap_endpoint_renders(client):
    resp = client.get("/api/floorplan/heatmap")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_history_unavailable_without_db():
    client = TestClient(create_app())  # no pipeline at all
    assert client.get("/api/history/positions?minutes=5").status_code == 503
