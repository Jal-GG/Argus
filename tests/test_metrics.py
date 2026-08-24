"""Tests for the Prometheus /metrics endpoint and ops routes."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.rules.rule_engine import RuleViolation
from src.core.state_hub import StateHub
from src.core.types import Track


@pytest.fixture()
def client():
    hub = StateHub()
    hub.register_camera("cam_001", name="Ops Cam")
    hub.set_camera_status("cam_001", online=True, fps=20.0)
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    for _ in range(5):
        hub.bump_frames("cam_001")
        hub.set_frame("cam_001", frame)
    hub.update_tracks(
        "cam_001",
        [
            Track(track_id=i, camera_id="cam_001", bbox=(0, 0, 10, 20), confidence=0.9, global_id=i)
            for i in (1, 2, 3)
        ],
    )
    hub.set_world_state({1: (1.0, 2.0)}, {1: [(0.0, 0.0), (1.0, 2.0)]})
    hub.add_event_from_violation(
        RuleViolation(
            rule_type="restricted_zone",
            zone_name="Vault",
            global_id=3,
            reason="test violation",
            timestamp=99.0,
        )
    )
    return TestClient(create_app(hub=hub))


def test_metrics_exposition_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    # Prometheus text exposition sanity.
    assert "surveillance_active_tracks" in body
    assert "surveillance_cameras_online" in body
    assert "surveillance_pipeline_running" in body


def test_metrics_counter_values_advance(client):
    first = client.get("/api/health").json()
    assert first["active_tracks"] == 3

    body1 = client.get("/metrics").text
    # After one scrape the events counter should include our violation.
    assert "surveillance_events_total" in body1

    hub_extra = RuleViolation(
        rule_type="loitering", zone_name="Lobby", global_id=1, reason="second", timestamp=100.0
    )
    client.app.state.hub.add_event_from_violation(hub_extra)

    body2 = client.get("/metrics").text
    loitering_lines = [
        line
        for line in body2.splitlines()
        if line.startswith("surveillance_events_total{") and "loitering" in line
    ]
    assert len(loitering_lines) == 1


def test_frames_endpoint(client):
    data = client.get("/api/frames").json()
    assert "cam_001" in data
    assert data["cam_001"]["fps"] == pytest.approx(20.0)
