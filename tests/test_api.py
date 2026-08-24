"""API tests: exercise REST + WebSocket against a synthetic StateHub."""

import sys
from pathlib import Path

import cv2
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
    hub.register_camera("cam_001", name="Test Cam", source="test")
    hub.set_camera_status("cam_001", online=True, fps=12.3, dropped=2)

    frame = np.full((120, 160, 3), 90, dtype=np.uint8)
    cv2.circle(frame, (80, 60), 30, (0, 140, 255), -1)
    hub.set_frame("cam_001", frame, quality=70)

    hub.update_tracks(
        "cam_001",
        [
            Track(
                track_id=1, camera_id="cam_001", bbox=(10, 10, 50, 110), confidence=0.9, global_id=7
            ),
            Track(
                track_id=2,
                camera_id="cam_001",
                bbox=(200, 10, 240, 100),
                confidence=0.8,
                global_id=8,
            ),
        ],
    )
    hub.set_world_state(
        {7: (120.5, 300.25), 8: (400.0, 210.0)},
        {7: [(10, 10), (60, 80), (120.5, 300.25)]},
        alerts={8},
    )
    hub.add_event_from_violation(
        RuleViolation(
            rule_type="restricted_zone",
            zone_name="Server Room",
            global_id=8,
            reason="Entered restricted zone 'Server Room'",
            timestamp=1234.5,
        )
    )

    app = create_app(hub=hub)
    yield TestClient(app)


class TestSystemRoutes:
    def test_health(self, client):
        data = client.get("/api/health").json()
        assert data["status"] == "idle"  # no pipeline attached
        assert data["cameras_total"] == 1
        assert data["active_tracks"] == 2
        assert data["events_total"] == 1

    def test_cameras_list(self, client):
        cams = client.get("/api/cameras").json()
        assert len(cams) == 1
        cam = cams[0]
        assert cam["id"] == "cam_001"
        assert cam["online"] is True
        assert cam["fps"] == pytest.approx(12.3, abs=0.01)
        assert cam["has_frame"] is True

    def test_snapshot_jpeg(self, client):
        resp = client.get("/api/cameras/cam_001/snapshot")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[:2] == (120, 160)

    def test_snapshot_unknown_camera_404(self, client):
        assert client.get("/api/cameras/ghost/snapshot").status_code == 404

    def test_mjpeg_stream_content_type_and_first_frame(self, client):
        resp = client.get("/api/cameras/cam_001/stream?fps=60&max_frames=2")
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.headers["content-type"]
        assert b"\xff\xd8" in resp.content  # JPEG SOI marker present


class TestTrackingRoutes:
    def test_tracks(self, client):
        tracks = client.get("/api/tracks").json()
        gids = {t["global_id"] for t in tracks}
        assert gids == {7, 8}
        t7 = next(t for t in tracks if t["global_id"] == 7)
        assert t7["camera_id"] == "cam_001"
        assert len(t7["bbox"]) == 4

    def test_floorplan_unavailable_without_pipeline(self, client):
        # No pipeline -> 503, not a crash.
        assert client.get("/api/floorplan").status_code == 503

    def test_events(self, client):
        events = client.get("/api/events?limit=10").json()
        assert len(events) == 1
        ev = events[0]
        assert ev["rule_type"] == "restricted_zone"
        assert ev["zone_name"] == "Server Room"
        assert ev["global_id"] == 8

    def test_paths_requires_pipeline(self, client):
        assert client.get("/api/paths/7").status_code == 503


class TestDashboard:
    def test_root_redirects_to_dashboard(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 307)

    def test_static_assets_served(self, client):
        for asset in ("index.html", "app.js", "style.css"):
            resp = client.get(f"/static/{asset}")
            assert resp.status_code == 200, asset


class TestWebSocket:
    def test_ws_pushes_state(self, client):
        with client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "state"
            assert set(msg["positions"].keys()) == {"7", "8"}
            assert msg["alerts"] == [8]
            assert msg["events"][0]["rule_type"] == "restricted_zone"
            assert msg["cameras"]["cam_001"]["fps"] == pytest.approx(12.3, abs=0.1)
