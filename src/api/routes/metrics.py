"""Prometheus metrics endpoint (pull-based, dependency-optional).

Metrics are derived from the StateHub at scrape time, so the pipeline never
needs to know about Prometheus. Counters are delta-synced (hub buffers only
keep recent events, but cumulative totals live in hub.frames_processed and a
high-water mark for events). If prometheus_client is missing, /metrics falls
back to JSON instead of crashing.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["ops"])

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        generate_latest,
    )

    _FRAMES = Counter("surveillance_frames_total", "Frames processed per camera", ["camera"])
    _EVENTS = Counter("surveillance_events_total", "Rule violations raised", ["rule_type"])
    _TRACKS = Gauge("surveillance_active_tracks", "Currently confirmed tracks")
    _PEOPLE = Gauge("surveillance_people_on_plan", "Distinct global IDs on the floor plan")
    _CAMERAS_ONLINE = Gauge("surveillance_cameras_online", "Cameras currently online")
    _PIPELINE_UP = Gauge("surveillance_pipeline_running", "Pipeline thread running")
    _UPTIME = Gauge("surveillance_uptime_seconds", "Seconds since hub start")

    PROMETHEUS = True
except ImportError:  # pragma: no cover - optional dependency
    PROMETHEUS = False


class _SyncState:
    """High-water marks so counters only advance monotonically."""

    def __init__(self) -> None:
        self.events_seen = 0
        self.frames_synced: dict[str, int] = {}


_sync = _SyncState()


def _sync_metrics(hub) -> None:
    statuses = hub.camera_statuses()
    snapshot = hub.live_snapshot()

    _TRACKS.set(len(snapshot["tracks"]))
    _PEOPLE.set(len(snapshot["positions"]))
    _CAMERAS_ONLINE.set(sum(1 for c in statuses if c.online))
    _PIPELINE_UP.set(1 if hub.pipeline_running else 0)
    _UPTIME.set(time.time() - hub.started_at)

    # Frames: delta-sync per camera.
    for cam_id, total in hub.frames_totals().items():
        prev = _sync.frames_synced.get(cam_id, 0)
        delta = max(0, total - prev)
        if delta > 0:
            _FRAMES.labels(camera=cam_id).inc(delta)
        _sync.frames_synced[cam_id] = total

    # Events: monotonic hub counter, delta-synced per rule type.
    new_total = hub.events_total
    unseen = max(0, new_total - _sync.events_seen)
    if unseen:
        recent = hub.recent_events(limit=min(unseen * 2, 200))
        for ev in recent[:unseen]:
            _EVENTS.labels(rule_type=ev.rule_type).inc()
        _sync.events_seen = new_total


@router.get("/metrics")
def metrics(request: Request):
    hub = request.app.state.hub
    if not PROMETHEUS:  # pragma: no cover - optional dependency
        return JSONResponse({"error": "prometheus_client not installed"})
    try:
        _sync_metrics(hub)
    except Exception as exc:  # never break scrapes on transient state races
        logger.warning("metrics sync failed: %s", exc)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/api/frames")
def frames_metrics(request: Request) -> dict:
    """Plain-JSON per-camera throughput stats (no Prometheus required)."""
    hub = request.app.state.hub
    return {
        cam.camera_id: {
            "fps": round(cam.fps, 2),
            "dropped": cam.dropped,
            "frames_processed": hub.frames_totals().get(cam.camera_id, 0),
            "last_frame_age_s": round(time.time() - cam.last_frame_ts, 1)
            if cam.last_frame_ts
            else None,
        }
        for cam in hub.camera_statuses()
    }
