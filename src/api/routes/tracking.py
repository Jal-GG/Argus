"""Tracking routes: live tracks, floor plan metadata/render, paths, events,
plus Phase 9 analytics endpoints (heatmap, historical playback)."""

from __future__ import annotations

import threading
import time

import cv2
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from ...api.schemas.schemas import EventItem, FloorPlanInfo, TrackInfo, ZoneInfo
from ...core.mapping.floor_plan import FloorPlan

router = APIRouter(prefix="/api", tags=["tracking"])

_render_lock = threading.Lock()
_render_cache: dict[str, tuple[float, bytes]] = {}
_RENDER_TTL_S = 1.0


@router.get("/tracks", response_model=list[TrackInfo])
def tracks(request: Request) -> list[TrackInfo]:
    return [TrackInfo(**t) for t in request.app.state.hub.tracks_snapshot()]


def _floor_plan(request: Request) -> FloorPlan:
    pipeline = getattr(request.app.state, "pipeline", None)
    plan = getattr(pipeline, "floor_plan", None) if pipeline else None
    if plan is None:
        raise HTTPException(503, "Floor plan unavailable (pipeline not loaded)")
    return plan


@router.get("/floorplan", response_model=FloorPlanInfo)
def floorplan(request: Request) -> FloorPlanInfo:
    plan = _floor_plan(request)
    h, w = plan.image.shape[:2]
    zones = [
        ZoneInfo(
            name=z["name"],
            type=z["type"],
            rule=z.get("rule"),
            polygon=[[float(x), float(y)] for x, y in z["polygon"]],
            params=z.get("params") or {},
        )
        for z in plan.zones
    ]
    return FloorPlanInfo(floor_id=plan.floor_id, width=w, height=h, zones=zones)


@router.get("/floorplan/image")
def floorplan_image(request: Request) -> Response:
    """Raw base floor-plan image as PNG."""
    plan = _floor_plan(request)
    ok, buf = cv2.imencode(".png", plan.image)
    if not ok:
        raise HTTPException(500, "Failed to encode floor plan image")
    return Response(content=buf.tobytes(), media_type="image/png")


@router.get("/floorplan/render")
def floorplan_render(request: Request) -> Response:
    """Server-rendered plan with live positions/paths/zone overlays (TTL-cached)."""
    _floor_plan(request)
    cached = _render_cache.get("plan")
    if cached and (time.time() - cached[0]) < _RENDER_TTL_S:
        return Response(content=cached[1], media_type="image/jpeg")

    pipeline = request.app.state.pipeline
    canvas = pipeline.render_floor_plan([])
    with _render_lock:
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise HTTPException(500, "Failed to render floor plan")
        _render_cache["plan"] = (time.time(), buf.tobytes())
    return Response(content=_render_cache["plan"][1], media_type="image/jpeg")


@router.get("/paths/{global_id}")
def path_history(request: Request, global_id: int, max_points: int = Query(200, le=1000)) -> dict:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "Pipeline unavailable")
    path = pipeline.path_tracker.active_paths.get(global_id) or next(
        (p for p in pipeline.path_tracker.completed_paths if p.global_id == global_id), None
    )
    if path is None:
        raise HTTPException(404, f"No path for global_id={global_id}")

    pts = path.path_points(max_points=max_points)
    pos = path.current_position
    return {
        "global_id": global_id,
        "status": path.status,
        "first_seen": path.first_seen,
        "last_seen": path.last_seen,
        "total_distance": round(path.total_distance, 2),
        "zones_visited": sorted(path.zone_visits.keys()),
        "current_position": [round(pos[0], 2), round(pos[1], 2)] if pos else None,
        "points": [[round(x, 2), round(y, 2)] for x, y in pts],
    }


@router.get("/events", response_model=list[EventItem])
def events(request: Request, limit: int = Query(50, le=500)) -> list[EventItem]:
    return [EventItem(**e.to_dict()) for e in request.app.state.hub.recent_events(limit)]


# ---------------------------------------------------------------------- #
# Phase 9 analytics: heatmap + historical playback
# ---------------------------------------------------------------------- #
@router.get("/floorplan/heatmap")
def floorplan_heatmap(request: Request) -> Response:
    """Live traffic heat-map blended over the plan (TTL-cached 2 s)."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "Pipeline unavailable")
    cached = _render_cache.get("heat")
    if cached and (time.time() - cached[0]) < 2.0:
        return Response(content=cached[1], media_type="image/jpeg")
    canvas = pipeline.render_heatmap()
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(500, "Failed to render heatmap")
    _render_cache["heat"] = (time.time(), buf.tobytes())
    return Response(content=_render_cache["heat"][1], media_type="image/jpeg")


def _require_db(request: Request):
    db = getattr(getattr(request.app.state, "pipeline", None), "db", None)
    if db is None:
        raise HTTPException(503, "History DB unavailable")
    return db


@router.get("/history/positions")
def history_positions(
    request: Request,
    minutes: float = Query(10.0, gt=0, le=1440),
    global_id: int | None = None,
    limit: int = Query(20000, le=100000),
) -> dict:
    """Raw position samples for client-side playback."""
    db = _require_db(request)
    since = time.time() - minutes * 60.0
    pts = db.positions_in_range(since=since, global_id=global_id, limit=limit)
    trails: dict[str, list[list[float]]] = {}
    for p in pts:
        trails.setdefault(str(p["global_id"]), []).append([round(p["x"], 1), round(p["y"], 1)])
    return {"since": since, "count": len(pts), "trails": trails}


@router.get("/history/heatmap-points")
def history_heatmap_points(request: Request, minutes: float = Query(60.0, gt=0, le=1440)) -> dict:
    """Server-side aggregated density for a past window."""
    db = _require_db(request)
    since = time.time() - minutes * 60.0
    points = db.heatmap_points(since=since)
    return {"since": since, "samples": len(points)}


@router.get("/history/events")
def history_events(
    request: Request,
    minutes: float = Query(60.0, gt=0, le=10080),
    event_type: str | None = None,
    limit: int = Query(200, le=1000),
) -> list[dict]:
    db = _require_db(request)
    since = time.time() - minutes * 60.0
    return db.events_in_range(since=since, event_type=event_type, limit=limit)


@router.get("/history/persons")
def history_persons(request: Request, limit: int = Query(100, le=500)) -> list[dict]:
    return _require_db(request).person_summaries(limit=limit)
