"""Pydantic response schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CameraInfo(BaseModel):
    id: str
    name: str = ""
    source: str = ""
    online: bool = False
    fps: float = 0.0
    dropped: int = 0
    has_frame: bool = False


class TrackInfo(BaseModel):
    camera_id: str
    track_id: int
    global_id: int | None = None
    bbox: list[float] = Field(default_factory=list)
    confidence: float = 0.0


class EventItem(BaseModel):
    ts: float
    rule_type: str
    zone_name: str
    global_id: int | None = None
    reason: str
    snapshot: bool = False


class ZoneInfo(BaseModel):
    name: str
    type: str
    rule: str | None = None
    polygon: list[list[float]]
    params: dict = Field(default_factory=dict)


class FloorPlanInfo(BaseModel):
    floor_id: str
    width: int
    height: int
    zones: list[ZoneInfo]


class SystemHealth(BaseModel):
    status: str
    pipeline_running: bool
    uptime_s: float
    cameras_online: int
    cameras_total: int
    active_tracks: int
    events_total: int
