"""System routes: health, camera list, snapshots, MJPEG streams."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ...api.schemas.schemas import CameraInfo
from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["system"])

MJPEG_BOUNDARY = "frame"


@router.get("/health")
def health(request: Request) -> dict:
    hub = request.app.state.hub
    cams = hub.camera_statuses()
    return {
        "status": "ok" if hub.pipeline_running else "idle",
        "pipeline_running": hub.pipeline_running,
        "uptime_s": round(time.time() - hub.started_at, 1),
        "cameras_online": sum(1 for c in cams if c.online),
        "cameras_total": len(cams),
        "active_tracks": len(hub.tracks_snapshot()),
        "events_total": len(hub.recent_events(limit=10**6)),
    }


@router.get("/cameras", response_model=list[CameraInfo])
def cameras(request: Request) -> list[CameraInfo]:
    hub = request.app.state.hub
    return [
        CameraInfo(
            id=st.camera_id,
            name=st.name,
            source=st.source if len(st.source) < 64 else st.source[:61] + "...",
            online=st.online,
            fps=round(st.fps, 1),
            dropped=st.dropped,
            has_frame=hub.get_frame(st.camera_id) is not None,
        )
        for st in hub.camera_statuses()
    ]


@router.get("/cameras/{camera_id}/snapshot")
def snapshot(request: Request, camera_id: str, annotated: bool = True) -> Response:
    """Latest frame as JPEG (single shot)."""
    jpeg = request.app.state.hub.get_frame(camera_id, annotated=annotated)
    if jpeg is None:
        raise HTTPException(404, f"No frame available for camera '{camera_id}'")
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/cameras/{camera_id}/stream")
def mjpeg_stream(
    request: Request, camera_id: str, fps: float = 12.0, max_frames: int = 0
) -> StreamingResponse:
    """MJPEG live stream; serves the latest frame repeatedly at ~`fps`.

    ``max_frames`` bounds the stream (0 = unlimited) for tests and exports.
    """
    hub = request.app.state.hub
    if camera_id not in {c.camera_id for c in hub.camera_statuses()}:
        raise HTTPException(404, f"Unknown camera '{camera_id}'")

    interval = 1.0 / max(min(fps, 30.0), 1.0)

    async def _frames():
        sent = 0
        while not (max_frames and sent >= max_frames):
            if await request.is_disconnected():
                break
            jpeg = hub.get_frame(camera_id)
            if jpeg is not None:
                yield (
                    (
                        f"--{MJPEG_BOUNDARY}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode()
                    + jpeg
                    + b"\r\n"
                )
                sent += 1
            await asyncio.sleep(interval)

    return StreamingResponse(
        _frames(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
    )
