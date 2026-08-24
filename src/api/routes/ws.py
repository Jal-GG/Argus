"""WebSocket live state broadcast."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["ws"])

PUSH_INTERVAL_S = 0.35


@router.websocket("/ws")
async def ws_state(websocket: WebSocket) -> None:
    """Pushes the full live snapshot ~3x/second; client sends 'ping' to test."""
    await websocket.accept()
    hub = websocket.app.state.hub
    logger.info("WS client connected")
    try:
        while True:
            payload = hub.live_snapshot()
            await websocket.send_json(payload)

            # Allow a short receive window so clients can ping/close.
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=PUSH_INTERVAL_S)
                if msg == "close":
                    break
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        logger.info("WS client disconnected")
    except Exception as exc:  # pragma: no cover - transport level
        logger.debug("WS error: %s", exc)
