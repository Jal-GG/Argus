"""FastAPI application factory.

Usage A — API only (attach to an already-running pipeline's hub):
    app = create_app(hub=hub)

Usage B — full stack (pipeline thread + server in one process):
    uvicorn src.api.app:get_app --factory
    # or simply: python src/api/main.py --source data/videos/vtest.avi
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..core.state_hub import StateHub
from ..utils.logger import get_logger
from .routes import identity, metrics, system, tracking, ws

logger = get_logger(__name__)

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard" / "static"


def create_app(
    hub: StateHub | None = None,
    pipeline=None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if pipeline is not None and not getattr(pipeline, "_thread_started", False):
            pipeline.run_in_thread(max_frames=getattr(pipeline, "_max_frames", None))
            pipeline._thread_started = True
            logger.info("Pipeline thread started from app startup")
        yield
        logger.info("API shutdown")

    app = FastAPI(
        title="Surveillance System API",
        version="0.2.0",
        description="Multi-camera person tracking & floor-plan mapping",
        lifespan=lifespan,
    )

    app.state.hub = hub or StateHub()
    app.state.pipeline = pipeline

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(system.router)
    app.include_router(tracking.router)
    app.include_router(metrics.router)
    app.include_router(identity.router)
    app.include_router(ws.router)

    if DASHBOARD_DIR.exists():
        app.mount("/static", StaticFiles(directory=DASHBOARD_DIR, html=True), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> RedirectResponse:
            return RedirectResponse(url="/static/index.html")
    else:  # pragma: no cover
        logger.warning("Dashboard static dir missing: %s", DASHBOARD_DIR)

    return app


def get_app() -> FastAPI:
    """Uvicorn factory target: `uvicorn src.api.app:get_app --factory`.

    Configure via env vars (SURVEILLANCE_SOURCE / SURVEILLANCE_CONFIG) so the
    process boots the full stack without custom code.
    """
    import os

    from ..main import _write_override
    from ..pipeline import SurveillancePipeline

    hub = StateHub()
    source = os.getenv("SURVEILLANCE_SOURCE")
    cameras_cfg = (
        _write_override(source)
        if source
        else os.getenv("SURVEILLANCE_CONFIG", "config/cameras.yaml")
    )
    device = os.getenv("SURVEILLANCE_DEVICE") or None
    max_frames = int(os.getenv("SURVEILLANCE_MAX_FRAMES", "0")) or None

    pipeline = SurveillancePipeline(
        cameras_config=cameras_cfg,
        zones_config=os.getenv("SURVEILLANCE_ZONES", "config/zones.yaml"),
        model_config={"detection": {"model_path": os.getenv("SURVEILLANCE_MODEL", "yolo11n.pt")}},
        device=device,
        enable_reid=True,
        state_hub=hub,
    )
    pipeline._max_frames = max_frames
    return create_app(hub=hub, pipeline=pipeline)
