"""Identity management routes (Phase 11, opt-in face recognition)."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/identities", tags=["identity"])

_IMAGE_FILE = File(...)  # module-level singleton (FastAPI B008 pattern)


def _require_fr(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    registry = getattr(pipeline, "identity_registry", None) if pipeline else None
    if registry is None:
        raise HTTPException(
            503,
            "Face recognition disabled — enable it in config/identity.yaml and restart",
        )
    return registry


class EnrollRequest(BaseModel):
    name: str
    roles: list[str] = []


@router.get("")
def list_identities(request: Request) -> dict:
    registry = _require_fr(request)
    pipeline = request.app.state.pipeline
    engine = getattr(pipeline, "face_engine", None)
    return {
        "available": bool(getattr(engine, "available", False)),
        "detector": getattr(engine, "detector_backend", None),
        "embedder": getattr(engine, "embedder_backend", None),
        "identities": registry.list_identities(),
    }


@router.post("/enroll")
async def enroll(
    request: Request,
    image: UploadFile = _IMAGE_FILE,
    name: str = "",
    roles: str = "",
) -> dict:
    """Enroll a face from an uploaded JPEG/PNG.

    ``roles`` is a comma-separated string (multipart limitation), e.g.
    ``roles=security,admin``. Multiple uploads for the same name add more
    templates and improve robustness.
    """
    registry = _require_fr(request)
    if not name.strip():
        raise HTTPException(422, "Query param 'name' is required")

    payload = await image.read()
    arr = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not decode image")

    pipeline = request.app.state.pipeline
    engine = pipeline.face_engine
    if not getattr(engine, "available", False):
        raise HTTPException(503, "Face models unavailable (Tier 0)")

    faces = engine.detect(bgr)
    if not faces:
        raise HTTPException(422, "No face found in the uploaded image")
    best = max(faces, key=lambda f: f.score)
    face_img = engine.align_crop(bgr, best)
    embedding = engine.embed(face_img)
    if embedding is None:
        raise HTTPException(503, "Embedder unavailable")

    role_list = [r.strip() for r in roles.split(",") if r.strip()]
    ident = registry.enroll(name=name, embeddings=[embedding], roles=role_list)
    return {
        "enrolled": ident.name,
        "roles": ident.roles,
        "num_templates": len(ident.template_ids),
    }


@router.delete("/{name}")
def delete_identity(request: Request, name: str) -> dict:
    registry = _require_fr(request)
    if not registry.remove(name):
        raise HTTPException(404, f"Unknown identity '{name}'")
    # Drop cached identity bindings so alerts reflect removal immediately.
    pipeline = request.app.state.pipeline
    if hasattr(pipeline, "person_identities"):
        pipeline.person_identities = {
            gid: (n, r) for gid, (n, r) in pipeline.person_identities.items() if n != name
        }
    return {"deleted": name}
