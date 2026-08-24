"""Shared dataclasses used across all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Bounding box as (x1, y1, x2, y2)
BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    """A single person detection produced by a detector."""

    bbox: BBox
    confidence: float
    frame_id: int = -1
    camera_id: str = ""

    @property
    def tlwh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)


@dataclass(slots=True)
class Track:
    """A tracked object within a single camera view."""

    track_id: int
    camera_id: str
    bbox: BBox
    confidence: float
    is_confirmed: bool = True
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    # Latest appearance embedding (set by the Re-ID stage when enabled).
    feature: np.ndarray | None = field(default=None, repr=False)
    # Assigned by the GlobalIDManager after cross-camera matching.
    global_id: int | None = None
    # Phase 11: filled by the identity stage when face recognition is on.
    name: str | None = None
    roles: tuple[str, ...] = ()


@dataclass(slots=True)
class WorldObservation:
    """A tracked object projected onto floor-plan world coordinates."""

    global_id: int
    camera_id: str
    track_id: int
    timestamp: float
    world_x: float
    world_y: float
    zones: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(slots=True)
class FramePacket:
    """A frame plus its source metadata flowing through the pipeline."""

    camera_id: str
    frame_id: int
    timestamp: float
    image: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image.shape
