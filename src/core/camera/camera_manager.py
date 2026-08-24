"""Multi-camera management: load config, start/stop all streams, poll frames."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from typing_extensions import Self

from ...utils.config import load_cameras_config
from ...utils.logger import get_logger
from .camera import Camera

logger = get_logger(__name__)


class CameraManager:
    """Owns the lifecycle of every configured camera."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.cameras: dict[str, Camera] = {}

        raw = load_cameras_config(self.config_path)

        # Global settings live under `settings:` in cameras.yaml.
        import yaml

        with open(self.config_path, encoding="utf-8") as fh:
            full = yaml.safe_load(fh) or {}
        settings_block = full.get("settings", {}) or {}

        buffer_reconnect = float(settings_block.get("reconnect_delay", 5.0))
        max_attempts = int(settings_block.get("max_reconnect_attempts", -1))

        for entry in raw:
            cam = Camera(
                camera_id=entry["id"],
                source=entry["source"],
                fps=float(entry.get("fps", 15.0)),
                reconnect_delay=buffer_reconnect,
                max_reconnect_attempts=max_attempts,
            )
            self.cameras[cam.id] = cam
        logger.info("Loaded %d camera(s) from %s", len(self.cameras), self.config_path)

    def start_all(self) -> int:
        """Start every camera; returns the number successfully running."""
        started = 0
        for cam in self.cameras.values():
            if cam.start():
                started += 1
        return started

    def stop_all(self) -> None:
        for cam in self.cameras.values():
            cam.stop()

    def read_frames(self) -> dict[str, tuple[float, np.ndarray]]:
        """Poll every camera once, returning only sources with fresh frames."""
        frames: dict[str, tuple[float, np.ndarray]] = {}
        for cam_id, cam in self.cameras.items():
            packet = cam.read()
            if packet is not None:
                frames[cam_id] = packet
        return frames

    def iter_frames(self) -> Iterator[tuple[str, float, np.ndarray]]:
        """Blocking generator yielding ``(camera_id, timestamp, frame)``."""
        import time

        idle_sleep = 0.005
        while True:
            got_any = False
            for cam_id, (ts, frame) in self.read_frames().items():
                got_any = True
                yield cam_id, ts, frame
            if not got_any:
                time.sleep(idle_sleep)

    def __enter__(self) -> Self:
        self.start_all()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop_all()
