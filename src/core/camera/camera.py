"""Camera source abstraction for RTSP / USB / video-file inputs."""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

from ...utils.logger import get_logger

logger = get_logger(__name__)

# Prefer TCP for RTSP to avoid UDP packet-loss artifacts; lower latency buffering.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer")


def parse_source(source: str | int) -> str | int:
    """Coerce YAML string sources into what VideoCapture expects."""
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    return text


class Camera:
    """A single video source with a dedicated reader thread.

    The reader keeps only the most recent frame so downstream processing never
    accumulates latency — essential for multi-camera CCTV workloads.
    """

    def __init__(
        self,
        camera_id: str,
        source: str | int,
        fps: float = 15.0,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = -1,
    ) -> None:
        self.id = str(camera_id)
        self.source = parse_source(source)
        self.target_fps = float(fps)
        self.reconnect_delay = float(reconnect_delay)
        self.max_reconnect_attempts = int(max_reconnect_attempts)

        self._cap: cv2.VideoCapture | None = None
        self._reader: StreamReader | None = None
        self.is_running = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> bool:
        """Open the capture device/stream. Returns True on success."""
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            logger.error("Failed to open source for camera %s: %s", self.id, self.source)
            return False
        self._cap = cap
        return True

    def start(self) -> bool:
        """Open the stream and spawn the reader thread."""
        if self.is_running:
            return True
        if not self.connect():
            return False
        assert self._cap is not None
        self._reader = StreamReader(camera=self, cap=self._cap)
        self._reader.start()
        self.is_running = True
        logger.info("Camera %s started (source=%s)", self.id, self.source)
        return True

    def stop(self) -> None:
        """Stop reading and release the capture."""
        self.is_running = False
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera %s stopped", self.id)

    # ------------------------------------------------------------------ #
    # Frame access
    # ------------------------------------------------------------------ #
    def read(self) -> tuple[float, np.ndarray] | None:
        """Fetch the latest ``(timestamp, frame)``, or ``None`` if none new."""
        if self._reader is None:
            return None
        return self._reader.latest_frame()

    @property
    def actual_fps(self) -> float:
        if self._reader is None:
            return 0.0
        return self._reader.measured_fps

    @property
    def dropped_frames(self) -> int:
        if self._reader is None:
            return 0
        return self._reader.dropped


class StreamReader:
    """Background thread that continuously reads frames from a capture."""

    def __init__(self, camera: Camera, cap: cv2.VideoCapture) -> None:
        import threading

        self._camera = camera
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._timestamp: float = 0.0
        self._frame_seq: int = 0
        self._consumed_seq: int = 0
        self.dropped = 0
        self._measured_fps = 0.0
        self._last_tick = time.time()
        self._tick_frames = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        import threading

        self._thread = threading.Thread(
            target=self._run, name=f"stream-{self._camera.id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest_frame(self) -> tuple[float, np.ndarray] | None:
        """Return the newest unread frame; older ones are dropped."""
        with self._lock:
            if self._frame is None or self._frame_seq == self._consumed_seq:
                return None
            self._consumed_seq = self._frame_seq
            self.dropped += max(0, self._frame_seq - self._consumed_seq)
            return self._timestamp, self._frame.copy()

    @property
    def measured_fps(self) -> float:
        return self._measured_fps

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        min_interval = 1.0 / max(self._camera.target_fps, 0.1)
        next_read = time.time()

        while not self._stop_event.is_set():
            now = time.time()
            if now < next_read:
                time.sleep(min(next_read - now, 0.005))
                continue
            next_read = now + min_interval

            ok, frame = self._cap.read()
            if not ok or frame is None:
                if not self._handle_read_failure():
                    break
                continue

            timestamp = time.time()
            with self._lock:
                if self._frame is not None:
                    self.dropped += 1
                self._frame = frame
                self._timestamp = timestamp
                self._frame_seq += 1
                self._tick_frames += 1

            # Rolling FPS estimate once per second.
            elapsed = timestamp - self._last_tick
            if elapsed >= 1.0:
                self._measured_fps = self._tick_frames / elapsed
                self._tick_frames = 0
                self._last_tick = timestamp

    def _handle_read_failure(self) -> bool:
        """Attempt reconnection with backoff. Returns False when giving up."""
        attempts = 0
        while not self._stop_event.is_set() and (
            self._camera.max_reconnect_attempts < 0
            or attempts < self._camera.max_reconnect_attempts
        ):
            wait = self._camera.reconnect_delay * (attempts + 1)
            logger.warning(
                "Camera %s: read failure, retrying in %.1fs (attempt %d)",
                self._camera.id,
                wait,
                attempts + 1,
            )
            if self._stop_event.wait(wait):
                return False
            attempts += 1

            src_is_file = (
                isinstance(self._camera.source, (str, Path))
                and Path(str(self._camera.source)).is_file()
            )
            self._cap.release()
            if self._camera.connect():
                # Video files should loop rather than be treated as dropouts.
                if src_is_file:
                    return True
                return True
        return False
