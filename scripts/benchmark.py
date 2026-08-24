"""Performance benchmark: detector, tracker, Re-ID, and full-pipeline throughput.

Usage:
    python scripts/benchmark.py --source data/videos/vtest.avi
    python scripts/benchmark.py --synthetic --frames 300
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.detection.detector import create_detector
from src.core.reid.embedder import create_embedder
from src.core.tracking.tracker_manager import TrackerManager
from src.utils.logger import get_logger, setup_logging

logger = get_logger("benchmark")


def _load_frames(source: str | int, n: int) -> list[np.ndarray]:
    if source == "synthetic":
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from demo import SyntheticScene

        scene = SyntheticScene()
        return [scene.step()[0] for _ in range(n)]

    cap = cv2.VideoCapture(source)
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {source}")
    return frames


def _rate(fn, frames) -> tuple[float, float]:
    """Run fn(frame) over frames; returns (fps, mean_ms)."""
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        fn(frame)
        times.append(time.perf_counter() - t0)
    mean_s = statistics.mean(times[2:] or times)  # skip warmup frames
    return 1.0 / mean_s, mean_s * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline performance benchmark")
    parser.add_argument("--source", default="data/videos/vtest.avi")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    args = parser.parse_args()

    setup_logging()
    source = "synthetic" if args.synthetic else args.source
    frames = _load_frames(source, args.frames)
    print(
        f"\nBenchmarking on {len(frames)} frames "
        f"({frames[0].shape[1]}x{frames[0].shape[0]}) from {source}\n"
    )

    rows: list[tuple[str, float, float]] = []

    # ---- Detector ---------------------------------------------------- #
    detector = create_detector(model_path=args.model, device=args.device, confidence_threshold=0.4)
    fps, ms = _rate(lambda f: detector.detect(f), frames)
    rows.append((f"Detection ({args.model}, {args.device or 'auto'})", fps, ms))

    detections_per_frame = [detector.detect(frames[0]) for _ in range(1)]
    n_det = len(detections_per_frame[0])

    # ---- Tracker ------------------------------------------------------ #
    tracker = TrackerManager(["bench"], tracker_config={"track_high_thresh": 0.5})
    ts = 0.0

    def run_tracker(frame):
        nonlocal ts
        dets = detector.detect(frame)
        tracker.update("bench", dets, frame=frame, timestamp=ts)
        ts += 1 / 25.0

    fps, ms = _rate(run_tracker, frames)
    rows.append(("Tracking stage (detect+associate+embed)", fps, ms))

    # ---- Re-ID embedder alone ---------------------------------------- #
    embedder = create_embedder("auto")
    crops = []
    for det in detections_per_frame[0][:6]:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        crops.append(frames[0][y1:y2, x1:x2])
    if crops:
        reps = max(10, len(frames) // 4)
        t0 = time.perf_counter()
        for _ in range(reps):
            embedder.embed(crops)
        per_call = (time.perf_counter() - t0) / reps * 1000
        rows.append(
            (
                f"Re-ID embedder ({type(embedder).__name__}, {len(crops)} crops)",
                1000.0 / per_call,
                per_call,
            )
        )

    # ---- Report ------------------------------------------------------- #
    print(f"{'Stage':<48}{'FPS':>9}{'ms/frame':>12}")
    print("-" * 69)
    for name, fps_v, ms_v in rows:
        print(f"{name:<48}{fps_v:>9.1f}{ms_v:>12.1f}")
    print(f"\nDetections on first frame: {n_det}")
    print("Note: full-pipeline FPS ≈ 1/(sum of stage ms + camera/render overhead).")

    detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
