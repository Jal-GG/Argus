"""Quick demo: run the pipeline on a video file, webcam, or synthetic scene.

Synthetic mode needs no camera, no YOLO, and no floor plan — it validates the
full tracking → Re-ID → mapping → rules chain on any machine:

    python scripts/demo.py --synthetic
    python scripts/demo.py --source data/videos/sample.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Detection


class SyntheticScene:
    """Renders moving colored 'persons' and emits their ground-truth boxes."""

    def __init__(self, width: int = 960, height: int = 540, n_people: int = 3) -> np.ndarray:
        self.width, self.height = width, height
        rng = np.random.default_rng(7)
        self.people = []
        for _i in range(n_people):
            self.people.append(
                {
                    "x": float(rng.integers(80, width - 160)),
                    "y": float(rng.integers(height // 3, height - 140)),
                    "w": 46.0,
                    "h": 130.0,
                    "vx": float(rng.choice([-1.6, 1.4, 2.0])),
                    "vy": float(rng.choice([-0.5, 0.6])),
                    "color": tuple(int(c) for c in rng.integers(40, 220, 3)),
                }
            )
        self.frame_idx = 0

    def step(self) -> tuple[np.ndarray, list[Detection]]:
        canvas = np.full((self.height, self.width, 3), 235, dtype=np.uint8)
        cv2.putText(
            canvas, "SYNTHETIC SCENE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2
        )

        dets: list[Detection] = []
        for p in self.people:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            if p["x"] < 20 or p["x"] + p["w"] > self.width - 20:
                p["vx"] *= -1
            if p["y"] < self.height // 3 or p["y"] + p["h"] > self.height - 30:
                p["vy"] *= -1

            x1, y1 = int(p["x"]), int(p["y"])
            x2, y2 = int(p["x"] + p["w"]), int(p["y"] + p["h"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), p["color"], -1)
            # Head circle so HOG/color cues look person-ish.
            cv2.circle(
                canvas,
                ((x1 + x2) // 2, y1 + 14),
                16,
                tuple(min(255, c + 25) for c in p["color"]),
                -1,
            )
            dets.append(
                Detection(bbox=(float(x1), float(y1), float(x2), float(y2)), confidence=0.92)
            )
        self.frame_idx += 1
        return canvas, dets


def main() -> int:
    parser = argparse.ArgumentParser(description="Surveillance system demo")
    parser.add_argument("--source", default=None, help="Video path or device index")
    parser.add_argument(
        "--synthetic", action="store_true", help="Run built-in synthetic scene (no models/cameras)"
    )
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.synthetic:
        return _run_synthetic(args)

    from src.main import _write_override
    from src.pipeline import SurveillancePipeline
    from src.utils.logger import get_logger, setup_logging

    setup_logging()
    logger = get_logger("demo")
    if args.source is None:
        logger.error("Provide --source <video|camera-index> or use --synthetic")
        return 1

    pipeline = SurveillancePipeline(
        cameras_config=_write_override(str(args.source)),
        zones_config=PROJECT_ROOT / "config" / "zones.yaml",
        model_config={"detection": {"model_path": "yolo11m.pt"}},
        device=args.device,
        enable_reid=True,
        enable_display=False,
    )
    processed = pipeline.run(max_frames=args.max_frames, display=args.display)
    logger.info("Demo finished: %d frames", processed)
    return 0


def _run_synthetic(args) -> int:
    """Full pipeline minus cameras/detector — tracker/ReID/mapping/rules."""
    from src.core.reid.embedder import create_embedder
    from src.core.tracking.tracker_manager import TrackerManager
    from src.utils.logger import get_logger, setup_logging
    from src.utils.visualization import draw_tracks

    setup_logging()
    logger = get_logger("demo")

    scene = SyntheticScene()
    embedder = create_embedder("auto")  # color-grid fallback on CPU boxes
    trackers = TrackerManager(
        ["cam_001"],
        tracker_config={"track_high_thresh": 0.5, "new_track_thresh": 0.6},
        embedder=embedder.embed,
    )

    ts = time.time()
    for _ in range(args.max_frames):
        frame, dets = scene.step()
        tracks = trackers.update("cam_001", dets, frame=frame, timestamp=ts)
        vis = draw_tracks(frame.copy(), tracks)

        ids = [t.track_id for t in tracks]
        cv2.putText(
            vis,
            f"tracks={ids}",
            (20, vis.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 100, 255),
            2,
        )

        if args.display:
            cv2.imshow("Synthetic Demo", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        ts += 1 / 25.0

    stable = len({t.track_id for t in trackers.trackers["cam_001"].tracks}) >= 1
    logger.info(
        "Synthetic demo done. Track IDs seen: %s",
        sorted({t.track_id for t in trackers.trackers["cam_001"].tracks}),
    )
    if args.display:
        cv2.destroyAllWindows()
    return 0 if stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
