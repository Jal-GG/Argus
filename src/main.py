"""Main entrypoint for the surveillance pipeline.

Examples:
    python src/main.py --config config/cameras.yaml --display
    python src/main.py --source data/videos/sample.mp4 --max-frames 300
    python src/main.py --source 0 --display --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline import SurveillancePipeline
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-camera person tracking pipeline")
    parser.add_argument("--config", default="config/cameras.yaml", help="Path to cameras.yaml")
    parser.add_argument("--zones", default="config/zones.yaml", help="Path to zones.yaml")
    parser.add_argument(
        "--alerts", default="config/alert_config.yaml", help="Path to alert_config.yaml"
    )
    parser.add_argument(
        "--source", default=None, help="Single-source override: video file path or device index"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=[None, "cpu", "cuda", "mps"],
        help="Force inference device",
    )
    parser.add_argument("--model", default="yolo11m.pt", help="YOLO weights")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["basic", "full"],
        help="basic = detect+track only; full = +ReID+mapping+rules",
    )
    parser.add_argument("--display", action="store_true", help="Show live windows")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    cameras_cfg = args.config
    if args.source is not None:
        cameras_cfg = _write_override(args.source)

    pipeline = SurveillancePipeline(
        cameras_config=cameras_cfg,
        zones_config=args.zones,
        model_config={
            "detection": {"model_path": args.model},
            "reid": {"backend": "auto" if args.mode == "full" else "none"},
            "tracking": {"track_high_thresh": 0.5, "new_track_thresh": 0.6},
        },
        alerts_config_path=args.alerts,
        device=args.device,
        enable_reid=(args.mode == "full"),
        enable_display=args.display,
    )

    processed = pipeline.run(max_frames=args.max_frames, display=args.display)
    logger.info("Processed %d frame(s). Done.", processed)
    return 0


def _write_override(source: str) -> str:
    """Create a temp cameras.yaml wrapping a single --source."""
    import tempfile

    cfg_text = (
        f"cameras:\n"
        f"  - id: cam_001\n"
        f"    name: Override\n"
        f'    source: "{source}"\n'
        f"    fps: 25\n"
        f"    enabled: true\n"
        f"settings:\n"
        f"  reconnect_delay: 3\n"
        f"  max_reconnect_attempts: -1\n"
    )
    tmp = Path(tempfile.gettempdir()) / "surveillance_override_cameras.yaml"
    tmp.write_text(cfg_text, encoding="utf-8")
    logger.info("Using single-source override -> %s", source)
    return str(tmp)


if __name__ == "__main__":
    raise SystemExit(main())
