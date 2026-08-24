"""Full-stack entrypoint: pipeline thread + API + dashboard in one process.

Examples:
    python src/api/main.py --source data/videos/vtest.avi --max-frames 600
    python src/api/main.py --config config/cameras.yaml --host 0.0.0.0 --port 8000

Dashboard:  http://localhost:8000/
API docs:   http://localhost:8000/docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surveillance API + dashboard server")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--zones", default="config/zones.yaml")
    parser.add_argument("--alerts", default="config/alert_config.yaml")
    parser.add_argument(
        "--source", default=None, help="Single-source override (video file or device index)"
    )
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"])
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    import uvicorn

    from src.api.app import create_app
    from src.core.state_hub import StateHub
    from src.main import _write_override
    from src.pipeline import SurveillancePipeline
    from src.utils.logger import get_logger, setup_logging

    setup_logging()
    logger = get_logger("api")

    args = parse_args()
    cameras_cfg = _write_override(args.source) if args.source else args.config

    hub = StateHub()
    pipeline = SurveillancePipeline(
        cameras_config=cameras_cfg,
        zones_config=args.zones,
        model_config={"detection": {"model_path": args.model}},
        alerts_config_path=args.alerts,
        device=args.device,
        enable_reid=True,
        state_hub=hub,
    )
    app = create_app(hub=hub, pipeline=pipeline)

    logger.info("Dashboard: http://%s:%d/ | Docs: /docs", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
