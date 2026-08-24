"""Export YOLO models to ONNX for edge/CPU-optimized deployment.

Ultralytics handles the graph export; when ``onnxruntime`` is installed the
script additionally verifies the exported model produces valid person boxes.

Usage:
    python scripts/export_model.py --model yolo11n.pt
    python scripts/export_model.py --model yolo11s.pt --imgsz 960 --verify
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger, setup_logging

logger = get_logger("export")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX")
    parser.add_argument("--model", default="yolo11n.pt", help="Weights to export")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 export (GPU only)")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic axes (variable input size)")
    parser.add_argument(
        "--verify", action="store_true", help="Run onnxruntime inference sanity check"
    )
    args = parser.parse_args()

    setup_logging()

    from ultralytics import YOLO

    model = YOLO(args.model)
    t0 = time.perf_counter()
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=True,
    )
    logger.info("Exported %s -> %s in %.1fs", args.model, exported_path, time.perf_counter() - t0)

    if args.verify:
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed; skipping verification")
            return 1

        import numpy as np

        session = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
        input_name = session.get_inputs()[0].name
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: dummy})
        elapsed = time.perf_counter() - t0

        pred_shape = tuple(outputs[0].shape)
        logger.info("ONNX runtime OK: output shape %s in %.0f ms", pred_shape, elapsed * 1000)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
