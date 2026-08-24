"""Generate demo assets: a synthetic floor plan + a plausible calibration.

The homography maps the walkable ground region of ``data/videos/vtest.avi``
(image size 768x576) onto the plan's courtyard rectangle, letting the full
detection → tracking → Re-ID → mapping → rules chain run without real
site data:

    python scripts/generate_demo_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLAN_W, PLAN_H = 800, 600


def build_floor_plan(path: Path) -> None:
    """Render a simple annotated building plan."""
    plan = np.full((PLAN_H, PLAN_W, 3), 245, dtype=np.uint8)

    # Outer walls
    cv2.rectangle(plan, (40, 40), (760, 560), (60, 60, 60), 3)
    # Rooms
    rooms = {
        "Security Office": ((50, 50), (160, 160)),
        "Lobby": ((330, 180), (610, 460)),
        "Server Room": ((650, 90), (760, 210)),
    }
    for name, (p1, p2) in rooms.items():
        cv2.rectangle(plan, p1, p2, (140, 140, 140), 2)
        cv2.putText(
            plan,
            name,
            (p1[0] + 4, p1[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    # Courtyard where pedestrians are mapped
    cv2.rectangle(plan, (110, 170), (690, 500), (90, 130, 90), 2)
    cv2.putText(
        plan,
        "Courtyard / Walkway",
        (250, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (90, 130, 90),
        2,
        cv2.LINE_AA,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), plan)
    print(f"Wrote {path}")


def build_calibration(path: Path, image_size=(768, 576)) -> None:
    """Homography: image ground band -> plan courtyard rectangle."""
    iw, ih = image_size
    src = np.float32(
        [
            [0, ih],  # bottom-left of frame
            [iw - 1, ih],  # bottom-right
            [int(iw * 0.25), int(ih * 0.55)],
            [int(iw * 0.75), int(ih * 0.55)],
        ]
    )
    dst = np.float32(
        [
            [110, 500],  # courtyard bottom-left
            [690, 500],  # courtyard bottom-right
            [230, 260],
            [570, 260],
        ]
    )

    H, _ = cv2.findHomography(src, dst, 0)

    calib = {
        "camera_id": "cam_001",
        "floor_id": "floor_0",
        "source_video": "data/videos/vtest.avi",
        "note": "Approximate demo calibration; replace via scripts/camera_calibration.py",
        "homography_matrix": [[float(v) for v in row] for row in H],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(calib, fh, sort_keys=False)
    print(f"Wrote {path}")


def main() -> int:
    build_floor_plan(PROJECT_ROOT / "data" / "floor_plans" / "floor_0.png")
    build_calibration(PROJECT_ROOT / "config" / "calibration_cam_001.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
