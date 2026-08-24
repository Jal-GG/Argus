"""Enroll a face into the identity registry (Phase 11).

Sources: an image file, a directory of images, or a live camera snapshot.

    python scripts/enroll_face.py --name "Aman" --roles security,admin \
        --source path/to/face.jpg

    python scripts/enroll_face.py --name "Aman" --roles security \
        --source 0 --count 3          # grab 3 frames from webcam

Requires face_recognition.enabled=true in config/identity.yaml (models
auto-download on first run; ~37 MB).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.identity.face_engine import FaceEngine
from src.core.identity.registry import IdentityRegistry
from src.utils.config import load_yaml
from src.utils.logger import get_logger, setup_logging

logger = get_logger("enroll")


def collect_from_image(engine: FaceEngine, path: Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    faces = engine.detect(bgr)
    if not faces:
        logger.warning("No face found in %s — skipping", path.name)
        return []
    best = max(faces, key=lambda f: f.score)
    emb = engine.embed(engine.align_crop(bgr, best))
    return [emb] if emb is not None else []


def collect_from_camera(engine: FaceEngine, device: int, count: int):
    cap = cv2.VideoCapture(device)
    embeddings = []
    print(f"Capturing {count} samples from camera {device} — look at the lens.")
    while len(embeddings) < count:
        ok, frame = cap.read()
        if not ok:
            break
        faces = engine.detect(frame)
        if faces:
            best = max(faces, key=lambda f: f.score)
            emb = engine.embed(engine.align_crop(frame, best))
            if emb is not None:
                embeddings.append(emb)
                print(f"  captured sample {len(embeddings)}/{count}")
                time.sleep(0.6)
                continue
        time.sleep(0.1)
    cap.release()
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll a face identity")
    parser.add_argument("--name", required=True)
    parser.add_argument("--roles", default="", help="comma-separated roles")
    parser.add_argument(
        "--source", required=True, help="image file, directory of images, or device index"
    )
    parser.add_argument(
        "--count", type=int, default=3, help="samples to capture from a live camera"
    )
    args = parser.parse_args()

    setup_logging()
    cfg_path = PROJECT_ROOT / "config" / "identity.yaml"
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    fr_cfg = dict(cfg.get("face_recognition", {}))

    if not fr_cfg.get("enabled", False):
        logger.error(
            "Face recognition is disabled. Set face_recognition.enabled=true "
            "in config/identity.yaml first."
        )
        return 2

    engine = FaceEngine(
        models_dir=cfg.get("models", {}).get("dir", "src/models/face"),
        min_face_size=int(fr_cfg.get("min_face_size_px", 40)),
    )
    if not engine.available:
        logger.error("Face models unavailable (download failed?) — cannot enroll")
        return 3

    source = args.source
    if source.isdigit():
        embeddings = collect_from_camera(engine, int(source), args.count)
    else:
        src_path = Path(source)
        files = (
            [src_path]
            if src_path.is_file()
            else sorted(
                p for p in src_path.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if src_path.is_dir()
            else []
        )
        embeddings = []
        for f in files:
            embeddings.extend(collect_from_image(engine, f))

    if not embeddings:
        logger.error("No usable face samples collected")
        return 4

    registry = IdentityRegistry(
        store_dir=fr_cfg.get("store_dir", "data/identities"),
        match_distance_threshold=float(fr_cfg.get("match_threshold", 0.363)),
    )
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    ident = registry.enroll(args.name, embeddings, roles)
    print(
        f"OK — enrolled '{ident.name}' with {len(ident.template_ids)} template(s), "
        f"roles={ident.roles}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
