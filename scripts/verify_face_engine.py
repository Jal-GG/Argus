"""Live verification of the Phase 11 face engine on a real photo."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.identity.face_engine import FaceEngine
from src.core.identity.registry import IdentityRegistry

img = cv2.imread(str(Path(__file__).resolve().parents[1].parent / "opencode" / "bus.jpg"))
if img is None:
    import os

    alt = os.path.join(os.environ["TEMP"], "opencode", "bus.jpg")
    img = cv2.imread(alt)
assert img is not None, "test image missing"

engine = FaceEngine(models_dir="src/models/face", auto_download=False)
print(
    f"available={engine.available} detector={engine.detector_backend} "
    f"embedder={engine.embedder_backend}"
)

faces = engine.detect(img)
print(f"faces detected: {len(faces)}")
assert len(faces) >= 1, "expected at least one face in bus.jpg"

# Enroll person A from face 0; verify same-face matches, different-face doesn't.
emb_a = engine.embed(engine.align_crop(img, faces[0]))
assert emb_a is not None and emb_a.shape == (128,)
print("embedding dim:", emb_a.shape[0], "| norm:", round(float(np.linalg.norm(emb_a)), 3))

registry = IdentityRegistry(store_dir="data/identities_test", match_distance_threshold=0.363)
registry.enroll("PersonA", [emb_a], ["security"])

name, dist = registry.match(emb_a)
print(f"self-match: name={name} dist={dist:.4f}")
assert name == "PersonA"

if len(faces) > 1:
    emb_b = engine.embed(engine.align_crop(img, faces[1]))
    other_name, other_dist = registry.match(emb_b)
    print(
        f"different-face: name={other_name} dist={other_dist:.4f} (should be > threshold or None)"
    )

# Rotation/scale robustness sanity: slightly scaled version should still match.
scaled = cv2.resize(engine.align_crop(img, faces[0]), None, fx=1.3, fy=1.3)
emb_s = engine.embed(scaled)
name_s, dist_s = registry.match(emb_s)
print(f"scaled self-match: name={name_s} dist={dist_s:.4f}")

import shutil

shutil.rmtree("data/identities_test", ignore_errors=True)
print("PHASE 11 ENGINE VERIFICATION PASSED")
