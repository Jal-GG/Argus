"""Hardware-agnostic face detection + recognition engine.

Backend ladder (resolved once at startup, logged honestly):

  Tier 2  YuNet ONNX detector      + SFace ONNX embedder   (full recognition)
  Tier 1  Haar cascade (bundled)   + SFace ONNX embedder   (recognition)
  Tier 0  nothing available        -> ``available == False``

The system never fabricates identities: without an embedder the engine
reports unavailable and downstream stages skip matching entirely.

Model files auto-download into ``models_dir`` on first use (YuNet ~350 KB,
SFace ~37 MB) from the OpenCV zoo. Air-gapped sites can pre-place the files.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...utils.logger import get_logger

logger = get_logger(__name__)

# Models live in the OpenCV zoo via Git LFS; raw URLs return pointer stubs,
# so we fetch through the LFS media endpoint.
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# SFace reference operating point (OpenCV samples): cosine distance <= 0.363
DEFAULT_MATCH_DISTANCE = 0.363


@dataclass
class FaceBox:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float


class FaceEngine:
    """Detects faces and produces comparable embeddings; CPU-first."""

    def __init__(
        self,
        models_dir: str | Path = "src/models/face",
        min_face_size: int = 40,
        auto_download: bool = True,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.min_face_size = int(min_face_size)

        self._yunet_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        self._sface_path = self.models_dir / "face_recognition_sface_2021dec.onnx"

        if auto_download:
            self._ensure_model(self._yunet_path, YUNET_URL, expected_min_bytes=100_000)
            self._ensure_model(self._sface_path, SFACE_URL, expected_min_bytes=1_000_000)

        # ---- Detector ladder ------------------------------------------ #
        self.detector_backend: str | None = None
        self._yunet: cv2.FaceDetectorYN | None = None
        self._haar: cv2.CascadeClassifier | None = None

        if self._yunet_path.exists():
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(self._yunet_path), "", (320, 320), score_threshold=0.6
                )
                self.detector_backend = "yunet"
            except cv2.error as exc:
                logger.warning("YuNet init failed (%s); falling back to Haar", exc)

        if self.detector_backend is None:
            haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if haar_path.exists():
                self._haar = cv2.CascadeClassifier(str(haar_path))
                self.detector_backend = "haar"

        # ---- Embedder -------------------------------------------------- #
        self.embedder_backend: str | None = None
        self._sface: cv2.FaceRecognizerSF | None = None
        if self._sface_path.exists():
            try:
                self._sface = cv2.FaceRecognizerSF.create(str(self._sface_path), "")
                self.embedder_backend = "sface"
                self.feature_dim = 128
            except cv2.error as exc:
                logger.warning("SFace init failed: %s", exc)

        self.available = self.detector_backend is not None and self.embedder_backend is not None
        logger.info(
            "FaceEngine ready: available=%s detector=%s embedder=%s",
            self.available,
            self.detector_backend,
            self.embedder_backend,
        )

    # ------------------------------------------------------------------ #
    def _ensure_model(self, path: Path, url: str, expected_min_bytes: int) -> bool:
        if path.exists() and path.stat().st_size >= expected_min_bytes * 0.5:
            return True
        logger.info("Downloading %s ...", url.rsplit("/", 1)[-1])
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < expected_min_bytes * 0.5:
                raise OSError(f"suspiciously small download ({tmp.stat().st_size} B)")
            tmp.replace(path)
            return True
        except Exception as exc:
            logger.warning("Model download failed (%s) — tier degraded", exc)
            tmp.unlink(missing_ok=True)
            return False

    # ------------------------------------------------------------------ #
    def detect(self, bgr: np.ndarray) -> list[FaceBox]:
        """Detect frontal faces in a BGR crop/frame."""
        h, w = bgr.shape[:2]
        if h < 40 or w < 40:
            return []

        if self._yunet is not None:
            self._yunet.setInputSize((w, h))
            ok, faces = self._yunet.detect(bgr)
            out: list[FaceBox] = []
            if ok and faces is not None:
                for f in faces:
                    x, y, fw, fh = (int(v) for v in f[:4])
                    if fw >= self.min_face_size or fh >= self.min_face_size:
                        out.append(FaceBox((x, y, fw, fh), float(f[14])))
            return out

        if self._haar is not None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rects = self._haar.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(self.min_face_size,) * 2
            )
            return [FaceBox(tuple(int(v) for v in r), 0.7) for r in rects]

        return []

    def align_crop(self, image_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
        x, y, bw, bh = box.bbox
        h, w = image_bgr.shape[:2]
        x, y = max(0, x), max(0, y)
        return image_bgr[y : min(h, y + bh), x : min(w, x + bw)]

    # ------------------------------------------------------------------ #
    def embed(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """128-d L2-normalized feature vector; ``None`` when unsupported."""
        if self._sface is None or face_bgr.size == 0:
            return None
        aligned = (
            cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
            if face_bgr.shape[:2] != (112, 112)
            else face_bgr
        )
        vec = self._sface.feature(aligned.astype(np.float32))
        vec = np.asarray(vec, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else None

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

    @staticmethod
    def head_region_from_person_bbox(
        bbox: tuple[float, float, float, float],
        frame_shape: tuple[int, ...],
        top_fraction: float = 0.28,
        width_fraction: float = 0.7,
    ) -> tuple[int, int, int, int]:
        """Estimate the head search region from a full-person box."""
        x1, y1, x2, y2 = bbox
        ph = max(1.0, y2 - y1)
        pw = max(1.0, x2 - x1)
        cx = (x1 + x2) / 2.0

        hw = pw * width_fraction
        hh = ph * top_fraction
        hx1 = int(max(0, cx - hw / 2))
        hy1 = int(max(0, y1 - hh * 0.25))  # slight upward slack
        hx2 = int(min(frame_shape[1], cx + hw / 2))
        hy2 = int(min(frame_shape[0], y1 + hh))
        return hx1, hy1, max(hx2 - hx1, 8), max(hy2 - hy1, 8)


def create_face_engine(config: dict | None = None, auto_download: bool | None = None) -> FaceEngine:
    cfg = dict((config or {}).get("face_recognition", {}))
    models_dir = ((config or {}).get("models", {}) or {}).get("dir", "src/models/face")
    dl = (
        auto_download
        if auto_download is not None
        else bool(os.getenv("SURVEILLANCE_FR_AUTODL", "1"))
    )
    return FaceEngine(
        models_dir=models_dir,
        min_face_size=int(cfg.get("min_face_size_px", 40)),
        auto_download=dl,
    )
