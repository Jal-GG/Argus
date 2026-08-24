# ---------------------------------------------------------------------------
# Surveillance System — CPU inference image (GPU notes at bottom)
# Build:  docker build -t surveillance-system .
# Run:    docker run --rm -p 8000:8000 \
#           -v $(pwd)/config:/app/config:ro \
#           -v $(pwd)/data:/app/data \
#           surveillance-system
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"

# OpenCV runtime libs (opencv wheels bundle ffmpeg; system libs still needed).
RUN apt-get update \
        -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
# CPU-only torch keeps the image ~10x smaller than the CUDA default wheel.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY requirements.txt ruff.toml setup.py ./

# Non-root runtime user.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data/output && chown -R appuser:appuser /app/data || true
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

# Full stack: pipeline thread + REST API + WebSocket + dashboard.
ENTRYPOINT ["python", "src/api/main.py"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--device", "cpu"]

# ---------------------------------------------------------------------------
# GPU variant (NVIDIA Container Toolkit required):
#   FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04  (or use base + CUDA torch)
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
#   docker run --gpus all ... --device cuda
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Edge/ONNX variant:
#   pip install onnxruntime  (replace ultralytics predict with ORT backend)
#   see scripts/export_model.py to produce the engine file
# ---------------------------------------------------------------------------
