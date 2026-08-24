# 🚀 Deployment Guide

Production deployment options for the Surveillance System, from a single
workstation to GPU servers and edge devices.

---

## 1. Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.12 |
| GPU (optional) | GTX 1060 6GB | RTX 3060+ 12GB |
| RAM | 8 GB | 16 GB |
| Storage | 20 GB SSD | 512 GB NVMe |

CPU-only mode works out of the box (YOLO11n ≈ 18–20 FPS @ 768×576 on a modern
laptop CPU — see `python scripts/benchmark.py`).

---

## 2. Bare-metal / VM install

```bash
git clone <repo> && cd surveillance-system
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Full stack (pipeline thread + REST API + dashboard)
python src/api/main.py --config config/cameras.yaml --host 0.0.0.0 --port 8000
```

Run as a service with **systemd**:

```ini
# /etc/systemd/system/surveillance.service
[Unit]
Description=Surveillance System
After=network-online.target

[Service]
WorkingDirectory=/opt/surveillance-system
ExecStart=/opt/surveillance-system/venv/bin/python src/api/main.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
User=svc-surveillance

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now surveillance
journalctl -u surveillance -f          # logs
curl localhost:8000/api/health         # probe
```

---

## 3. Docker (recommended)

```bash
# CPU stack
docker compose up --build -d

# Logs & health
docker compose logs -f surveillance
curl localhost:8000/api/health
```

Volumes keep state outside the container:

| Host path | Container path | Purpose |
|-----------|----------------|---------|
| `./config` | `/app/config` | cameras/zones/alerts YAML (hot-editable) |
| `./data` | `/app/data` | floor plans, recordings, outputs |

Single-video demo without editing config:

```yaml
environment:
  SURVEILLANCE_SOURCE: /app/data/videos/vtest.avi
```

### GPU profile

Requires NVIDIA Container Toolkit on the host:

```bash
docker compose --profile gpu up --build -d
```

### Monitoring profile (Prometheus + Grafana)

```bash
docker compose --profile monitoring up -d
# Prometheus: http://localhost:9090  (target: surveillance:8000/metrics)
# Grafana:    http://localhost:3000
```

Exposed metrics:

| Metric | Type | Meaning |
|--------|------|---------|
| `surveillance_frames_total{camera}` | counter | processed frames per camera |
| `surveillance_events_total{rule_type}` | counter | rule violations raised |
| `surveillance_active_tracks` | gauge | confirmed tracks right now |
| `surveillance_people_on_plan` | gauge | distinct global IDs mapped |
| `surveillance_cameras_online` | gauge | healthy camera count |
| `surveillance_pipeline_running` | gauge | pipeline thread alive |

---

## 4. Edge / ONNX deployment

For fan-out on machines without CUDA or for lower latency:

```bash
# Export once (on any machine with ultralytics)
python scripts/export_model.py --model yolo11n.pt --verify

# Serve with the ONNX file; set in model_config.yaml:
detection:
  model_path: src/models/yolo/yolo11n.onnx
```

Tips:
- `--imgsz 480` trades accuracy for ~40% more FPS on small scenes.
- TensorRT (`model.export(format="engine")`) gives another 2–3× on Jetson/dGPUs.

---

## 5. Reverse proxy (TLS termination)

```nginx
server {
    listen 443 ssl;
    server_name cctv.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }

    # WebSocket upgrade for /ws
    location /ws {
        proxy_pass http://127.0.0.1:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
    }

    # Long-lived MJPEG streams need buffering off
    location ~ /api/cameras/.*/stream$ {
        proxy_pass http://127.0.0.1:8000;
        proxy_buffering off;
    }
}
```

---

## 6. Operational checklist

- [ ] `/api/health` wired into your uptime monitor (or use the built-in HEALTHCHECK)
- [ ] Prometheus scraping `/metrics`; alert if `surveillance_pipeline_running == 0`
- [ ] Camera credentials via `.env` (never committed); see `.env.example`
- [ ] Log rotation configured (`LOG_LEVEL`, journal/docker log drivers)
- [ ] Data retention policy for `data/output` snapshots
- [ ] Verify zone rules after every `config/zones.yaml` change (coordinates are plan-pixels)
- [ ] Privacy review: signage/consent per local regulations before enabling alerts

### Known environment note
On networks where Docker Hub TLS is intercepted by a corporate proxy,
`docker build/pull` fails with `tls: protocol version not supported`. Either
allow-list `registry-1.docker.io` + `auth.docker.io`, configure the daemon's
proxy settings, or pre-load a saved image:
`docker save surveillance-system | gzip > image.tgz` → transfer → `docker load`.
