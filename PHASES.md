# 📋 Implementation Phases

Step-by-step build plan for the Multi-Camera Person Tracking & Floor-Plan Mapping System.
Each phase ships working, tested code before the next begins.

> **Rev 2 (Aug 2026)** — Plan modernized from the original 2023-era blueprint.
> Key changes: YOLOv8/v9 → **YOLO11/12** · DeepSORT → **BYTE two-stage / BoT-SORT-style association**
> · unmaintained torchreid-only Re-ID → **pluggable embedder (OSNet ⇄ CLIP-ReID ⇄ fallback)**
> · Python 3.8 → **3.10–3.12** · black/flake8/isort → **ruff** · Phase 0 scaffolding added ·
> detection+tracking merged into one GPU stage to avoid duplicate model loads.

---

## 📊 Overview & Status

| Phase | Name | Status | Complexity | Priority |
|-------|------|--------|------------|----------|
| **Phase 0** | Project scaffolding (config loader, logger, shared types) | ✅ Done | ⭐ | Critical |
| **Phase 1** | Camera Integration & Stream Processing | ✅ Done | ⭐⭐ | Critical |
| **Phase 2** | Person Detection (YOLO11 + graceful CPU fallback) | ✅ Done | ⭐⭐ | Critical |
| **Phase 3** | Single-Camera Tracking (Kalman + BYTE association) | ✅ Done | ⭐⭐⭐ | Critical |
| **Phase 4** | Cross-Camera Re-Identification (pluggable embedder) | ✅ Done | ⭐⭐⭐⭐ | Critical |
| **Phase 5** | Camera Calibration & Spatial Mapping (homography) | ✅ Done | ⭐⭐⭐ | Critical |
| **Phase 6** | Path Tracking Engine | ✅ Done | ⭐⭐⭐ | Critical |
| **Phase 7** | Rules Engine & Alert Dispatcher | ✅ Done | ⭐⭐ | High |
| **Phase 8** | Dashboard & API (FastAPI + WebSocket + canvas UI) | ✅ Done | ⭐⭐⭐ | High |
| **Phase 9** | Advanced Features | ✅ Done* (face-recog & 3D deferred — need GPU-scale deps) | ⭐⭐⭐⭐⭐ | Medium |
| **Phase 10** | Testing, Optimization & Deployment (Docker) | ✅ Done* (image build blocked by local network) | ⭐⭐⭐ | High |
| **Phase 11** | Hardware-Agnostic Face Recognition (opt-in) | ✅ Done | ⭐⭐⭐⭐ | Optional-High |

Legend: ✅ Done · 🟡 In Progress · ⬜ Pending

---

## Phase 0: Project Scaffolding

### 🎯 Objectives
- Shared dataclasses (`Detection`, `Track`, `TrackedObject`) used by every stage
- Config loader with YAML + `.env` merge and validation
- Structured logging (loguru-style formatting on stdlib `logging`)

### 📦 Deliverables
- `src/core/types.py` — shared dataclasses
- `src/utils/logger.py`, `src/utils/config.py`, `src/utils/geometry.py`

---

## Phase 1: Camera Integration & Stream Processing

### 🎯 Objectives
- Connect RTSP / USB / video-file sources via a unified interface
- Dedicated reader thread per camera, latest-frame queue (drop-old policy)
- Auto-reconnect with backoff; per-camera stats (fps, drops)

### 📦 Deliverables
- `src/core/camera/camera.py`
- `src/core/camera/stream_reader.py` (threaded capture worker)
- `src/core/camera/camera_manager.py`

### Design notes (modernized)
- OpenCV `VideoCapture` behind a `StreamReader` thread; queue holds **only the latest frame**
  so slow consumers never build latency (classic mistake in CCTV pipelines).
- Source parsing accepts `int` device index, file path, or `rtsp://` URL.
- `FFMPEG` env tuning for RTSP over TCP to avoid UDP packet-loss artifacts.

---

## Phase 2: Person Detection

### 🎯 Objectives
- YOLO11/12 via `ultralytics` (person class only)
- Graceful degradation: if ultralytics/torch unavailable, fall back to OpenCV HOG person detector
- Return typed `Detection` objects incl. crop-ready bboxes

### 📦 Deliverables
- `src/core/detection/detector.py` (protocol + factory)
- `src/core/detection/yolo_detector.py`
- `src/core/detection/hog_detector.py` (dependency-free fallback)

---

## Phase 3: Single-Camera Tracking

### 🎯 Objectives
- Constant-velocity Kalman filter on bounding boxes (state `x, y, a, h`)
- **BYTE two-stage association** (high-conf first pass, low-conf second pass) — the core of
  ByteTrack/BoT-SORT — instead of 2017-era DeepSORT greedy matching
- Optional appearance gating slot (Re-ID embeddings from Phase 4)
- Track lifecycle: tentative → confirmed → lost → deleted

### 📦 Deliverables
- `src/core/tracking/kalman_filter.py`
- `src/core/tracking/track.py`
- `src/core/tracking/byte_tracker.py`
- `src/core/tracking/tracker_manager.py`

### Design notes (modernized)
- Association solved with `scipy.optimize.linear_sum_assignment` (Hungarian).
- No `deep-sort-realtime` dependency; in-repo tracker = fully testable + tunable.

---

## Phase 4: Cross-Camera Re-Identification

### 🎯 Objectives
- Pluggable embedder protocol with three implementations:
  1. **OSNet** (torchreid) when available — best accuracy
  2. **CLIP-ReID** style embeddings (future)
  3. **Color-histogram grid embedder** — zero-dependency fallback so pipeline runs anywhere
- Global ID manager with feature galleries, cosine matching, and
  **spatio-temporal constraints** (same person can't be 50 m apart within 1 s across cameras)

### 📦 Deliverables
- `src/core/reid/embedder.py` (protocol + color-grid fallback)
- `src/core/reid/osnet_embedder.py` (optional import)
- `src/core/reid/global_id_manager.py`

---

## Phase 5: Camera Calibration & Spatial Mapping

### 🎯 Objectives
- Per-camera homography (pixel → floor-plan meters), RANSAC-stabilized
- Interactive calibration tool (click ≥4 correspondences)
- Floor-plan manager: zones (required/restricted/safe), point-in-zone queries, rendering

### 📦 Deliverables
- `src/core/mapping/coordinate_mapper.py`
- `src/core/mapping/floor_plan.py`
- `scripts/camera_calibration.py`, `scripts/zone_drawer.py`

---

## Phase 6: Path Tracking Engine

### 🎯 Objectives
- Per-global-ID trajectory aggregation with zone-visit ledger
- Speed/distance/dwell statistics; stale-path reaper

### 📦 Deliverables
- `src/core/path/path_tracker.py`
- `src/core/path/database.py` (SQLite via SQLAlchemy, async-ready schema)

---

## Phase 7: Rules Engine & Alert Dispatcher

### 🎯 Objectives
- Declarative zone rules from `config/zones.yaml`:
  `required_zone` · `restricted_zone` · `loitering` · `speed_anomaly`
- Channel dispatcher: console/log always; webhook/email/SMS/Telegram opt-in

### 📦 Deliverables
- `src/core/rules/rule_engine.py`
- `src/alerts/alert_manager.py` + channel handlers

---

## Phase 8: Dashboard & API Development

### 🎯 Objectives
- FastAPI REST + WebSocket broadcast of tracks/events
- Lightweight canvas dashboard served by FastAPI (no Node build step required initially;
  React/Vite optional later)
- MJPEG snapshot endpoints per camera

### 📦 Deliverables
- `src/api/main.py`, `src/api/routes/*`, `src/api/schemas/*`
- `src/dashboard/app.py` + static canvas UI

---

## Phase 9: Advanced Features (Optional)

- Face recognition verification (ArcFace/insightface)
- Path prediction (LSTM/Transformer over trajectories)
- Anomaly detection on movement patterns
- Heat-map generation & historical playback
- 3D building visualization (Three.js)

---

## Phase 10: Testing, Optimization & Deployment

### 🎯 Activities
- pytest unit suite (geometry, Kalman, association, Re-ID matching, homography, rules)
- Integration smoke test on synthetic video
- ONNX Runtime / TensorRT export for edge deployment
- Docker + docker-compose with GPU profile
- ruff lint gate in CI

---

## 📈 Progress Tracking

- [x] Phase 0: Project scaffolding
- [x] Phase 1: Camera Integration
- [x] Phase 2: Person Detection
- [x] Phase 3: Single-Camera Tracking
- [x] Phase 4: Cross-Camera Re-ID
- [x] Phase 5: Spatial Mapping
- [x] Phase 6: Path Tracking
- [x] Phase 7: Rules & Alerts
- [x] Phase 8: Dashboard & API
- [x] Phase 9: Advanced Features* — SQLite history/playback, heat-maps,
      path prediction (velocity + zone-Markov), movement anomaly detection.
      Deferred: 3D visualization.
      ~~Deferred: face recognition~~ → moved to Phase 11.
- [x] Phase 10: Testing & Deployment
- [x] Phase 11: Face Recognition — Tier ladder verified on CPU:
      YuNet detected faces in real photo; SFace embeddings self-match
      dist=0.000 / scaled 0.018; different-person rejected at 0.868
      (threshold 0.363). Role-gated zone rules enforced via registry.

### Verified working (Aug 2026)
- `pytest tests` → 106 passed · ruff lint/format clean
- Live smoke: heatmap JPEG from live traffic · history DB persisting ~3.1k
  positions/40 s across 14 identities · WebSocket pushing velocity-based
  path predictions for 6/7 tracked people · anomaly events persisted

### Verified working (Aug 2026)
- `pytest tests` → 62 passed (incl. API + metrics suites)
- Full pipeline on `data/videos/vtest.avi`: YOLO11n → BYTE → Re-ID → homography
  → zone visits (`python src/main.py --source data/videos/vtest.avi --max-frames 200`)
- Live server smoke test: `/api/health` ok · MJPEG stream · floor-plan render
  · WebSocket pushing live tracks @ ~3 Hz · dashboard assets served
- Benchmark (`scripts/benchmark.py`): YOLO11n ≈ 18.6 FPS CPU @ 768×576;
  tracking association <1 ms; color-grid embedder negligible
- Deployment artifacts: Dockerfile + compose (CPU/GPU/monitoring profiles),
  GitHub Actions CI (ruff + pytest matrix), Prometheus `/metrics`,
  `docs/DEPLOYMENT.md`. *Note: image build untested here — Docker Hub TLS is
  intercepted on this network.*
- `ruff check` / `ruff format` clean

### Quick start (full stack)
```bash
python src/api/main.py --source data/videos/vtest.avi
# Dashboard: http://127.0.0.1:8000/   API docs: /docs   Metrics: /metrics
```

---

**Start here:** run `python scripts/demo.py --source <video-or-camera>` once Phases 1–6 are checked.

---

## Phase 11: Hardware-Agnostic Face Recognition (Opt-In)

> **Design principle:** recognition must work on *any* hardware — or explicitly
> report itself unavailable. It never degrades into guessing identities.

### 🎯 Objectives
- Face detection + recognition that runs **CPU-only by default**, GPU when present
- Tiered backend ladder with guaranteed graceful degradation
- Identity registry (name ↔ roles ↔ face templates) powering the existing
  `allowed_roles` zone rules
- Strict privacy posture: disabled by default, explicit enrollment, local-only
  template storage

### 🔧 Backend Ladder

| Tier | Detection | Embedding | Needs | Result |
|------|-----------|-----------|-------|--------|
| **2** | YuNet ONNX (~350 KB) | SFace ONNX (~37 MB, 128-d) | one-time auto-download; CPU OK | full recognition |
| **1** | Haar cascade (**bundled with OpenCV**) | SFace ONNX | model download | recognition, weaker boxes |
| **0** | — | — | nothing | FR reports `unavailable`; pipeline unaffected |

Resolution order: best available at startup → logged. If SFace weights can't be
downloaded (air-gapped sites), Tier 0 is honest instead of fabricating matches.

### 📦 Deliverables
- `src/core/identity/face_engine.py` — detector/embedder ladder + model fetcher
- `src/core/identity/registry.py` — enroll/match/persist identities & roles
- `scripts/enroll_face.py` — CLI enrollment from image file or live camera
- API: `GET /api/identities`, `POST /api/identities/enroll`, `DELETE /api/identities/{name}`
- Rules integration: `allowed_roles` enforced against enrolled roles;
  `strict_unknown_roles` option for high-security zones
- Dashboard: enrolled names shown on floor-plan markers

### ⚙️ Configuration (`config/identity.yaml`)
```yaml
face_recognition:
  enabled: false              # OFF until explicitly opted in
  match_threshold: 0.363      # cosine distance (OpenCV SFace reference)
  sample_every_n_frames: 10   # per-track sampling rate (CPU budget)
  min_face_size_px: 40
  strict_unknown_roles: false # treat unrecognized people as role-less
models:
  dir: src/models/face        # auto-downloaded ONNX files live here
```

### 🔒 Privacy guardrails
- Biometric templates stay on-device (`data/identities/`); no cloud calls
- Enrollment is a deliberate operator action per person (consent artifact)
- Deleting an identity removes its templates immediately
- Documentation points operators to GDPR/BIPA review before enabling

---
