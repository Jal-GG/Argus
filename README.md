# 🎥 Multi-Camera Person Tracking & Floor-Plan Mapping System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A real-time surveillance system that connects multiple CCTV cameras, performs cross-camera person identification using deep learning, and maps synchronized movement onto a 2D floor plan. Using zone-based rules, it verifies if persons reached permitted destinations or entered restricted areas and generates alerts accordingly.

---

## 🧠 Core Concept

This system replicates **airport-style security tracking** for residential societies, office buildings, or any multi-camera environment:

```
Camera 1 ──┐
Camera 2 ──┼──> Person Detection ──> Re-Identification ──> Map Coordinates ──> Path Tracking ──> Alert Engine
Camera 3 ──┘         (YOLO)           (DeepReID)         (Homography)        (Floor Plan)      (Rules)
```

### What It Does

1. **Detects** people in every camera feed
2. **Tracks** them within each camera using unique IDs
3. **Identifies** the same person across different cameras (Re-ID)
4. **Maps** their position onto the building's floor plan
5. **Tracks** their movement path in real-time
6. **Validates** against zone rules (required/restricted areas)
7. **Alerts** when suspicious behavior or rule violations occur

---

## 🎯 Use Cases

| Scenario | Application |
|----------|-------------|
| 🏢 **Delivery Verification** | Ensure delivery person reached the correct door |
| 🚫 **Restricted Zones** | Alert when unauthorized person enters sensitive areas |
| 🏠 **Society Security** | Track visitor movement from gate to apartment |
| 🏭 **Industrial Safety** | Monitor if workers followed designated paths |
| 🏥 **Hospital Tracking** | Track patient/staff movement across floors |
| 🎓 **Campus Security** | Monitor student movement in restricted hours |

---

## ✨ Key Features

### 🔍 Core Features
- ✅ Multi-camera integration (RTSP/IP/USB/Video files)
- ✅ Real-time person detection using YOLOv8/v9
- ✅ Cross-camera person re-identification
- ✅ 2D floor plan mapping with homography projection
- ✅ Live path tracking and visualization
- ✅ Zone-based rule engine
- ✅ Multi-channel alert system (SMS/Email/Webhook)

### 🚀 Advanced Features
- ⚡ Multi-person concurrent tracking
- ⚡ Face recognition integration
- ⚡ Path prediction using LSTM
- ⚡ Anomaly detection
- ⚡ 3D building visualization
- ⚡ Heat map generation
- ⚡ Historical playback

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CAMERA LAYER                              │
│  [Cam1]  [Cam2]  [Cam3]  [Cam4]  ...  [CamN]                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                   DETECTION & TRACKING                           │
│  • YOLOv8/v9 Person Detection                                   │
│  • DeepSORT/ByteTrack (Per-Camera Tracking)                     │
│  • Bounding Box Extraction                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                  RE-IDENTIFICATION LAYER                         │
│  • OSNet/FastReID Feature Extraction                            │
│  • Global ID Management                                          │
│  • Cross-Camera Identity Matching                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                   SPATIAL MAPPING LAYER                          │
│  • Camera Calibration (Homography Matrix)                       │
│  • Pixel → World Coordinate Conversion                          │
│  • Floor Plan Integration                                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                    PATH TRACKING ENGINE                          │
│  • Real-time Path Aggregation                                   │
│  • Movement History Storage                                      │
│  • Live Visualization on Floor Plan                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                     RULE ENGINE & ALERTS                         │
│  • Zone Definition (Required/Restricted/Safe)                   │
│  • Path Validation Logic                                         │
│  • Multi-Channel Alerting (SMS/Email/Webhook/Telegram)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
surveillance-system/
├── README.md                      # This file
├── ARCHITECTURE.md                # Detailed technical architecture
├── PHASES.md                      # Implementation phases guide
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── .env.example                   # Environment variables template
│
├── config/                        # Configuration files
│   ├── cameras.yaml              # Camera definitions & calibration
│   ├── zones.yaml                # Zone rules & coordinates
│   ├── model_config.yaml         # ML model configurations
│   └── alert_config.yaml         # Alert channel settings
│
├── src/                          # Source code
│   ├── core/                     # Core modules
│   │   ├── camera/              # Camera integration
│   │   ├── detection/           # Person detection (YOLO)
│   │   ├── tracking/            # Per-camera tracking (DeepSORT)
│   │   ├── reid/                # Re-identification models
│   │   ├── mapping/             # Coordinate mapping
│   │   ├── path/                # Path tracking engine
│   │   └── rules/               # Rule engine
│   │
│   ├── models/                   # ML models & weights
│   │   ├── yolo/
│   │   ├── reid/
│   │   └── face/
│   │
│   ├── utils/                    # Utilities
│   │   ├── video_utils.py
│   │   ├── image_utils.py
│   │   ├── geometry.py
│   │   └── logger.py
│   │
│   ├── api/                      # REST API (FastAPI)
│   │   ├── routes/
│   │   └── schemas/
│   │
│   ├── dashboard/                # Web dashboard
│   │   ├── static/
│   │   ├── templates/
│   │   └── app.py
│   │
│   └── alerts/                   # Alert handlers
│       ├── sms.py
│       ├── email.py
│       ├── webhook.py
│       └── telegram.py
│
├── data/                         # Data storage
│   ├── floor_plans/             # Building floor plan images
│   ├── calibration/             # Camera calibration data
│   ├── videos/                  # Sample videos
│   └── output/                  # Processed results
│
├── tests/                        # Unit tests
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_reid.py
│   └── test_mapping.py
│
├── scripts/                      # Utility scripts
│   ├── camera_calibration.py   # Camera calibration tool
│   ├── zone_drawer.py          # Interactive zone definition
│   ├── benchmark.py            # Performance testing
│   └── demo.py                 # Quick demo
│
├── docs/                         # Documentation
│   ├── INSTALLATION.md
│   ├── CONFIGURATION.md
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── TROUBLESHOOTING.md
│
└── notebooks/                    # Jupyter notebooks
    ├── 01_detection_demo.ipynb
    ├── 02_reid_training.ipynb
    └── 03_calibration_guide.ipynb
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenCV 4.x
- PyTorch 2.x

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/surveillance-system.git
cd surveillance-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py

# Configure cameras and zones
cp .env.example .env
# Edit .env with your settings
```

### Run Demo

```bash
# Run with sample video
python scripts/demo.py --video data/videos/sample.mp4

# Run with live cameras
python src/main.py --config config/cameras.yaml
```

---

## 📋 Implementation Phases

This project is designed to be built in **8 progressive phases**:

| Phase | Focus | Duration | Deliverable |
|-------|-------|----------|-------------|
| **Phase 1** | Camera Integration | 1 week | Multi-camera stream pipeline |
| **Phase 2** | Person Detection | 1 week | Real-time YOLO detection |
| **Phase 3** | Single-Camera Tracking | 1 week | DeepSORT tracking |
| **Phase 4** | Re-Identification | 2 weeks | Cross-camera person matching |
| **Phase 5** | Spatial Mapping | 1 week | Floor plan coordinate mapping |
| **Phase 6** | Path Tracking | 1 week | Live path visualization |
| **Phase 7** | Rules & Alerts | 1 week | Zone-based alert system |
| **Phase 8** | Dashboard & API | 2 weeks | Web interface & REST API |

**Total Timeline**: ~10 weeks for full implementation

📖 **Detailed phase breakdown**: See [PHASES.md](PHASES.md)

---

## 🛠️ Technology Stack

### Computer Vision & ML
- **Detection**: YOLOv8/v9, Detectron2
- **Tracking**: DeepSORT, ByteTrack, SORT
- **Re-ID**: OSNet, FastReID, MGN
- **Face Recognition**: FaceNet, ArcFace (optional)

### Core Libraries
- **OpenCV**: Image processing & homography
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **SciPy**: Spatial algorithms

### Backend & API
- **FastAPI**: REST API server
- **SQLite/PostgreSQL**: Database
- **Redis**: Caching & pub/sub
- **Celery**: Background tasks

### Frontend & Visualization
- **React/Vue.js**: Dashboard UI
- **Three.js**: 3D visualization (optional)
- **Plotly/D3.js**: Charts & heatmaps
- **WebSocket**: Real-time updates

### DevOps & Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **NVIDIA Docker**: GPU support
- **Kubernetes**: Production scaling (optional)

---

## 🎓 Learning Resources

- 📚 [Person Re-Identification: Past, Present and Future](https://arxiv.org/abs/1610.02984)
- 📚 [Deep Learning for Person Re-identification: A Survey and Outlook](https://arxiv.org/abs/2001.04193)
- 🎥 [YOLOv8 Official Docs](https://docs.ultralytics.com/)
- 🎥 [DeepSORT Explained](https://nanonets.com/blog/object-tracking-deepsort/)
- 📄 [Camera Calibration with OpenCV](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

---

## 🤝 Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- DeepSORT by nwojke
- FastReID by JDAI-CV
- OSNet by KaiyangZhou

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/surveillance-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/surveillance-system/discussions)
- **Email**: your.email@example.com

---

**⚠️ Disclaimer**: This system is intended for authorized security and surveillance purposes only. Ensure compliance with local privacy laws and regulations. Always obtain proper consent before deploying in any environment.

---

Made with ❤️ for safer communities
