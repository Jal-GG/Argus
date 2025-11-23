# Quick Start Guide

Get the Multi-Camera Person Tracking System running in 10 minutes!

## 🚀 Fast Setup

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/surveillance-system.git
cd surveillance-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Downloads YOLOv8 and Re-ID models automatically
python scripts/download_models.py
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit if needed (defaults work for testing)
```

### 4. Run Demo with Sample Video

```bash
# Test with a sample video file
python scripts/demo.py --video data/videos/sample.mp4
```

## 🎥 Using Your Own Cameras

### Option 1: IP Camera (RTSP)

Edit `config/cameras.yaml`:

```yaml
cameras:
  - id: "my_camera"
    name: "My Camera"
    source: "rtsp://admin:password@192.168.1.100:554/stream"
    fps: 15
    enabled: true
```

Run:
```bash
python src/main.py --config config/cameras.yaml
```

### Option 2: USB Webcam

Edit `config/cameras.yaml`:

```yaml
cameras:
  - id: "webcam"
    name: "Webcam"
    source: 0  # Device index (0, 1, 2, ...)
    fps: 30
    enabled: true
```

### Option 3: Video File

```yaml
cameras:
  - id: "video"
    name: "Video File"
    source: "/path/to/your/video.mp4"
    fps: 30
    enabled: true
```

## 🗺️ Camera Calibration & Floor Plan

### 1. Prepare Floor Plan Image

- Save your building's floor plan as PNG/JPG
- Place in `data/floor_plans/`

### 2. Calibrate Cameras

```bash
# Interactive calibration tool
python scripts/camera_calibration.py cam_001 frame.jpg floor_plan.png
```

Follow on-screen instructions:
1. Click 4+ points on camera view
2. Press 's' to switch
3. Click corresponding points on floor plan
4. Press 'c' to compute and save

### 3. Define Zones

```bash
# Interactive zone drawing tool
python scripts/zone_drawer.py data/floor_plans/floor_plan.png
```

Or edit `config/zones.yaml` manually.

## 🚨 Setting Up Alerts

### Email Alerts

Edit `.env`:

```bash
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
```

Edit `config/alert_config.yaml`:

```yaml
enabled_channels:
  - email

email:
  enabled: true
  to_emails:
    - "admin@yourdomain.com"
```

### Webhook Alerts

```yaml
webhook:
  enabled: true
  endpoints:
    - url: "https://your-server.com/api/alerts"
      method: "POST"
```

## 🎮 Running the Full System

### Basic Mode (Detection + Tracking Only)

```bash
python src/main.py --mode basic
```

### Full Mode (With Re-ID and Mapping)

```bash
python src/main.py --mode full --config config/cameras.yaml
```

### With Web Dashboard

```bash
# Start API server
python src/api/main.py

# In another terminal, start dashboard
cd src/dashboard
python app.py

# Access at: http://localhost:8000
```

## 📊 Testing Individual Components

### Test Camera Connection

```bash
python scripts/test_cameras.py
```

### Test Person Detection

```bash
python scripts/test_detection.py
```

### Test Tracking

```bash
python scripts/test_tracking.py
```

### Test Re-ID

```bash
python scripts/test_reid.py
```

## 🐳 Docker (Alternative)

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Access dashboard at: http://localhost:8000
```

## 🔧 Common Issues

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA drivers and PyTorch with CUDA
```

### Camera Connection Failed

- Verify camera URL in browser or VLC
- Check network connectivity
- Ensure credentials are correct

### Slow Performance

- Use smaller YOLO model: `yolov8n.pt` instead of `yolov8m.pt`
- Reduce FPS in camera config
- Enable GPU acceleration

## 📚 Next Steps

1. **Calibrate cameras** - [Camera Calibration Guide](docs/INSTALLATION.md#camera-calibration)
2. **Define zones** - [Zone Configuration](config/zones.yaml)
3. **Set up alerts** - [Alert Configuration](config/alert_config.yaml)
4. **Deploy** - [Deployment Guide](docs/DEPLOYMENT.md)

## 🆘 Need Help?

- 📖 [Full Documentation](README.md)
- 🔧 [Installation Guide](docs/INSTALLATION.md)
- 🏗️ [Architecture Details](ARCHITECTURE.md)
- 📋 [Phase-by-Phase Guide](PHASES.md)
- 💬 [GitHub Issues](https://github.com/yourusername/surveillance-system/issues)

---

**Happy Tracking! 🎥📍**
