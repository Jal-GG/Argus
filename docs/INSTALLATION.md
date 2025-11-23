# Installation Guide

This guide will help you set up the Multi-Camera Person Tracking & Floor-Plan Mapping System.

## Prerequisites

### Hardware Requirements

**Minimum**:
- CPU: 4 cores (Intel i5 or equivalent)
- RAM: 8GB
- GPU: NVIDIA GTX 1060 6GB (or equivalent with CUDA support)
- Storage: 256GB SSD

**Recommended**:
- CPU: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- RAM: 16GB+
- GPU: NVIDIA RTX 3060 12GB or better
- Storage: 512GB NVMe SSD

### Software Requirements

- Operating System: Ubuntu 20.04+, Windows 10+, or macOS
- Python 3.8 or higher
- CUDA 11.7+ (for GPU acceleration)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/surveillance-system.git
cd surveillance-system
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (with CUDA support)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

```bash
# This will download YOLOv8 and Re-ID models
python scripts/download_models.py
```

### 5. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

### 6. Configure Cameras

Edit `config/cameras.yaml` to add your camera sources:

```yaml
cameras:
  - id: "cam_001"
    name: "Main Entrance"
    source: "rtsp://admin:password@192.168.1.10:554/stream"
    fps: 15
    resolution: [1920, 1080]
    location: "Building A - Ground Floor"
    floor_id: "floor_0"
    enabled: true
```

### 7. Set Up Database

```bash
# For SQLite (default, no setup needed)
# Database will be created automatically

# For PostgreSQL:
# 1. Install PostgreSQL
# 2. Create database:
createdb surveillance

# 3. Update DATABASE_URL in .env
# DATABASE_URL=postgresql://user:password@localhost:5432/surveillance

# Run migrations
python scripts/init_database.py
```

### 8. Verify Installation

```bash
# Test camera connection
python scripts/test_cameras.py

# Test detection
python scripts/test_detection.py

# Run system check
python scripts/system_check.py
```

## GPU Setup

### NVIDIA GPU (CUDA)

1. Install NVIDIA drivers:
```bash
# Ubuntu
sudo ubuntu-drivers autoinstall

# Check installation
nvidia-smi
```

2. Install CUDA Toolkit (11.7 or higher):
   - Download from: https://developer.nvidia.com/cuda-downloads

3. Verify CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### TensorRT (Optional, for optimization)

```bash
# Install TensorRT
pip install tensorrt

# Convert YOLO to TensorRT
python scripts/optimize_models.py --model yolo --backend tensorrt
```

## Docker Installation (Alternative)

If you prefer using Docker:

```bash
# Build image
docker-compose build

# Run services
docker-compose up -d

# View logs
docker-compose logs -f
```

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size in `config/model_config.yaml`
- Use smaller YOLO model (yolov8n or yolov8s)
- Enable frame skipping

### Camera Connection Failed

- Check camera URL and credentials
- Ensure camera is accessible from your network
- Try accessing camera URL in VLC player first

### Slow Performance

- Enable GPU acceleration
- Use TensorRT optimization
- Reduce camera FPS
- Enable frame skipping

### Import Errors

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade

# Check for conflicts
pip check
```

## Next Steps

- [Configuration Guide](CONFIGURATION.md)
- [Camera Calibration](../scripts/camera_calibration.py)
- [Zone Setup](../scripts/zone_drawer.py)
- [API Documentation](API.md)

## Support

If you encounter issues:
- Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- Open an issue on GitHub
- Contact support
