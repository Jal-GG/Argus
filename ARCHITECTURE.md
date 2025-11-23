# 🏗️ Technical Architecture

This document provides an in-depth technical overview of the Multi-Camera Person Tracking & Floor-Plan Mapping System.

---

## 📊 System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                 │
│  Multiple Camera Streams (RTSP/RTMP/USB/Video Files)                │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────────┐
│                      STREAM PROCESSOR                                 │
│  • Frame Extraction                                                   │
│  • Preprocessing (Resize, Normalize)                                  │
│  • Frame Buffer Management                                            │
│  • Multi-threaded Processing                                          │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────────┐
│                      DETECTION MODULE                                 │
│  Model: YOLOv8/v9                                                     │
│  Input: 640x640 RGB frames                                            │
│  Output: [x1,y1,x2,y2,conf,class] for each person                    │
│  Performance: ~30-60 FPS on GPU                                       │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────────┐
│                   PER-CAMERA TRACKING MODULE                          │
│  Algorithm: DeepSORT / ByteTrack                                      │
│  • Kalman Filter for motion prediction                                │
│  • Hungarian Algorithm for assignment                                 │
│  • Appearance features (Re-ID embeddings)                             │
│  Output: Local Track IDs per camera                                   │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────────┐
│                   RE-IDENTIFICATION MODULE                            │
│  Model: OSNet / FastReID                                              │
│  • Extract 512/2048-dim embeddings                                    │
│  • Cosine similarity matching                                         │
│  • Global ID assignment across cameras                                │
│  • Handling occlusions & re-entry                                     │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────────┐
│                    SPATIAL MAPPING MODULE                             │
│  • Homography matrix per camera                                       │
│  • Pixel coordinates → World coordinates                              │
│  • Floor plan overlay                                                 │
│  • Multi-floor support                                                │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────────┐
│                    PATH TRACKING ENGINE                               │
│  • Trajectory aggregation                                             │
│  • Speed & direction analysis                                         │
│  • Dwell time calculation                                             │
│  • Historical path storage (DB)                                       │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────────┐
│                    RULE ENGINE & ANALYTICS                            │
│  • Zone entry/exit detection                                          │
│  • Path validation against rules                                      │
│  • Anomaly detection                                                  │
│  • Event generation                                                   │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────────┐
│                        ALERT SYSTEM                                   │
│  • SMS (Twilio)                                                       │
│  • Email (SMTP)                                                       │
│  • Webhook (HTTP POST)                                                │
│  • Telegram Bot                                                       │
│  • Push Notifications                                                 │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────────┐
│                    WEB DASHBOARD & API                                │
│  • FastAPI REST endpoints                                             │
│  • WebSocket for real-time updates                                    │
│  • React/Vue.js frontend                                              │
│  • Live camera feeds                                                  │
│  • Interactive floor plan                                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Component Details

### 1. Camera Integration Layer

#### Supported Input Types
- RTSP streams (IP cameras)
- RTMP streams
- USB cameras
- Video files (MP4, AVI, MKV)
- Image sequences

#### Implementation
```python
class CameraManager:
    def __init__(self, config):
        self.cameras = {}
        self.load_cameras(config)

    def load_cameras(self, config):
        for cam_config in config['cameras']:
            camera = Camera(
                id=cam_config['id'],
                source=cam_config['source'],
                fps=cam_config['fps'],
                calibration=cam_config['calibration']
            )
            self.cameras[cam_config['id']] = camera

    def read_frames(self):
        # Multi-threaded frame reading
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(cam.read): cam_id
                      for cam_id, cam in self.cameras.items()}

            for future in as_completed(futures):
                cam_id = futures[future]
                frame = future.result()
                yield cam_id, frame
```

#### Key Features
- Automatic reconnection on stream failure
- Frame buffering to handle variable FPS
- Synchronization across multiple cameras
- Hardware acceleration support (NVDEC)

---

### 2. Person Detection Module

#### Model: YOLOv8/v9

**Architecture Choice**:
- YOLOv8-medium: Best balance of speed & accuracy
- Input: 640x640 RGB images
- Output: Person bounding boxes with confidence scores

**Optimization**:
- TensorRT conversion for 2-3x speedup
- Half-precision (FP16) inference
- Batch processing when possible

**Detection Pipeline**:
```python
class PersonDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, classes=[0], conf=self.conf_threshold)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                detections.append({
                    'bbox': box,
                    'confidence': conf
                })

        return detections
```

**Performance Metrics**:
- Inference time: 15-20ms on NVIDIA RTX 3060
- mAP: ~0.85 on COCO person class
- Throughput: 50-60 FPS

---

### 3. Tracking Module (DeepSORT)

#### Algorithm Overview

**Components**:
1. **Kalman Filter**: Predicts object position in next frame
2. **Hungarian Algorithm**: Associates detections with tracks
3. **Appearance Features**: Re-ID embeddings for robustness
4. **Track Management**: Handles track creation, deletion, occlusion

**State Vector** (8-dimensional):
```
[x, y, a, h, vx, vy, va, vh]
where:
  x, y = center coordinates
  a = aspect ratio
  h = height
  vx, vy, va, vh = velocities
```

**Matching Cascade**:
```
Cost = λ * IoU_distance + (1-λ) * Cosine_distance
where:
  IoU_distance = 1 - IoU(box1, box2)
  Cosine_distance = 1 - cosine_similarity(feat1, feat2)
  λ = 0.5 (tunable parameter)
```

**Implementation**:
```python
class Tracker:
    def __init__(self, max_age=30, n_init=3):
        self.max_age = max_age  # Frames to keep track alive
        self.n_init = n_init    # Frames to confirm track
        self.tracks = []
        self.track_id_counter = 0

    def update(self, detections, features):
        # Predict new locations
        for track in self.tracks:
            track.predict()

        # Match detections to tracks
        matches, unmatched_dets, unmatched_tracks = \
            self._match(detections, features)

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                detections[det_idx],
                features[det_idx]
            )

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._initiate_track(detections[det_idx], features[det_idx])

        # Mark unmatched tracks for deletion
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.is_alive()]

        return self.tracks
```

---

### 4. Re-Identification (Re-ID) Module

#### Purpose
Match the same person across different cameras using appearance features.

#### Model: OSNet / FastReID

**Architecture**:
- Input: Person crops (256x128 or 384x128)
- Output: 512 or 2048-dim feature vector
- Backbone: ResNet50 / OSNet
- Loss: Triplet + CrossEntropy

**Feature Extraction**:
```python
class ReIDModel:
    def __init__(self, model_path, device='cuda'):
        self.model = load_pretrained_model(model_path)
        self.model.eval()
        self.device = device

    def extract_features(self, person_crops):
        """
        Args:
            person_crops: List of person images (numpy arrays)
        Returns:
            features: numpy array of shape (N, feature_dim)
        """
        preprocessed = self.preprocess(person_crops)
        with torch.no_grad():
            features = self.model(preprocessed.to(self.device))

        # L2 normalization
        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()
```

**Global Identity Management**:
```python
class GlobalIDManager:
    def __init__(self, similarity_threshold=0.6):
        self.global_ids = {}  # global_id -> feature_gallery
        self.camera_to_global = {}  # (cam_id, local_id) -> global_id
        self.threshold = similarity_threshold
        self.next_id = 1

    def match_or_create(self, cam_id, local_id, feature):
        """
        Match feature to existing global ID or create new one
        """
        max_similarity = -1
        matched_global_id = None

        # Compare with all existing global IDs
        for global_id, gallery in self.global_ids.items():
            similarity = self._compute_similarity(feature, gallery)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_global_id = global_id

        # Create new global ID if no match
        if max_similarity < self.threshold:
            matched_global_id = self.next_id
            self.global_ids[matched_global_id] = [feature]
            self.next_id += 1
        else:
            # Update gallery with new feature
            self.global_ids[matched_global_id].append(feature)
            # Keep only recent N features
            self.global_ids[matched_global_id] = \
                self.global_ids[matched_global_id][-100:]

        self.camera_to_global[(cam_id, local_id)] = matched_global_id
        return matched_global_id

    def _compute_similarity(self, feature, gallery):
        # Cosine similarity with gallery
        similarities = np.dot(gallery, feature)
        return np.max(similarities)
```

**Challenges & Solutions**:

| Challenge | Solution |
|-----------|----------|
| Different viewing angles | Multi-view training data |
| Lighting variations | Data augmentation, normalization |
| Occlusions | Multiple feature gallery, temporal smoothing |
| Similar appearances | Combine with spatial-temporal constraints |
| Re-entry after long time | Soft threshold, time-based decay |

---

### 5. Spatial Mapping Module

#### Camera Calibration

**Goal**: Map pixel coordinates (x, y) in camera frame to world coordinates (X, Y) on floor plan.

**Homography Matrix**:
```
| X |       | h11  h12  h13 | | x |
| Y | = H * | h21  h22  h23 | | y |
| 1 |       | h31  h32  h33 | | 1 |

where H is a 3x3 transformation matrix
```

**Calibration Process**:
1. Identify 4+ corresponding points between camera view and floor plan
2. Solve for homography matrix using DLT (Direct Linear Transform)
3. Validate with additional test points

**Implementation**:
```python
class CameraCalibration:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.H = None  # Homography matrix
        self.floor_id = None

    def calibrate(self, image_points, world_points):
        """
        Args:
            image_points: (N, 2) array of (x,y) in camera frame
            world_points: (N, 2) array of (X,Y) on floor plan
        """
        # Compute homography
        self.H, _ = cv2.findHomography(
            image_points,
            world_points,
            cv2.RANSAC
        )

    def pixel_to_world(self, x, y):
        """Convert camera pixel to world coordinates"""
        if self.H is None:
            raise ValueError("Camera not calibrated")

        point = np.array([[[x, y]]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point, self.H)
        return world_point[0, 0]

    def world_to_pixel(self, X, Y):
        """Convert world coordinates to camera pixel"""
        if self.H is None:
            raise ValueError("Camera not calibrated")

        H_inv = np.linalg.inv(self.H)
        point = np.array([[[X, Y]]], dtype=np.float32)
        pixel_point = cv2.perspectiveTransform(point, H_inv)
        return pixel_point[0, 0]
```

**Floor Plan Integration**:
```python
class FloorPlan:
    def __init__(self, image_path, scale=1.0):
        self.image = cv2.imread(image_path)
        self.scale = scale  # meters per pixel
        self.zones = []

    def add_zone(self, name, polygon, zone_type):
        """
        Args:
            name: Zone name (e.g., "Lobby", "Restricted Area")
            polygon: List of (x,y) coordinates
            zone_type: 'required', 'restricted', 'safe'
        """
        self.zones.append({
            'name': name,
            'polygon': np.array(polygon),
            'type': zone_type
        })

    def check_point_in_zone(self, x, y):
        """Check which zones contain the point"""
        point = (x, y)
        zones_containing_point = []

        for zone in self.zones:
            if cv2.pointPolygonTest(zone['polygon'], point, False) >= 0:
                zones_containing_point.append(zone)

        return zones_containing_point
```

---

### 6. Path Tracking Engine

#### Data Structure
```python
class PersonPath:
    def __init__(self, global_id):
        self.global_id = global_id
        self.positions = []  # [(timestamp, x, y, cam_id)]
        self.zones_visited = set()
        self.first_seen = None
        self.last_seen = None
        self.total_distance = 0.0

    def add_position(self, timestamp, x, y, cam_id):
        """Add new position to path"""
        if self.first_seen is None:
            self.first_seen = timestamp

        self.last_seen = timestamp
        self.positions.append((timestamp, x, y, cam_id))

        # Calculate distance from last position
        if len(self.positions) > 1:
            prev_pos = self.positions[-2]
            dist = np.sqrt((x - prev_pos[1])**2 + (y - prev_pos[2])**2)
            self.total_distance += dist

    def get_current_speed(self, window=5):
        """Calculate speed over last N positions"""
        if len(self.positions) < 2:
            return 0.0

        recent = self.positions[-window:]
        time_diff = recent[-1][0] - recent[0][0]

        if time_diff == 0:
            return 0.0

        dist = sum([
            np.sqrt((recent[i][1] - recent[i-1][1])**2 +
                   (recent[i][2] - recent[i-1][2])**2)
            for i in range(1, len(recent))
        ])

        return dist / time_diff  # meters per second

    def get_dwell_time(self, zone_polygon, threshold_distance=2.0):
        """Calculate time spent in a zone"""
        dwell_time = 0.0
        in_zone = False
        enter_time = None

        for timestamp, x, y, _ in self.positions:
            point = (x, y)
            is_in_zone = cv2.pointPolygonTest(zone_polygon, point, False) >= 0

            if is_in_zone and not in_zone:
                in_zone = True
                enter_time = timestamp
            elif not is_in_zone and in_zone:
                in_zone = False
                dwell_time += timestamp - enter_time

        # Handle case where person is still in zone
        if in_zone:
            dwell_time += self.last_seen - enter_time

        return dwell_time
```

#### Path Storage (Database Schema)
```sql
-- Person tracking
CREATE TABLE persons (
    global_id INTEGER PRIMARY KEY,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    total_distance REAL,
    status VARCHAR(20)  -- 'active', 'completed', 'suspicious'
);

-- Position history
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id INTEGER,
    timestamp TIMESTAMP,
    x REAL,
    y REAL,
    camera_id VARCHAR(50),
    floor_id VARCHAR(50),
    FOREIGN KEY (global_id) REFERENCES persons(global_id)
);

-- Zone visits
CREATE TABLE zone_visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id INTEGER,
    zone_name VARCHAR(100),
    enter_time TIMESTAMP,
    exit_time TIMESTAMP,
    dwell_time REAL,
    FOREIGN KEY (global_id) REFERENCES persons(global_id)
);

-- Events
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id INTEGER,
    event_type VARCHAR(50),  -- 'zone_entry', 'zone_exit', 'alert'
    timestamp TIMESTAMP,
    details JSON,
    FOREIGN KEY (global_id) REFERENCES persons(global_id)
);
```

---

### 7. Rule Engine

#### Rule Types

**1. Required Zone Rules**
```python
class RequiredZoneRule:
    def __init__(self, zone_name, timeout=300):
        self.zone_name = zone_name
        self.timeout = timeout  # seconds

    def evaluate(self, person_path):
        """Check if person visited required zone within timeout"""
        if self.zone_name not in person_path.zones_visited:
            time_elapsed = person_path.last_seen - person_path.first_seen
            if time_elapsed > self.timeout:
                return {
                    'violated': True,
                    'reason': f'Did not visit {self.zone_name} within {self.timeout}s'
                }

        return {'violated': False}
```

**2. Restricted Zone Rules**
```python
class RestrictedZoneRule:
    def __init__(self, zone_polygon, allowed_roles=[]):
        self.zone_polygon = zone_polygon
        self.allowed_roles = allowed_roles

    def evaluate(self, person_path, person_role=None):
        """Check if unauthorized person entered restricted zone"""
        for timestamp, x, y, _ in person_path.positions:
            point = (x, y)
            is_in_zone = cv2.pointPolygonTest(self.zone_polygon, point, False) >= 0

            if is_in_zone and person_role not in self.allowed_roles:
                return {
                    'violated': True,
                    'reason': f'Unauthorized entry at {timestamp}',
                    'position': (x, y)
                }

        return {'violated': False}
```

**3. Loitering Detection**
```python
class LoiteringRule:
    def __init__(self, zone_polygon, max_dwell_time=600):
        self.zone_polygon = zone_polygon
        self.max_dwell_time = max_dwell_time

    def evaluate(self, person_path):
        dwell_time = person_path.get_dwell_time(self.zone_polygon)

        if dwell_time > self.max_dwell_time:
            return {
                'violated': True,
                'reason': f'Loitering for {dwell_time:.0f}s'
            }

        return {'violated': False}
```

**4. Speed Anomaly**
```python
class SpeedAnomalyRule:
    def __init__(self, max_speed=3.0):  # m/s
        self.max_speed = max_speed

    def evaluate(self, person_path):
        speed = person_path.get_current_speed()

        if speed > self.max_speed:
            return {
                'violated': True,
                'reason': f'Abnormal speed: {speed:.2f} m/s'
            }

        return {'violated': False}
```

---

### 8. Alert System

#### Alert Manager
```python
class AlertManager:
    def __init__(self, config):
        self.handlers = {
            'sms': SMSHandler(config['sms']),
            'email': EmailHandler(config['email']),
            'webhook': WebhookHandler(config['webhook']),
            'telegram': TelegramHandler(config['telegram'])
        }

    def send_alert(self, alert_data, channels=['email']):
        """
        Args:
            alert_data: {
                'type': 'restricted_zone' | 'loitering' | 'missing_destination',
                'global_id': int,
                'timestamp': datetime,
                'reason': str,
                'position': (x, y),
                'camera_id': str,
                'snapshot': image
            }
            channels: List of alert channels to use
        """
        for channel in channels:
            if channel in self.handlers:
                try:
                    self.handlers[channel].send(alert_data)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")
```

#### SMS Handler (Twilio)
```python
class SMSHandler:
    def __init__(self, config):
        self.client = Client(config['account_sid'], config['auth_token'])
        self.from_number = config['from_number']
        self.to_numbers = config['to_numbers']

    def send(self, alert_data):
        message = self._format_message(alert_data)

        for to_number in self.to_numbers:
            self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
```

#### Email Handler
```python
class EmailHandler:
    def __init__(self, config):
        self.smtp_server = config['smtp_server']
        self.smtp_port = config['smtp_port']
        self.username = config['username']
        self.password = config['password']
        self.from_email = config['from_email']
        self.to_emails = config['to_emails']

    def send(self, alert_data):
        msg = MIMEMultipart()
        msg['Subject'] = f"Security Alert: {alert_data['type']}"
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)

        # Add text body
        body = self._format_html(alert_data)
        msg.attach(MIMEText(body, 'html'))

        # Attach snapshot if available
        if 'snapshot' in alert_data:
            img = MIMEImage(alert_data['snapshot'])
            msg.attach(img)

        # Send email
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
```

---

## 🔄 Data Flow

### Real-Time Processing Pipeline

```
Frame Captured (t=0ms)
    ↓
Preprocessing (t=5ms)
    ↓
Person Detection (t=20ms)
    ↓
Feature Extraction (t=10ms)
    ↓
Tracking Update (t=5ms)
    ↓
Re-ID Matching (t=15ms)
    ↓
Coordinate Mapping (t=2ms)
    ↓
Path Update (t=3ms)
    ↓
Rule Evaluation (t=5ms)
    ↓
Alert Generation (if needed) (t=10ms)
    ↓
Database Storage (t=5ms)
    ↓
WebSocket Broadcast (t=2ms)
───────────────────────────
Total: ~82ms per frame
Target: 12-15 FPS per camera
```

---

## ⚡ Performance Optimization

### GPU Acceleration
```python
# TensorRT optimization
import tensorrt as trt

def optimize_yolo_trt(onnx_path, engine_path):
    """Convert YOLO to TensorRT for 2-3x speedup"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # Half precision

    engine = builder.build_engine(network, config)

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

### Multi-threading Strategy
```python
import threading
from queue import Queue

class MultiCameraProcessor:
    def __init__(self, cameras, detector, tracker):
        self.cameras = cameras
        self.detector = detector
        self.tracker = tracker
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue(maxsize=30)

    def start(self):
        # Thread 1: Frame reading
        threading.Thread(target=self._read_frames, daemon=True).start()

        # Thread 2: Detection & tracking
        threading.Thread(target=self._process_frames, daemon=True).start()

        # Thread 3: Re-ID & mapping
        threading.Thread(target=self._reid_and_map, daemon=True).start()
```

### Caching Strategy
```python
from functools import lru_cache
import redis

# Redis for distributed caching
redis_client = redis.Redis(host='localhost', port=6379)

def cache_reid_features(global_id, feature, ttl=3600):
    """Cache Re-ID features for faster matching"""
    key = f"reid:features:{global_id}"
    redis_client.setex(key, ttl, feature.tobytes())

@lru_cache(maxsize=1000)
def get_homography_matrix(camera_id):
    """Cache homography matrices"""
    # Expensive computation cached
    return compute_homography(camera_id)
```

---

## 🔒 Security Considerations

### Privacy Protection
- Hash global IDs before storage
- Blur faces in stored snapshots (optional)
- Automatic data retention policies
- GDPR-compliant data deletion

### Access Control
- Role-based access (Admin, Security, Viewer)
- API key authentication
- Audit logging for all access

### Data Encryption
- TLS for all network communication
- Encrypted database storage
- Secure credential management

---

## 📊 Performance Benchmarks

### Hardware Requirements

| Component | Minimum | Recommended | Enterprise |
|-----------|---------|-------------|------------|
| **GPU** | GTX 1060 6GB | RTX 3060 12GB | RTX 4090 24GB |
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 8GB | 16GB | 32GB |
| **Storage** | 256GB SSD | 512GB NVMe | 1TB+ NVMe |

### Expected Performance

| Configuration | Cameras | FPS/Camera | Latency |
|--------------|---------|------------|---------|
| Entry | 4 | 10 | 150ms |
| Standard | 8 | 15 | 100ms |
| Professional | 16 | 20 | 80ms |
| Enterprise | 32+ | 30 | 50ms |

---

## 🔄 Scalability

### Horizontal Scaling
```
Load Balancer
    ├── Worker 1 (Cameras 1-4)
    ├── Worker 2 (Cameras 5-8)
    ├── Worker 3 (Cameras 9-12)
    └── Worker N (Cameras ...)
         ↓
    Centralized Re-ID Server
         ↓
    Shared Database (PostgreSQL)
         ↓
    Redis Cache
```

### Microservices Architecture
```
API Gateway (FastAPI)
    ├── Detection Service (gRPC)
    ├── Tracking Service (gRPC)
    ├── Re-ID Service (gRPC)
    ├── Mapping Service (gRPC)
    └── Alert Service (Message Queue)
```

---

## 📚 References

- [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
- [OSNet Paper](https://arxiv.org/abs/1905.00953)
- [FastReID Documentation](https://github.com/JDAI-CV/fast-reid)
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

---

**Next**: See [PHASES.md](PHASES.md) for step-by-step implementation guide.
