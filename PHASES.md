# 📋 Implementation Phases

This document provides a detailed, step-by-step guide to implementing the Multi-Camera Person Tracking & Floor-Plan Mapping System. Each phase builds upon the previous one, allowing for incremental development and testing.

---

## 📊 Overview

| Phase | Name | Duration | Complexity | Priority |
|-------|------|----------|------------|----------|
| **Phase 1** | Camera Integration & Stream Processing | 1 week | ⭐⭐ | Critical |
| **Phase 2** | Person Detection (YOLO) | 1 week | ⭐⭐ | Critical |
| **Phase 3** | Single-Camera Tracking (DeepSORT) | 1 week | ⭐⭐⭐ | Critical |
| **Phase 4** | Cross-Camera Re-Identification | 2 weeks | ⭐⭐⭐⭐ | Critical |
| **Phase 5** | Camera Calibration & Spatial Mapping | 1 week | ⭐⭐⭐ | Critical |
| **Phase 6** | Path Tracking Engine | 1 week | ⭐⭐⭐ | Critical |
| **Phase 7** | Rules Engine & Alert System | 1 week | ⭐⭐ | High |
| **Phase 8** | Dashboard & API Development | 2 weeks | ⭐⭐⭐ | High |
| **Phase 9** | Advanced Features (Optional) | 2-4 weeks | ⭐⭐⭐⭐⭐ | Medium |
| **Phase 10** | Testing, Optimization & Deployment | 1-2 weeks | ⭐⭐⭐ | High |

**Total Timeline**: 10-14 weeks

---

## Phase 1: Camera Integration & Stream Processing

### 🎯 Objectives
- Connect multiple camera streams (RTSP, USB, video files)
- Implement multi-threaded frame reading
- Create frame buffering and synchronization
- Handle reconnection and error recovery

### 📦 Deliverables
- `src/core/camera/camera_manager.py`
- `src/core/camera/stream_reader.py`
- `src/utils/video_utils.py`
- Configuration file: `config/cameras.yaml`

### 🛠️ Implementation Steps

#### Step 1.1: Create Camera Configuration Schema
```yaml
# config/cameras.yaml
cameras:
  - id: "cam_001"
    name: "Main Entrance"
    source: "rtsp://admin:password@192.168.1.10:554/stream"
    fps: 15
    resolution: [1920, 1080]
    location: "Building A - Ground Floor"

  - id: "cam_002"
    name: "Lobby"
    source: "rtsp://admin:password@192.168.1.11:554/stream"
    fps: 15
    resolution: [1920, 1080]
    location: "Building A - Ground Floor"

  - id: "cam_003"
    name: "Corridor 1"
    source: "/path/to/video.mp4"  # For testing with video files
    fps: 30
    resolution: [1920, 1080]
    location: "Building A - 1st Floor"
```

#### Step 1.2: Implement Camera Class
```python
# src/core/camera/camera.py
import cv2
import threading
from queue import Queue
import time

class Camera:
    def __init__(self, camera_id, source, fps=15, buffer_size=30):
        self.id = camera_id
        self.source = source
        self.fps = fps
        self.buffer_size = buffer_size

        self.cap = None
        self.frame_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None

    def connect(self):
        """Connect to camera stream"""
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ConnectionError(f"Failed to connect to camera {self.id}")

        # Set properties
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return True

    def start(self):
        """Start reading frames in background thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _read_frames(self):
        """Background thread to continuously read frames"""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._reconnect()
                time.sleep(1)
                continue

            ret, frame = self.cap.read()

            if not ret:
                self._reconnect()
                continue

            # Add to queue (drop oldest if full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass

            self.frame_queue.put((time.time(), frame))

            # Control frame rate
            time.sleep(1.0 / self.fps)

    def _reconnect(self):
        """Attempt to reconnect to stream"""
        if self.cap:
            self.cap.release()

        time.sleep(2)

        try:
            self.connect()
        except Exception as e:
            print(f"Reconnection failed for {self.id}: {e}")

    def read(self):
        """Read latest frame from queue"""
        if self.frame_queue.empty():
            return None

        return self.frame_queue.get()

    def stop(self):
        """Stop the camera stream"""
        self.running = False

        if self.thread:
            self.thread.join(timeout=2)

        if self.cap:
            self.cap.release()
```

#### Step 1.3: Implement Camera Manager
```python
# src/core/camera/camera_manager.py
import yaml
from .camera import Camera

class CameraManager:
    def __init__(self, config_path):
        self.cameras = {}
        self.load_config(config_path)

    def load_config(self, config_path):
        """Load camera configuration from YAML"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for cam_config in config['cameras']:
            camera = Camera(
                camera_id=cam_config['id'],
                source=cam_config['source'],
                fps=cam_config.get('fps', 15)
            )
            self.cameras[cam_config['id']] = camera

    def start_all(self):
        """Start all cameras"""
        for cam_id, camera in self.cameras.items():
            try:
                camera.connect()
                camera.start()
                print(f"Started camera {cam_id}")
            except Exception as e:
                print(f"Failed to start camera {cam_id}: {e}")

    def read_frames(self):
        """Read frames from all cameras"""
        frames = {}
        for cam_id, camera in self.cameras.items():
            frame_data = camera.read()
            if frame_data:
                frames[cam_id] = frame_data
        return frames

    def stop_all(self):
        """Stop all cameras"""
        for camera in self.cameras.values():
            camera.stop()
```

#### Step 1.4: Create Test Script
```python
# scripts/test_cameras.py
import cv2
from src.core.camera.camera_manager import CameraManager

def main():
    # Initialize camera manager
    manager = CameraManager('config/cameras.yaml')
    manager.start_all()

    print("Reading frames... Press 'q' to quit")

    while True:
        frames = manager.read_frames()

        # Display frames
        for cam_id, (timestamp, frame) in frames.items():
            cv2.imshow(f"Camera {cam_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    manager.stop_all()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### ✅ Testing Checklist
- [ ] Connect to RTSP camera successfully
- [ ] Read frames from USB camera
- [ ] Play video file as camera source
- [ ] Handle disconnection and auto-reconnect
- [ ] Multi-threaded reading works without frame drops
- [ ] Frame synchronization across multiple cameras

---

## Phase 2: Person Detection (YOLO)

### 🎯 Objectives
- Integrate YOLOv8 for person detection
- Optimize inference speed (TensorRT optional)
- Filter detections (confidence threshold, NMS)
- Extract person crops for Re-ID

### 📦 Deliverables
- `src/core/detection/detector.py`
- `src/core/detection/yolo_detector.py`
- Pre-trained YOLO model weights

### 🛠️ Implementation Steps

#### Step 2.1: Install Dependencies
```bash
pip install ultralytics
pip install torch torchvision
```

#### Step 2.2: Implement YOLO Detector
```python
# src/core/detection/yolo_detector.py
from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolov8m.pt', conf_threshold=0.5, device='cuda'):
        """
        Args:
            model_path: Path to YOLO weights
            conf_threshold: Confidence threshold for detections
            device: 'cuda' or 'cpu'
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

        # Warm up model
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8),
                          device=self.device, verbose=False)

    def detect(self, frame):
        """
        Detect persons in frame

        Args:
            frame: numpy array (H, W, 3)

        Returns:
            detections: List of dicts with keys:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - crop: person image crop
        """
        # Run inference (class 0 = person in COCO)
        results = self.model.predict(
            frame,
            classes=[0],  # Person class only
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confs = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box.astype(int)

                # Extract person crop
                crop = frame[y1:y2, x1:x2]

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'crop': crop
                })

        return detections

    def detect_batch(self, frames):
        """Batch detection for multiple frames"""
        # TODO: Implement batch processing for better GPU utilization
        return [self.detect(frame) for frame in frames]
```

#### Step 2.3: Visualization Utilities
```python
# src/utils/visualization.py
import cv2

def draw_detections(frame, detections):
    """Draw bounding boxes on frame"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw confidence
        label = f"{conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
```

#### Step 2.4: Test Detection
```python
# scripts/test_detection.py
import cv2
from src.core.camera.camera_manager import CameraManager
from src.core.detection.yolo_detector import YOLODetector
from src.utils.visualization import draw_detections

def main():
    # Initialize
    camera_manager = CameraManager('config/cameras.yaml')
    detector = YOLODetector(model_path='yolov8m.pt', device='cuda')

    camera_manager.start_all()

    while True:
        frames = camera_manager.read_frames()

        for cam_id, (timestamp, frame) in frames.items():
            # Detect persons
            detections = detector.detect(frame)

            # Draw and display
            frame_vis = draw_detections(frame.copy(), detections)
            cv2.imshow(f"Camera {cam_id} - Detections", frame_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_manager.stop_all()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### ✅ Testing Checklist
- [ ] YOLO model loads successfully
- [ ] Person detection works in real-time (>15 FPS)
- [ ] Confidence threshold filters low-quality detections
- [ ] Bounding boxes are accurate
- [ ] Person crops extracted correctly

---

## Phase 3: Single-Camera Tracking (DeepSORT)

### 🎯 Objectives
- Implement DeepSORT tracker for each camera
- Assign unique track IDs per camera
- Handle occlusions and re-entry
- Extract appearance features for Re-ID

### 📦 Deliverables
- `src/core/tracking/deepsort_tracker.py`
- `src/core/tracking/kalman_filter.py`
- `src/core/tracking/track.py`

### 🛠️ Implementation Steps

#### Step 3.1: Install DeepSORT
```bash
pip install deep-sort-realtime
# OR implement from scratch (recommended for learning)
```

#### Step 3.2: Implement Tracker
```python
# src/core/tracking/deepsort_tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort

class CameraTracker:
    def __init__(self, camera_id, max_age=30, n_init=3):
        """
        Args:
            camera_id: Unique camera identifier
            max_age: Max frames to keep track alive without detection
            n_init: Frames required to confirm track
        """
        self.camera_id = camera_id
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=0.7,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder="mobilenet",  # Feature extractor for Re-ID
            embedder_gpu=True
        )

    def update(self, detections):
        """
        Update tracker with new detections

        Args:
            detections: List of dicts from YOLODetector

        Returns:
            tracks: List of dicts with keys:
                - track_id: int
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - feature: embedding vector
        """
        # Convert to DeepSORT format
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            dets.append(([x1, y1, x2-x1, y2-y1], conf, None))  # [ltwh, conf, class]

        # Update tracker
        tracks = self.tracker.update_tracks(dets, frame=None)

        # Convert to our format
        result_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            ltwh = track.to_ltwh()
            x1, y1, w, h = ltwh
            x2, y2 = x1 + w, y1 + h

            result_tracks.append({
                'camera_id': self.camera_id,
                'track_id': track.track_id,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': track.det_conf,
                'feature': track.get_feature()  # Re-ID embedding
            })

        return result_tracks
```

#### Step 3.3: Multi-Camera Tracker Manager
```python
# src/core/tracking/tracker_manager.py
from .deepsort_tracker import CameraTracker

class TrackerManager:
    def __init__(self, camera_ids):
        self.trackers = {}

        for cam_id in camera_ids:
            self.trackers[cam_id] = CameraTracker(cam_id)

    def update(self, camera_id, detections):
        """Update tracker for specific camera"""
        if camera_id not in self.trackers:
            raise ValueError(f"Unknown camera: {camera_id}")

        return self.trackers[camera_id].update(detections)

    def update_all(self, camera_detections):
        """
        Update all camera trackers

        Args:
            camera_detections: Dict {cam_id: detections_list}

        Returns:
            camera_tracks: Dict {cam_id: tracks_list}
        """
        camera_tracks = {}

        for cam_id, detections in camera_detections.items():
            camera_tracks[cam_id] = self.update(cam_id, detections)

        return camera_tracks
```

#### Step 3.4: Test Tracking
```python
# scripts/test_tracking.py
import cv2
from src.core.camera.camera_manager import CameraManager
from src.core.detection.yolo_detector import YOLODetector
from src.core.tracking.tracker_manager import TrackerManager

def draw_tracks(frame, tracks):
    """Draw tracked persons with IDs"""
    for track in tracks:
        x1, y1, x2, y2 = track['bbox']
        track_id = track['track_id']

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw ID
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame

def main():
    # Initialize
    camera_manager = CameraManager('config/cameras.yaml')
    detector = YOLODetector()
    tracker_manager = TrackerManager(list(camera_manager.cameras.keys()))

    camera_manager.start_all()

    while True:
        frames = camera_manager.read_frames()

        # Detect in all frames
        camera_detections = {}
        for cam_id, (timestamp, frame) in frames.items():
            camera_detections[cam_id] = detector.detect(frame)

        # Track in all cameras
        camera_tracks = tracker_manager.update_all(camera_detections)

        # Visualize
        for cam_id, (timestamp, frame) in frames.items():
            tracks = camera_tracks.get(cam_id, [])
            frame_vis = draw_tracks(frame.copy(), tracks)
            cv2.imshow(f"Camera {cam_id} - Tracking", frame_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_manager.stop_all()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### ✅ Testing Checklist
- [ ] Track IDs assigned correctly per camera
- [ ] Tracks persist across frames
- [ ] Re-identification works when person reappears
- [ ] Occlusion handling works
- [ ] Multiple persons tracked simultaneously

---

## Phase 4: Cross-Camera Re-Identification

### 🎯 Objectives
- Match same person across different cameras
- Create global person IDs
- Handle appearance variations
- Maintain feature gallery for matching

### 📦 Deliverables
- `src/core/reid/reid_model.py`
- `src/core/reid/global_id_manager.py`
- Pre-trained Re-ID model weights

### 🛠️ Implementation Steps

#### Step 4.1: Download Pre-trained Re-ID Model
```bash
# Using torchreid
pip install torchreid

# Download OSNet model (automatic on first use)
```

#### Step 4.2: Implement Re-ID Model
```python
# src/core/reid/reid_model.py
import torch
import torchvision.transforms as T
import torchreid

class ReIDModel:
    def __init__(self, model_name='osnet_x1_0', device='cuda'):
        self.device = device
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        )
        self.model.eval()
        self.model.to(device)

        # Preprocessing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

    def extract_feature(self, image_crop):
        """
        Extract Re-ID feature from person crop

        Args:
            image_crop: numpy array (H, W, 3)

        Returns:
            feature: numpy array (512,)
        """
        # Preprocess
        img_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)

        # Extract feature
        with torch.no_grad():
            feature = self.model(img_tensor)

        # L2 normalize
        feature = torch.nn.functional.normalize(feature, p=2, dim=1)

        return feature.cpu().numpy().flatten()

    def extract_features_batch(self, image_crops):
        """Batch feature extraction"""
        if len(image_crops) == 0:
            return []

        # Preprocess batch
        img_tensors = torch.stack([
            self.transform(crop) for crop in image_crops
        ]).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(img_tensors)

        # L2 normalize
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features.cpu().numpy()
```

#### Step 4.3: Implement Global ID Manager
```python
# src/core/reid/global_id_manager.py
import numpy as np
from scipy.spatial.distance import cosine

class GlobalIDManager:
    def __init__(self, similarity_threshold=0.6, gallery_size=50):
        """
        Args:
            similarity_threshold: Min similarity to match (0-1)
            gallery_size: Max features to keep per global ID
        """
        self.threshold = similarity_threshold
        self.gallery_size = gallery_size

        self.global_ids = {}  # global_id -> list of features
        self.camera_to_global = {}  # (cam_id, track_id) -> global_id
        self.next_global_id = 1

        # Metadata
        self.global_id_metadata = {}  # global_id -> {first_seen, last_seen, cameras}

    def match_or_create(self, camera_id, track_id, feature):
        """
        Match feature to existing global ID or create new

        Returns:
            global_id: int
        """
        # Check if this (cam, track) already has global ID
        key = (camera_id, track_id)
        if key in self.camera_to_global:
            global_id = self.camera_to_global[key]
            # Update feature gallery
            self._update_gallery(global_id, feature)
            return global_id

        # Search for match across all global IDs
        best_match_id = None
        best_similarity = -1

        for gid, gallery in self.global_ids.items():
            similarity = self._compute_similarity(feature, gallery)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = gid

        # Create new global ID if no match
        if best_similarity < self.threshold:
            global_id = self.next_global_id
            self.next_global_id += 1
            self.global_ids[global_id] = [feature]
            self.global_id_metadata[global_id] = {
                'first_seen': None,
                'last_seen': None,
                'cameras': {camera_id}
            }
        else:
            global_id = best_match_id
            self._update_gallery(global_id, feature)
            self.global_id_metadata[global_id]['cameras'].add(camera_id)

        # Map (cam, track) to global ID
        self.camera_to_global[key] = global_id

        return global_id

    def _compute_similarity(self, feature, gallery):
        """Compute max cosine similarity with gallery"""
        if len(gallery) == 0:
            return 0.0

        gallery_array = np.array(gallery)
        similarities = np.dot(gallery_array, feature)
        return np.max(similarities)

    def _update_gallery(self, global_id, feature):
        """Add feature to gallery (keep recent N)"""
        self.global_ids[global_id].append(feature)

        # Keep only recent features
        if len(self.global_ids[global_id]) > self.gallery_size:
            self.global_ids[global_id] = self.global_ids[global_id][-self.gallery_size:]

    def get_global_id(self, camera_id, track_id):
        """Get global ID for (camera, track) pair"""
        return self.camera_to_global.get((camera_id, track_id), None)
```

#### Step 4.4: Integration Test
```python
# scripts/test_reid.py
import cv2
from src.core.camera.camera_manager import CameraManager
from src.core.detection.yolo_detector import YOLODetector
from src.core.tracking.tracker_manager import TrackerManager
from src.core.reid.reid_model import ReIDModel
from src.core.reid.global_id_manager import GlobalIDManager

def main():
    # Initialize all components
    camera_manager = CameraManager('config/cameras.yaml')
    detector = YOLODetector()
    tracker_manager = TrackerManager(list(camera_manager.cameras.keys()))
    reid_model = ReIDModel()
    global_id_manager = GlobalIDManager()

    camera_manager.start_all()

    # Color map for global IDs
    colors = {}

    while True:
        frames = camera_manager.read_frames()

        # Detect
        camera_detections = {}
        for cam_id, (timestamp, frame) in frames.items():
            camera_detections[cam_id] = detector.detect(frame)

        # Track
        camera_tracks = tracker_manager.update_all(camera_detections)

        # Re-ID: Assign global IDs
        for cam_id, tracks in camera_tracks.items():
            for track in tracks:
                # Get person crop
                x1, y1, x2, y2 = track['bbox']
                timestamp, frame = frames[cam_id]
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Extract Re-ID feature
                feature = reid_model.extract_feature(crop)

                # Assign global ID
                global_id = global_id_manager.match_or_create(
                    cam_id,
                    track['track_id'],
                    feature
                )

                track['global_id'] = global_id

                # Assign color
                if global_id not in colors:
                    colors[global_id] = tuple(np.random.randint(0, 255, 3).tolist())

        # Visualize with global IDs
        for cam_id, (timestamp, frame) in frames.items():
            tracks = camera_tracks.get(cam_id, [])

            for track in tracks:
                if 'global_id' not in track:
                    continue

                x1, y1, x2, y2 = track['bbox']
                global_id = track['global_id']
                color = colors[global_id]

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Draw global ID
                label = f"Global ID: {global_id}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow(f"Camera {cam_id} - Global Re-ID", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_manager.stop_all()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### ✅ Testing Checklist
- [ ] Same person gets same global ID across cameras
- [ ] Different persons get different global IDs
- [ ] Re-ID works with appearance variations
- [ ] Gallery updating improves matching over time
- [ ] Performance is acceptable (< 50ms per person)

---

## Phase 5: Camera Calibration & Spatial Mapping

### 🎯 Objectives
- Calibrate each camera to floor plan
- Compute homography matrices
- Convert pixel coordinates to world coordinates
- Define zones on floor plan

### 📦 Deliverables
- `src/core/mapping/calibration.py`
- `src/core/mapping/floor_plan.py`
- `scripts/camera_calibration.py` (interactive tool)
- `scripts/zone_drawer.py` (interactive tool)

### 🛠️ Implementation Steps

#### Step 5.1: Create Calibration Tool
```python
# scripts/camera_calibration.py
import cv2
import numpy as np
import yaml

class CalibrationTool:
    def __init__(self, camera_frame, floor_plan_image):
        self.camera_frame = camera_frame
        self.floor_plan = floor_plan_image

        self.camera_points = []
        self.floor_points = []

        self.selecting_camera = True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selecting_camera:
                self.camera_points.append([x, y])
                print(f"Camera point {len(self.camera_points)}: ({x}, {y})")
            else:
                self.floor_points.append([x, y])
                print(f"Floor point {len(self.floor_points)}: ({x}, {y})")

    def run(self):
        print("Select 4+ corresponding points")
        print("First: Select points on CAMERA view")

        cv2.namedWindow('Camera View')
        cv2.setMouseCallback('Camera View', self.mouse_callback)

        while True:
            # Draw points on camera view
            img = self.camera_frame.copy()
            for i, pt in enumerate(self.camera_points):
                cv2.circle(img, tuple(pt), 5, (0, 255, 0), -1)
                cv2.putText(img, str(i+1), tuple(pt),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Camera View', img)

            # Draw points on floor plan
            floor_img = self.floor_plan.copy()
            for i, pt in enumerate(self.floor_points):
                cv2.circle(floor_img, tuple(pt), 5, (0, 0, 255), -1)
                cv2.putText(floor_img, str(i+1), tuple(pt),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('Floor Plan', floor_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # Switch between camera and floor
                if len(self.camera_points) >= 4:
                    self.selecting_camera = False
                    print("Now: Select corresponding points on FLOOR PLAN")
                else:
                    print("Need at least 4 points on camera view first")

            elif key == ord('c'):  # Compute homography
                if len(self.camera_points) >= 4 and \
                   len(self.camera_points) == len(self.floor_points):
                    H, _ = cv2.findHomography(
                        np.float32(self.camera_points),
                        np.float32(self.floor_points),
                        cv2.RANSAC
                    )
                    print("Homography matrix:")
                    print(H)
                    return H
                else:
                    print("Need equal points on both views (min 4)")

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        return None

def main():
    import sys

    if len(sys.argv) != 4:
        print("Usage: python camera_calibration.py <camera_id> <video_frame> <floor_plan>")
        sys.exit(1)

    camera_id = sys.argv[1]

    # Load images
    camera_frame = cv2.imread(sys.argv[2])
    floor_plan = cv2.imread(sys.argv[3])

    # Run calibration
    tool = CalibrationTool(camera_frame, floor_plan)
    H = tool.run()

    if H is not None:
        # Save calibration
        calibration = {
            'camera_id': camera_id,
            'homography_matrix': H.tolist()
        }

        with open(f'config/calibration_{camera_id}.yaml', 'w') as f:
            yaml.dump(calibration, f)

        print(f"Saved calibration to config/calibration_{camera_id}.yaml")

if __name__ == '__main__':
    main()
```

#### Step 5.2: Implement Coordinate Mapping
```python
# src/core/mapping/coordinate_mapper.py
import cv2
import numpy as np
import yaml

class CoordinateMapper:
    def __init__(self):
        self.calibrations = {}  # camera_id -> homography matrix

    def load_calibration(self, camera_id, calibration_path):
        """Load calibration for camera"""
        with open(calibration_path, 'r') as f:
            calib = yaml.safe_load(f)

        H = np.array(calib['homography_matrix'])
        self.calibrations[camera_id] = H

    def pixel_to_world(self, camera_id, x, y):
        """Convert camera pixel to floor plan coordinates"""
        if camera_id not in self.calibrations:
            raise ValueError(f"Camera {camera_id} not calibrated")

        H = self.calibrations[camera_id]

        # Apply homography
        point = np.array([[[x, y]]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point, H)

        return world_point[0, 0]

    def bbox_center_to_world(self, camera_id, bbox):
        """Convert bbox center to world coordinates"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = y2  # Use bottom center for ground position

        return self.pixel_to_world(camera_id, center_x, center_y)
```

#### Step 5.3: Implement Floor Plan Manager
```python
# src/core/mapping/floor_plan.py
import cv2
import numpy as np

class FloorPlan:
    def __init__(self, image_path, scale=1.0):
        """
        Args:
            image_path: Path to floor plan image
            scale: Meters per pixel (optional)
        """
        self.image = cv2.imread(image_path)
        self.scale = scale
        self.zones = []

    def add_zone(self, name, polygon, zone_type, color=None):
        """
        Add a zone to the floor plan

        Args:
            name: Zone name
            polygon: List of (x, y) points defining polygon
            zone_type: 'required', 'restricted', 'safe', 'custom'
            color: BGR tuple (optional)
        """
        if color is None:
            # Default colors by type
            colors = {
                'required': (0, 255, 0),    # Green
                'restricted': (0, 0, 255),  # Red
                'safe': (255, 255, 0),      # Cyan
                'custom': (128, 128, 128)   # Gray
            }
            color = colors.get(zone_type, (128, 128, 128))

        self.zones.append({
            'name': name,
            'polygon': np.array(polygon, dtype=np.int32),
            'type': zone_type,
            'color': color
        })

    def check_point_in_zone(self, x, y):
        """Check which zones contain the point"""
        zones_containing = []

        for zone in self.zones:
            dist = cv2.pointPolygonTest(
                zone['polygon'],
                (float(x), float(y)),
                False
            )

            if dist >= 0:  # Point inside polygon
                zones_containing.append(zone)

        return zones_containing

    def draw_zones(self, image=None):
        """Draw all zones on floor plan"""
        if image is None:
            image = self.image.copy()

        # Draw polygons
        for zone in self.zones:
            # Fill polygon with transparency
            overlay = image.copy()
            cv2.fillPoly(overlay, [zone['polygon']], zone['color'])
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

            # Draw outline
            cv2.polylines(image, [zone['polygon']], True, zone['color'], 2)

            # Draw label
            centroid = zone['polygon'].mean(axis=0).astype(int)
            cv2.putText(image, zone['name'], tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone['color'], 2)

        return image

    def draw_position(self, image, x, y, label='', color=(0, 255, 0)):
        """Draw a position marker on floor plan"""
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

        if label:
            cv2.putText(image, label, (int(x) + 10, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return image

    def draw_path(self, image, positions, color=(255, 0, 255)):
        """Draw a path on floor plan"""
        if len(positions) < 2:
            return image

        points = np.array(positions, dtype=np.int32)
        cv2.polylines(image, [points], False, color, 2)

        return image
```

### ✅ Testing Checklist
- [ ] Calibration tool works interactively
- [ ] Homography matrices computed correctly
- [ ] Pixel to world coordinate conversion accurate
- [ ] Zones defined correctly on floor plan
- [ ] Point-in-zone detection works

---

## Phase 6: Path Tracking Engine

### 🎯 Objectives
- Aggregate positions for each global ID
- Store path history in database
- Track zone visits
- Calculate movement statistics

### 📦 Deliverables
- `src/core/path/path_tracker.py`
- `src/core/path/database.py`
- Database schema

### 🛠️ Implementation Steps

#### Step 6.1: Implement Path Tracker
```python
# src/core/path/path_tracker.py
import time
import numpy as np

class PersonPath:
    def __init__(self, global_id):
        self.global_id = global_id
        self.positions = []  # [(timestamp, x, y, camera_id)]
        self.zones_visited = set()
        self.first_seen = None
        self.last_seen = None
        self.total_distance = 0.0
        self.status = 'active'  # 'active', 'completed', 'suspicious'

    def add_position(self, x, y, camera_id, timestamp=None):
        """Add new position to path"""
        if timestamp is None:
            timestamp = time.time()

        if self.first_seen is None:
            self.first_seen = timestamp

        self.last_seen = timestamp

        # Calculate distance from last position
        if len(self.positions) > 0:
            last_pos = self.positions[-1]
            dist = np.sqrt((x - last_pos[1])**2 + (y - last_pos[2])**2)
            self.total_distance += dist

        self.positions.append((timestamp, x, y, camera_id))

    def get_current_position(self):
        """Get most recent position"""
        if len(self.positions) == 0:
            return None
        return self.positions[-1]

    def get_speed(self, window=5):
        """Calculate recent speed (meters/second)"""
        if len(self.positions) < 2:
            return 0.0

        recent = self.positions[-min(window, len(self.positions)):]

        if len(recent) < 2:
            return 0.0

        time_diff = recent[-1][0] - recent[0][0]
        if time_diff == 0:
            return 0.0

        dist = sum([
            np.sqrt((recent[i][1] - recent[i-1][1])**2 +
                   (recent[i][2] - recent[i-1][2])**2)
            for i in range(1, len(recent))
        ])

        return dist / time_diff

    def add_zone_visit(self, zone_name):
        """Record zone visit"""
        self.zones_visited.add(zone_name)

class PathTracker:
    def __init__(self):
        self.active_paths = {}  # global_id -> PersonPath
        self.completed_paths = []
        self.timeout = 300  # 5 minutes

    def update(self, global_id, world_x, world_y, camera_id):
        """Update path for global ID"""
        if global_id not in self.active_paths:
            self.active_paths[global_id] = PersonPath(global_id)

        path = self.active_paths[global_id]
        path.add_position(world_x, world_y, camera_id)

        return path

    def get_path(self, global_id):
        """Get path for global ID"""
        return self.active_paths.get(global_id, None)

    def cleanup_inactive(self):
        """Move inactive paths to completed"""
        current_time = time.time()
        inactive = []

        for gid, path in self.active_paths.items():
            if current_time - path.last_seen > self.timeout:
                inactive.append(gid)

        for gid in inactive:
            path = self.active_paths.pop(gid)
            path.status = 'completed'
            self.completed_paths.append(path)
```

#### Step 6.2: Integrate with Previous Components
```python
# src/main.py (integration example)
import cv2
import time
from src.core.camera.camera_manager import CameraManager
from src.core.detection.yolo_detector import YOLODetector
from src.core.tracking.tracker_manager import TrackerManager
from src.core.reid.reid_model import ReIDModel
from src.core.reid.global_id_manager import GlobalIDManager
from src.core.mapping.coordinate_mapper import CoordinateMapper
from src.core.mapping.floor_plan import FloorPlan
from src.core.path.path_tracker import PathTracker

def main():
    # Initialize all components
    camera_manager = CameraManager('config/cameras.yaml')
    detector = YOLODetector()
    tracker_manager = TrackerManager(list(camera_manager.cameras.keys()))
    reid_model = ReIDModel()
    global_id_manager = GlobalIDManager()

    # Load calibration and floor plan
    coord_mapper = CoordinateMapper()
    for cam_id in camera_manager.cameras.keys():
        coord_mapper.load_calibration(
            cam_id,
            f'config/calibration_{cam_id}.yaml'
        )

    floor_plan = FloorPlan('data/floor_plans/building_a.png')
    # Load zones from config...

    path_tracker = PathTracker()

    camera_manager.start_all()

    while True:
        # Read frames
        frames = camera_manager.read_frames()

        # Detect persons
        camera_detections = {}
        for cam_id, (timestamp, frame) in frames.items():
            camera_detections[cam_id] = detector.detect(frame)

        # Track persons
        camera_tracks = tracker_manager.update_all(camera_detections)

        # Re-ID and map to floor plan
        floor_plan_img = floor_plan.draw_zones()

        for cam_id, tracks in camera_tracks.items():
            timestamp, frame = frames[cam_id]

            for track in tracks:
                # Extract Re-ID feature
                x1, y1, x2, y2 = track['bbox']
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                feature = reid_model.extract_feature(crop)

                # Assign global ID
                global_id = global_id_manager.match_or_create(
                    cam_id, track['track_id'], feature
                )

                # Convert to world coordinates
                world_x, world_y = coord_mapper.bbox_center_to_world(
                    cam_id, track['bbox']
                )

                # Update path
                path = path_tracker.update(global_id, world_x, world_y, cam_id)

                # Check zones
                zones = floor_plan.check_point_in_zone(world_x, world_y)
                for zone in zones:
                    path.add_zone_visit(zone['name'])

                # Draw on floor plan
                color = (0, 255, 0)  # Or per global ID
                floor_plan.draw_position(
                    floor_plan_img,
                    world_x, world_y,
                    f"ID:{global_id}",
                    color
                )

                # Draw path
                if len(path.positions) > 1:
                    positions = [(int(p[1]), int(p[2])) for p in path.positions]
                    floor_plan.draw_path(floor_plan_img, positions, color)

        # Display
        cv2.imshow('Floor Plan - Live Tracking', floor_plan_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Cleanup inactive paths periodically
        path_tracker.cleanup_inactive()

    camera_manager.stop_all()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### ✅ Testing Checklist
- [ ] Paths aggregate correctly for each global ID
- [ ] Position history stored
- [ ] Zone visits tracked
- [ ] Movement statistics calculated
- [ ] Inactive paths cleaned up

---

## Phase 7: Rules Engine & Alert System

*(Continuing in next message due to length...)*

### 🛠️ Quick Implementation

See detailed implementation in ARCHITECTURE.md. Key files:

- `src/core/rules/rule_engine.py`
- `src/alerts/alert_manager.py`
- `src/alerts/sms_handler.py`
- `src/alerts/email_handler.py`

### ✅ Testing Checklist
- [ ] Required zone rules work
- [ ] Restricted zone alerts trigger
- [ ] Loitering detection works
- [ ] Alerts sent via configured channels

---

## Phase 8: Dashboard & API Development

### 🎯 Objectives
- Build REST API with FastAPI
- Create real-time WebSocket updates
- Develop web dashboard
- Live camera feeds and floor plan visualization

### 📦 Deliverables
- `src/api/main.py`
- `src/dashboard/app.py`
- Frontend (React/Vue.js)

### ✅ Testing Checklist
- [ ] API endpoints functional
- [ ] WebSocket real-time updates work
- [ ] Dashboard displays live tracking
- [ ] Historical playback works

---

## Phase 9: Advanced Features (Optional)

- Face recognition integration
- 3D building visualization
- Path prediction with LSTM
- Anomaly detection
- Heat map generation

---

## Phase 10: Testing, Optimization & Deployment

### 🎯 Activities
- Unit testing all modules
- Integration testing
- Performance optimization (TensorRT, batching)
- Docker containerization
- Deployment documentation

---

## 📈 Progress Tracking

Use this checklist to track overall progress:

- [ ] Phase 1: Camera Integration ✓
- [ ] Phase 2: Person Detection ✓
- [ ] Phase 3: Single-Camera Tracking ✓
- [ ] Phase 4: Cross-Camera Re-ID ✓
- [ ] Phase 5: Spatial Mapping ✓
- [ ] Phase 6: Path Tracking ✓
- [ ] Phase 7: Rules & Alerts ✓
- [ ] Phase 8: Dashboard & API ✓
- [ ] Phase 9: Advanced Features (Optional)
- [ ] Phase 10: Deployment ✓

---

**Ready to start?** Begin with Phase 1 and work sequentially!
