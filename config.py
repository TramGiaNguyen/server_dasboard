# Shared configuration for Smart Parking System
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, 'static', 'models', 'yolov8l.pt')

# Gate camera model — separate, smaller model for faster line-crossing detection.
# Falls back to bare filename so Ultralytics auto-downloads on first run if the
# file is not already present in static/models/.
_gate_model_file = os.getenv('GATE_MODEL_FILE', 'yolov8s.pt')
_gate_model_local = os.path.join(BASE_DIR, 'static', 'models', _gate_model_file)
GATE_MODEL_PATH = _gate_model_local if os.path.exists(_gate_model_local) else _gate_model_file

# Video paths — use .env for RTSP URLs in production
# Local video files (offline demo/testing)
PARKING_VIDEO_URL = os.getenv('PARKING_RTSP_URL', '/static/video/CAM_PARKING.mp4')
GATE_VIDEO_URL = os.getenv('GATE_RTSP_URL', '/static/video/CAM_GATE.mp4')

# Sound paths
WARNING_SOUND_PATH = os.path.join(BASE_DIR, 'static', 'sound', 'warning-sound.mp3')

# Server configuration
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', '5000'))

# ============ Detection Configuration ============
MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', '0.35'))
MIN_AREA_SIZE = int(os.getenv('MIN_AREA_SIZE', '1500'))
MIN_DIMENSION = int(os.getenv('MIN_DIMENSION', '30'))

# COCO class list path
COCO_FILE_PATH = os.path.join(BASE_DIR, 'coco.txt')

# ============ Gate Camera Configuration ============
GATE_LINE_1_Y = int(os.getenv('GATE_LINE_1_Y', '80'))
GATE_LINE_2_Y = int(os.getenv('GATE_LINE_2_Y', '360'))
GATE_LINE_3_Y = int(os.getenv('GATE_LINE_3_Y', '220'))
GATE_LINE_THICKNESS = 2

# Gate inference + tracking knobs (Phase A/B/C overhaul).
GATE_USE_HALF_PRECISION = os.getenv('GATE_USE_HALF_PRECISION', 'True').lower() == 'true'
GATE_DETECT_CONF = float(os.getenv('GATE_DETECT_CONF', '0.25'))
GATE_DETECT_IOU = float(os.getenv('GATE_DETECT_IOU', '0.45'))
GATE_DETECT_IMGSZ = int(os.getenv('GATE_DETECT_IMGSZ', '640'))
_gate_tracker_cfg = os.getenv('GATE_TRACKER_CONFIG', 'cfg/trackers/bytetrack_gate.yaml')
GATE_TRACKER_CONFIG = (
    _gate_tracker_cfg
    if os.path.isabs(_gate_tracker_cfg)
    else os.path.join(BASE_DIR, _gate_tracker_cfg)
)
GATE_RTSP_BUFFER = int(os.getenv('GATE_RTSP_BUFFER', '1'))
GATE_OCR_WORKERS = int(os.getenv('GATE_OCR_WORKERS', '2'))
GATE_DEBUG_NDJSON_LOG = os.getenv('GATE_DEBUG_NDJSON_LOG', 'False').lower() == 'true'
GATE_OCR_DEBOUNCE_CONF = float(os.getenv('GATE_OCR_DEBOUNCE_CONF', '0.85'))
GATE_OCR_DEBOUNCE_FRAMES = int(os.getenv('GATE_OCR_DEBOUNCE_FRAMES', '10'))
GATE_LAG_DRAIN_MULTIPLIER = float(os.getenv('GATE_LAG_DRAIN_MULTIPLIER', '2.0'))
GATE_TARGET_FPS = float(os.getenv('GATE_TARGET_FPS', '30'))

# Ensure LINE_3 is between LINE_1 and LINE_2.
# Some parts of the gate pipeline assume this ordering for stable direction/zone logic.
_gate_y_low = min(GATE_LINE_1_Y, GATE_LINE_2_Y)
_gate_y_high = max(GATE_LINE_1_Y, GATE_LINE_2_Y)
if not (_gate_y_low < GATE_LINE_3_Y < _gate_y_high):
    # Put it in the middle to keep relative geometry consistent.
    GATE_LINE_3_Y = (_gate_y_low + _gate_y_high) // 2

# ============ Parking Camera Configuration ============
PARKING_DISPLAY_WIDTH = int(os.getenv('PARKING_DISPLAY_WIDTH', '1020'))
PARKING_DISPLAY_HEIGHT = int(os.getenv('PARKING_DISPLAY_HEIGHT', '500'))
PARKING_STARTUP_FRAMES = int(os.getenv('PARKING_STARTUP_FRAMES', '30'))
PARKING_SLOT_EMPTY_THRESHOLD = int(os.getenv('PARKING_SLOT_EMPTY_THRESHOLD', '45'))

# Upscale display - phóng to substream lên 1080p cho hiển thị đẹp hơn
PARKING_UPSCALE_DISPLAY = os.getenv('PARKING_UPSCALE_DISPLAY', 'True').lower() == 'true'
PARKING_UPSCALE_WIDTH = int(os.getenv('PARKING_UPSCALE_WIDTH', '1920'))
PARKING_UPSCALE_HEIGHT = int(os.getenv('PARKING_UPSCALE_HEIGHT', '1080'))

# ============ Performance Optimization ============
# Model half precision (FP16) - 20-40% faster on GPU with minimal accuracy loss
PARKING_USE_HALF_PRECISION = os.getenv('PARKING_USE_HALF_PRECISION', 'True').lower() == 'true'

# CLAHE (Contrast enhancement) - improves detection in low light but adds ~5-10% overhead
# Set to False to disable for better performance in good lighting conditions
PARKING_USE_CLAHE = os.getenv('PARKING_USE_CLAHE', 'False').lower() == 'true'

# Detection thread - run YOLO inference in separate thread (15-30% faster)
PARKING_USE_DETECTION_THREAD = os.getenv('PARKING_USE_DETECTION_THREAD', 'True').lower() == 'true'

# RTSP reconnection - auto-reconnect on stream failure
PARKING_USE_ROBUST_RTSP = os.getenv('PARKING_USE_ROBUST_RTSP', 'True').lower() == 'true'

# Frame skip monitoring - log frame skip rate and FPS
PARKING_ENABLE_SKIP_MONITORING = os.getenv('PARKING_ENABLE_SKIP_MONITORING', 'True').lower() == 'true'

# ============ Vehicle Tracking Configuration ============
TRACKING_MAX_WAIT_TIME = 60
TRACKING_CLEANUP_INTERVAL = 30
TRACKING_MATCH_THRESHOLD = 70
TRACKING_ENABLED = True

# ByteTrack config
PARKING_TRACKER_CONFIG = os.path.join(BASE_DIR, 'cfg', 'trackers', 'bytetrack_parking.yaml')

# ============ Database Configuration ============
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:1412@localhost:5432/PARKING_PLATE'
)

# ============ Logging Configuration ============
import logging

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FORMAT = '%(asctime)s [%(levelname)7s] %(name)s: %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)

# Module-specific loggers
logger_gate = logging.getLogger('gate')
logger_parking = logging.getLogger('parking')
logger_tracking = logging.getLogger('tracking')
logger_sync = logging.getLogger('sync')
logger_ocr = logging.getLogger('ocr')
