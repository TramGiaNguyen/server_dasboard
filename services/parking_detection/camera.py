"""
camera.py — Parking Detection Camera Entry Point.

Architecture mirrors gate_camera/:
  - detect_track_worker  : daemon thread — YOLO+ByteTrack + slot matching → render_queue
  - render_worker         : daemon thread — consumes render_queue → shared_state.parking_latest_jpeg
  - process_video_stream  : generator — reads shared_state for MJPEG relay to Flask

Both camera workers run as independent daemon threads.
Flask reads from shared_state directly (same pattern as gate camera).
"""
import cv2
import time
import os
import sys
import threading
import numpy as np
from queue import Queue

# ── Path setup ────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
from config import (
    COCO_FILE_PATH,
    TRACKING_ENABLED,
    TRACKING_MATCH_THRESHOLD,
    PARKING_TRACKER_CONFIG,
    MIN_CONFIDENCE,
    MIN_AREA_SIZE,
    MIN_DIMENSION,
    PARKING_SLOT_EMPTY_THRESHOLD,
    PARKING_STARTUP_FRAMES,
    PARKING_USE_CLAHE,
    PARKING_USE_DETECTION_THREAD,
    PARKING_UPSCALE_DISPLAY,
    PARKING_UPSCALE_WIDTH,
    PARKING_UPSCALE_HEIGHT,
    PARKING_VIDEO_URL,
)

# ── Imports from workers.py ────────────────────────────────────────────────────
from services.parking_detection.workers import (
    detect_track_worker,
    render_worker,
    SharedRefs,
)
from services.parking_detection.pipeline import ParkingPipelineRuntime
from services.parking_detection.detection import (
    load_class_list,
    define_parking_areas,
    define_entry_line_zones,
    create_inner_zones,
)

# ── RTSP capture ──────────────────────────────────────────────────────────────
try:
    from shared.rtsp_capture import RTSPCapture
except ImportError:
    RTSPCapture = None

# Import shared state (needed for MJPEG relay and parking_latest_jpeg)
import shared.state as shared_state

# ── Constants ─────────────────────────────────────────────────────────────────
RTSP_READ_TIMEOUT = 0.10
PARKING_CAPTURE_DIR = os.path.join(
    _BASE_DIR, 'static', 'parking_captures'
)
os.makedirs(PARKING_CAPTURE_DIR, exist_ok=True)

# ==============================================================================
# process_video_stream — thin relay (mirrors gate_camera/pipeline.py pattern)
# ==============================================================================

def process_video_stream(model, video_url, socketio=None, current_parking_status=None):
    """
    Thin generator relay:
      - Initializes config
      - Starts detect_track + render daemon threads
      - Yields MJPEG chunks
    """
    # ── Resolve video path ─────────────────────────────────────────────────
    if video_url.startswith('/static/'):
        video_path = os.path.join(_BASE_DIR, video_url.lstrip('/'))
    else:
        video_path = video_url

    is_stream = video_path.lower().startswith(('rtsp://', 'http://', 'https://'))

    if not is_stream and not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        print(f"[INFO] Current working directory: {os.getcwd()}")
        print(f"[INFO] Base directory: {_BASE_DIR}")
        return

    print(f"[INFO] Loading video from: {video_path}")

    # ── Startup sync (RTSP only) ─────────────────────────────────────────────
    if is_stream and RTSPCapture:
        rtsp_cap = RTSPCapture(video_path, buffer_size=2)
        if not rtsp_cap.open():
            print(f"[ERROR] Failed to open RTSP stream: {video_path}")
            return
        rtsp_cap.flush(wait_seconds=1.0)
    else:
        rtsp_cap = None

    # ── Load test frame to get native resolution ──────────────────────────
    if is_stream and rtsp_cap:
        ret, test_frame = rtsp_cap.read(timeout=RTSP_READ_TIMEOUT)
        if rtsp_cap:
            rtsp_cap.release()
    else:
        cap = cv2.VideoCapture(video_path)
        ret, test_frame = cap.read()
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap.release()

    if ret and test_frame is not None:
        DISPLAY_HEIGHT, DISPLAY_WIDTH = test_frame.shape[:2]
        print(f"[PARKING] Native resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    else:
        DISPLAY_WIDTH = 1020
        DISPLAY_HEIGHT = 500
        print("[PARKING] Could not grab test frame, defaulting to 1020x500")

    ANNOTATION_SCALE_FACTOR = min(DISPLAY_WIDTH / 1920.0, DISPLAY_HEIGHT / 1080.0)
    print(f"[PARKING] Annotation scale factor: {ANNOTATION_SCALE_FACTOR:.2f}")

    scale_ratio_x = DISPLAY_WIDTH / 1020.0
    scale_ratio_y = DISPLAY_HEIGHT / 500.0

    # ── Load zones ─────────────────────────────────────────────────────────
    class_list = load_class_list(COCO_FILE_PATH)

    raw_areas = define_parking_areas()
    areas = [
        [(int(x * scale_ratio_x), int(y * scale_ratio_y)) for x, y in area]
        for area in raw_areas
    ]
    inner_zones = create_inner_zones(areas, shrink_percentage=0.20)

    raw_entry_line_zones = define_entry_line_zones()
    entry_line_zones_scaled = [
        [(int(x * scale_ratio_x), int(y * scale_ratio_y)) for x, y in zone]
        for zone in raw_entry_line_zones
    ]

    if entry_line_zones_scaled and entry_line_zones_scaled[0]:
        _z = entry_line_zones_scaled[0]
        ENTRY_LINE_CENTER = (
            int(sum(p[0] for p in _z) / len(_z)),
            int(sum(p[1] for p in _z) / len(_z)),
        )
    else:
        ENTRY_LINE_CENTER = (int(201 * scale_ratio_x), int(58 * scale_ratio_y))

    area_arrays = [np.array(a, np.int32) for a in areas]
    inner_zone_arrays = [np.array(z, np.int32) for z in inner_zones]
    area_anchor_pts = [tuple(a[0]) for a in areas]

    # ── Pre-allocate slot layer buffers ─────────────────────────────────────
    _slot_layer = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    _slot_layer_mask = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), dtype=bool)

    # ── Runtime helpers ────────────────────────────────────────────────────
    runtime = ParkingPipelineRuntime()

    # ── Shared refs between threads ─────────────────────────────────────────
    refs = SharedRefs()
    refs.current_parking_status = current_parking_status
    refs.slot_layer = _slot_layer
    refs.slot_layer_mask = _slot_layer_mask
    refs.previous_slot_status = [0] * len(areas)

    # ── Pipeline queues ────────────────────────────────────────────────────
    parking_render_queue: Queue = Queue(maxsize=2)

    # ── Stop event for graceful shutdown ──────────────────────────────────
    stop_event = threading.Event()

    # ── Start daemon threads ───────────────────────────────────────────────
    detect_thread = threading.Thread(
        target=detect_track_worker,
        args=(
            model,
            video_path,
            is_stream,
            stop_event,
            parking_render_queue,
            refs,
            runtime,
            class_list,
            areas,
            area_arrays,
            inner_zone_arrays,
            entry_line_zones_scaled,
            ENTRY_LINE_CENTER,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            ANNOTATION_SCALE_FACTOR,
            scale_ratio_x,
            scale_ratio_y,
            PARKING_CAPTURE_DIR,
        ),
        daemon=True,
        name="ParkingDetectTrack",
    )
    detect_thread.start()
    print(f"[PARKING] Detect+Track thread started (pid={detect_thread.native_id})")

    # ── Start Render Worker as daemon thread ─────────────────────────────────
    # render_worker writes to shared_state.parking_latest_jpeg — never blocks this generator
    render_thread = threading.Thread(
        target=render_worker,
        args=(
            parking_render_queue,
            refs,
            socketio,
            runtime,
            stop_event,
        ),
        daemon=True,
        name="ParkingRender",
    )
    render_thread.start()
    print(f"[PARKING] Render thread started (pid={render_thread.native_id})")

    # ── MJPEG relay: yield parking_latest_jpeg from shared state ─────────────
    # Mirrors gate_camera/camera.py pattern. Never blocks on YOLO inference.
    while True:
        with shared_state.parking_jpeg_lock:
            frame_bytes = shared_state.parking_latest_jpeg
        if frame_bytes:
            yield frame_bytes
        time.sleep(0.033)  # ~30fps relay cadence

    # ── Cleanup ────────────────────────────────────────────────────────────
    stop_event.set()
    detect_thread.join(timeout=5.0)
    print("[PARKING] process_video_stream exited cleanly.")
