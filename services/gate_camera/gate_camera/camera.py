"""
Gate Camera Pipeline — Multi-threaded architecture
===================================================
Luồng xử lý được tách thành 3 worker độc lập để tránh OCR blocking video:

    RTSPCapture (background thread, có sẵn)
         ↓ latest_frame (Lock-protected)
    [Thread 1] Detect+Track Worker
         ├─ YOLO + ByteTrack (mỗi frame)
         ├─ Line-crossing logic (LINE_1, LINE_2)
         ├─ Push plate crop → gate_ocr_crop_queue          (non-blocking)
         └─ Push annotated frame → gate_render_queue        (non-blocking)
                  ↓                             ↓
    [Thread 2] OCR Worker          [Thread 3] Render Worker
         ↓                                     ↓
    stable_plate_cache              gate_latest_jpeg  (shared_state)
         ↑                                     ↓
    best_plate_by_track            MJPEG / API / SocketIO

Không có thay đổi logic trong quy trình FIFO plate matching.
"""

import cv2
import os
import sys
import time
import threading
from datetime import datetime
from collections import defaultdict
# Add parent directories to path for imports
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)

# Add OCR module path to sys.path first
ocr_module_path = os.path.join(base_dir, 'services', 'ocr', 'LicensePlate_OCR_Standalone')
if ocr_module_path not in sys.path:
    sys.path.insert(0, ocr_module_path)

# Import from LicensePlate_OCR_Standalone package
try:
    from plate_detector import LicensePlateDetector, VehicleInfo  # type: ignore[reportMissingImports]
    from ocr_utils import enhanced_plate_preprocessing  # type: ignore[reportMissingImports]
    OCR_AVAILABLE = True
    print("[INFO] OCR module loaded successfully!")
except ImportError as e:
    print(f"[WARNING] OCR module not available: {e}")
    OCR_AVAILABLE = False
    VehicleInfo = None
    def enhanced_plate_preprocessing(img, scale=6):
        import cv2
        if img is None: return img
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

# Import shared state
from shared.state import gate_ocr_results as default_gate_ocr_results
from shared.state import plate_fifo_queue
from shared.state import plate_fifo_lock
import shared.state as shared_state
from services.gate_camera.pipeline import GatePipelineRuntime
from services.gate_camera.workers import ocr_worker, render_worker, detect_track_worker

from config import COCO_FILE_PATH, GATE_LINE_1_Y, GATE_LINE_2_Y, GATE_LINE_3_Y, GATE_LINE_THICKNESS
from database.operations import update_gate_entry_plate, update_gate_entry_media

# Capture save directory for best plate crops
GATE_CAPTURE_DIR = os.path.join(base_dir, 'static', 'gate_captures')
os.makedirs(GATE_CAPTURE_DIR, exist_ok=True)


# Gate detection lines — imported from config.py (backed by .env)
LINE_1_Y = GATE_LINE_1_Y
LINE_2_Y = GATE_LINE_2_Y
LINE_3_Y = GATE_LINE_3_Y
LINE_THICKNESS = GATE_LINE_THICKNESS

# DeepSORT tracks storage (for line crossing logic)
# {track_id: {'line1_crossed': bool, 'line2_crossed': bool, 'direction': None, 'last_y': int, 'last_y2': int}}
vehicle_line_crossings = defaultdict(dict)
# Track history for direction detection
vehicle_tracks = defaultdict(list)  # {track_id: [(cx, cy, frame_count), ...]}

# Initialize OCR detector (lazy loading)
_ocr_detector = None

def get_ocr_detector():
    """Initialize OCR detector (singleton pattern)"""
    global _ocr_detector
    if _ocr_detector is None and OCR_AVAILABLE:
        try:
            print("[INFO] Initializing License Plate OCR detector (plate + OCR only; vehicle = shared yolov8l)...")
            _ocr_detector = LicensePlateDetector(
                vehicle_model=None,   # Skip: gate uses shared yolov8l + coco for vehicle detection
                ocr_method="onnx",
                vehicle_conf=0.5,
                plate_conf=0.20,
                ocr_conf=0.4
            )
            print("[INFO] OCR detector initialized successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to initialize OCR detector: {e}")
            return None
    return _ocr_detector


# Public entry point
# ---------------------------------------------------------------------------

def process_gate_video_stream(video_url, socketio=None, gate_ocr_results_dict=None, process_interval=5):
    runtime = GatePipelineRuntime()

    """
    Process video stream for gate camera with license plate OCR.

    Spawns 3 background daemon threads (Detect+Track, OCR, Render) and
    returns a generator that yields MJPEG frames from shared_state.gate_latest_jpeg.

    Args:
        video_url: Path to video file or RTSP stream URL.
        socketio:  SocketIO instance for real-time updates (optional).
        gate_ocr_results_dict: Dict to store OCR results state (optional).
        process_interval: Run YOLO every N frames (default 5 → ~6fps detection at 30fps input).
    """
    # Convert URL path for local file check
    if video_url.startswith('/static/'):
        video_path_check = os.path.join(base_dir, video_url.lstrip('/'))
    else:
        video_path_check = video_url

    is_stream = video_path_check.lower().startswith(('rtsp://', 'http://', 'https://'))
    if not is_stream and not os.path.exists(video_path_check):
        print(f"[ERROR] Gate video file not found: {video_path_check}")
        return

    if gate_ocr_results_dict is None:
        gate_ocr_results_dict = default_gate_ocr_results

    # ---- Shared mutable state (all threads share the same dict/list objects) ----
    plate_votes_by_track: dict = {}   # {track_id: {plate_text: {'votes': int, 'best_conf': float}}}
    best_plate_by_track:  dict = {}   # {track_id: {'plate': str, 'conf': float}}
    stable_plate_cache:   dict = {}   # {track_id: {'plate': str, 'conf': float, 'cx': int, 'cy': int}}
    best_plate_img_cache: dict = {'img': None}
    track_plate_images:   dict = {}   # {track_id: np.ndarray (upscaled)}
    gate_vehicle_handoffs: dict = {}  # {track_id: handoff_record_dict}

    # Thread stop event
    _stop = threading.Event()

    # Upsert helper — needs references to shared dicts above
    def _upsert_plate_fifo_for_track(track_id, plate_text: str, plate_conf: float, image_path: str = None, ocr_ts: int = None) -> bool:
        """Upgrade matching unassigned FIFO entry in-place if plate improved."""
        if not plate_text:
            return False
        with plate_fifo_lock:
            entry = None
            idx = runtime.queue_index_by_track.get(track_id)
            if idx is not None and 0 <= idx < len(plate_fifo_queue):
                probe = plate_fifo_queue[idx]
                if probe.get('gate_track_id') == track_id:
                    entry = probe
            if entry is None:
                for i, probe in enumerate(plate_fifo_queue):
                    if probe.get('gate_track_id') != track_id:
                        continue
                    entry = probe
                    runtime.queue_index_by_track[track_id] = i
                    break
            if entry is None:
                runtime.queue_index_by_track.pop(track_id, None)
                return False

            prev_plate = entry.get('plate')
            prev_conf  = float(entry.get('conf', 0.0) or 0.0)
            plate_improved = False

            if not prev_plate:
                entry['plate'] = plate_text
                entry['conf']  = float(plate_conf or 0.0)
                entry['timestamp'] = datetime.now()
                plate_improved = True
                print(f"[PLATE QUEUE] Updated no-plate → {plate_text} (track={track_id})")
            elif float(plate_conf or 0.0) > prev_conf:
                entry['plate'] = plate_text
                entry['conf']  = float(plate_conf or 0.0)
                entry['timestamp'] = datetime.now()
                plate_improved = True
                print(f"[PLATE QUEUE] Improved plate {prev_plate}→{plate_text} (track={track_id})")

            # Even if assigned=True, we MUST trigger DB update if OCR improved
            if plate_improved:
                record = gate_vehicle_handoffs.get(track_id, {})
                gate_log_id = record.get('gate_log_id')
                session_id  = record.get('session_id')
                if not gate_log_id:
                    rs = record.get('result_store') or {}
                    gate_log_id = rs.get('gate_log_id')
                    session_id = rs.get('session_id') or session_id
                if not gate_log_id:
                    _ctx = shared_state.gate_ocr_track_db_ctx.get(track_id) or {}
                    gate_log_id = _ctx.get('gate_log_id')
                    session_id = _ctx.get('session_id') or session_id

                if gate_log_id:
                    runtime.submit_high(
                        update_gate_entry_plate,
                        gate_log_id,
                        session_id,
                        plate_text,
                        float(plate_conf or 0.0),
                        image_path=image_path,
                        coalesce_group="gate_entry_update",
                        coalesce_key=f"{gate_log_id}:{ocr_ts}" if ocr_ts else str(gate_log_id),
                    )
                    print(f"[GATE OCR] Triggered DB update for {plate_text} (log_id={gate_log_id}) with image {image_path}")
                    with shared_state.gate_ocr_scheduler_lock:
                        shared_state.gate_ocr_track_db_ctx.pop(track_id, None)
            return True  # Entry found (regardless of whether upgraded)
        return False

    # ---- Launch worker threads ----
    detector = get_ocr_detector()

    ocr_thread = threading.Thread(
        target=ocr_worker,
        args=(detector, plate_votes_by_track, best_plate_by_track,
              track_plate_images, stable_plate_cache, best_plate_img_cache,
              gate_vehicle_handoffs, _upsert_plate_fifo_for_track, _stop,
              runtime.submit_low, GATE_CAPTURE_DIR, runtime),
        daemon=True,
        name="GateOCRWorker",
    )

    render_thread = threading.Thread(
        target=render_worker,
        args=(detector, stable_plate_cache, vehicle_line_crossings,
              vehicle_tracks, [LINE_1_Y], [LINE_2_Y], [LINE_3_Y],
              gate_ocr_results_dict, socketio, _stop, LINE_THICKNESS, runtime),
        daemon=True,
        name="GateRenderWorker",
    )

    detect_thread = threading.Thread(
        target=detect_track_worker,
        args=(video_url, socketio, gate_ocr_results_dict, process_interval, _stop,
              plate_votes_by_track, best_plate_by_track,
              stable_plate_cache, best_plate_img_cache,
              track_plate_images, gate_vehicle_handoffs,
              _upsert_plate_fifo_for_track, runtime.submit_io, runtime),
        kwargs=dict(
            base_dir=base_dir,
            gate_capture_dir=GATE_CAPTURE_DIR,
            get_ocr_detector=get_ocr_detector,
            coco_file_path=COCO_FILE_PATH,
            gate_line_1_y=GATE_LINE_1_Y,
            gate_line_2_y=GATE_LINE_2_Y,
            gate_line_3_y=GATE_LINE_3_Y,
            vehicle_line_crossings=vehicle_line_crossings,
            vehicle_tracks=vehicle_tracks,
        ),
        daemon=True,
        name="GateDetectTrackWorker",
    )

    ocr_thread.start()
    render_thread.start()
    detect_thread.start()

    print("[GATE] Pipeline started: Detect+Track, OCR, Render workers running in background.")

    # ---- Generator: relay gate_latest_jpeg at ~30fps ----
    # This generator is what the background thread in main.py consumes with `for _ in ...`.
    # It never blocks on YOLO or OCR — it just yields jpeg bytes from shared state.
    while True:
        with shared_state.gate_jpeg_lock:
            frame_bytes = shared_state.gate_latest_jpeg
        if frame_bytes:
            yield frame_bytes
        time.sleep(0.033)  # ~30fps relay
