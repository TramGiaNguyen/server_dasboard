"""
workers.py — Parking Detection 2-thread pipeline workers.

Pipeline (mirrors gate_camera/workers.py pattern):
  Thread 1: detect_track_worker
    → RTSP capture read
    → CLAHE (optional)
    → YOLO + ByteTrack inference
    → Slot matching + polygon checks
    → FIFO plate handoff pairing
    → Push annotated frame → parking_render_queue

  Thread 2: render_worker
    → Pop from parking_render_queue
    → Draw OSD overlays (slot occupancy, matched plates, entry zone)
    → JPEG encode (quality 80)
    → Write to shared_state.parking_latest_jpeg

process_video_stream() becomes a thin generator relay that:
  - Initializes config (zones, scale, CLAHE, etc.)
  - Starts 2 daemon threads
  - Yields MJPEG chunks to Flask/SocketIO consumers
"""
import cv2
import time
import os
import sys
import threading
from datetime import datetime
from queue import Empty

import numpy as np
import pandas as pd

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
    PARKING_UPSCALE_DISPLAY,
    PARKING_UPSCALE_WIDTH,
    PARKING_UPSCALE_HEIGHT,
)

# ── Shared state ───────────────────────────────────────────────────────────────
import shared.state as shared_state
from shared.state import (
    plate_fifo_queue,
    plate_fifo_lock,
    matched_vehicles_with_plates,
    parking_trigger_queue,
    parking_trigger_lock,
    handoff_match_lock,
    HANDOFF_QUEUE_ITEM_MAX_AGE_SEC,
    signal_camera_ready,
    wait_for_camera_sync,
    parking_latest_jpeg,
    parking_jpeg_lock,
)
from services.parking_detection.pipeline import ParkingPipelineRuntime

# ── Database operations ───────────────────────────────────────────────────────
from database.operations import (
    update_vehicle_slot,
    update_vehicle_parked_outside,
    release_slot,
    log_improper_parking,
    check_and_notify_slot_hijacked,
)

# ── RTSP capture ──────────────────────────────────────────────────────────────
try:
    from shared.rtsp_capture import RTSPCapture
except ImportError:
    # fallback — keep it simple for non-stream scenarios
    class _DummyRTSP:
        def __init__(self, path, **kw): self.path = path
        def open(self): return True
        def read(self, timeout=0.1):
            import cv2
            cap = cv2.VideoCapture(self.path)
            ret, frame = cap.read()
            cap.release()
            return ret, frame
        def release(self): pass
        def flush(self, **kw): pass
    RTSPCapture = _DummyRTSP

# ── Detection sub-modules ──────────────────────────────────────────────────────
from services.parking_detection.detection import (
    load_class_list,
    define_parking_areas,
    define_entry_line_zones,
    create_inner_zones,
)
from services.parking_detection.drawing import (
    draw_slot_overlay_cached,
    draw_entry_line_zone,
    draw_parked_vehicles,
)
from services.parking_detection.tracking import bbox_iou, rect_intersects_polygon

# ── Tracking ──────────────────────────────────────────────────────────────────
from services.vehicle_tracking.tracker import get_tracker

# ── Constants ─────────────────────────────────────────────────────────────────
RTSP_READ_TIMEOUT = 0.10
MAX_SYNC_WAIT_SEC = 0.15
TARGET_FPS_INTERVAL = 0.033  # ~30 fps

DEBUG_FIFO_MATCH = os.getenv("DEBUG_FIFO_MATCH", "0") == "1"
DEBUG_FIFO_LOG_EVERY_FRAMES = int(os.getenv("DEBUG_FIFO_LOG_EVERY_FRAMES", "30"))
_fifo_debug_last_enqueue_frame_by_track = {}
_fifo_debug_last_pair_miss_frame = 0
_fifo_debug_last_pair_success_frame = 0


# ==============================================================================
# FIFO helper functions (copied from camera.py — no logic changes)
# ==============================================================================

def _fifo_reserved_seq(item: dict):
    v = item.get('reserved_ingress_seq')
    if v is not None:
        return v
    return item.get('reserved_line_num')


def _purge_stale_unpaired_plate_fifo(now):
    with plate_fifo_lock:
        kept = []
        for p in plate_fifo_queue:
            if p.get('assigned'):
                kept.append(p)
                continue
            if _fifo_reserved_seq(p) is not None:
                kept.append(p)
                continue
            ts = p.get('timestamp')
            age = (now - ts).total_seconds() if ts else 0.0
            if age > HANDOFF_QUEUE_ITEM_MAX_AGE_SEC:
                print(f"[FIFO] Dropped stale unpaired gate entry ingress_seq={p.get('ingress_seq')!r} (age {age:.1f}s)")
                continue
            kept.append(p)
        plate_fifo_queue[:] = kept


def _purge_stale_trigger_queue(now):
    with parking_trigger_lock:
        while parking_trigger_queue:
            t = parking_trigger_queue[0]
            ts = t.get('ts')
            if ts is None:
                break
            if (now - ts).total_seconds() > HANDOFF_QUEUE_ITEM_MAX_AGE_SEC:
                _tid = parking_trigger_queue.pop(0).get('track_id')
                print(f"[FIFO] Dropped stale parking trigger track_id={_tid!r}")
            else:
                break


def _try_pair_plate_fifo_with_parking_trigger(frame_counter: int):
    global _fifo_debug_last_pair_miss_frame, _fifo_debug_last_pair_success_frame
    now = datetime.now()
    with handoff_match_lock:
        _purge_stale_unpaired_plate_fifo(now)
        _purge_stale_trigger_queue(now)

        while True:
            fifo_entry = None
            fifo_idx = -1
            with plate_fifo_lock:
                for i, p in enumerate(plate_fifo_queue):
                    if p.get('assigned'):
                        continue
                    if _fifo_reserved_seq(p) is not None:
                        continue
                    fifo_entry = p
                    fifo_idx = i
                    break

            with parking_trigger_lock:
                trigger_depth = len(parking_trigger_queue)
                trig = parking_trigger_queue[0] if parking_trigger_queue else None

            if fifo_entry is None or trig is None:
                if DEBUG_FIFO_MATCH and (frame_counter - _fifo_debug_last_pair_miss_frame) >= DEBUG_FIFO_LOG_EVERY_FRAMES:
                    _fifo_debug_last_pair_miss_frame = frame_counter
                    print(f"[DEBUG FIFO] miss @frame={frame_counter}: fifo_entry={'None' if fifo_entry is None else 'Found'} trig={'None' if trig is None else 'Found'} plate_fifo_depth={len(plate_fifo_queue)} trigger_depth={trigger_depth}")
                return

            dt = (trig['ts'] - fifo_entry['timestamp']).total_seconds()

            if dt < -5.0:
                print(f"[FIFO SYNC] Dropping stale parking trigger {trig.get('track_id')} (dt={dt:.1f}s < -5s)")
                with parking_trigger_lock:
                    if parking_trigger_queue and parking_trigger_queue[0] is trig:
                        parking_trigger_queue.pop(0)
                continue

            if dt > 45.0:
                print(f"[FIFO SYNC] Dropping stale gate entry {fifo_entry.get('ingress_seq')} (dt={dt:.1f}s > 45s)")
                with plate_fifo_lock:
                    if 0 <= fifo_idx < len(plate_fifo_queue):
                        plate_fifo_queue[fifo_idx]['assigned'] = True
                continue

            ingress_seq = fifo_entry.get('ingress_seq')
            if ingress_seq is None:
                return

            fifo_entry['reserved_ingress_seq'] = ingress_seq
            with parking_trigger_lock:
                if parking_trigger_queue and parking_trigger_queue[0] is trig:
                    parking_trigger_queue.pop(0)
                else:
                    return

            matched_vehicles_with_plates[ingress_seq] = {
                'plate': fifo_entry.get('plate'),
                'conf': float(fifo_entry.get('conf', 0.0) or 0.0),
                'plate_status': (
                    'pending' if fifo_entry.get('plate') is None
                    else ('confirmed' if float(fifo_entry.get('conf', 0.0) or 0.0) >= 0.80 else 'provisional')
                ),
                'cx': trig['cx'],
                'cy': trig['cy'],
                'bbox': trig['bbox'],
                'matched_frame': frame_counter,
                'track_id': trig['track_id'],
                'gate_track_id': fifo_entry.get('gate_track_id'),
                'assigned_slot': None,
                'queue_ts': fifo_entry.get('timestamp'),
                'ingress_seq': ingress_seq,
            }
            fifo_entry['assigned'] = True
            if DEBUG_FIFO_MATCH and (frame_counter - _fifo_debug_last_pair_success_frame) >= DEBUG_FIFO_LOG_EVERY_FRAMES:
                _fifo_debug_last_pair_success_frame = frame_counter
                print(f"[DEBUG FIFO] handoff @frame={frame_counter}: ingress_seq={ingress_seq} gate_track_id={fifo_entry.get('gate_track_id')!r} parking_track_id={trig.get('track_id')!r} plate={(fifo_entry.get('plate') or 'PENDING_PLATE')!r} conf={float(fifo_entry.get('conf', 0.0) or 0.0):.2f}")
            print(f"[QUEUE→PARKING] Handoff ingress_seq={ingress_seq} ← track {trig['track_id']} plate={fifo_entry.get('plate') or 'PENDING_PLATE'}")
            break


def _capture_improper_crop(frame, bbox, capture_dir, event_type, plate, frame_counter):
    if frame is None or not bbox:
        return None
    try:
        _fh, _fw = frame.shape[:2]
        _x1c = max(0, int(bbox[0]) - 10)
        _y1c = max(0, int(bbox[1]) - 10)
        _x2c = min(_fw, int(bbox[2]) + 10)
        _y2c = min(_fh, int(bbox[3]) + 10)
        _crop = frame[_y1c:_y2c, _x1c:_x2c]
        if _crop.size == 0:
            return None
        _plate_str = (plate or 'noplate').replace(' ', '_')
        _fname = f"improper_{event_type}_{_plate_str}_{frame_counter}.jpg"
        cv2.imwrite(os.path.join(capture_dir, _fname), _crop)
        return f"/static/parking_captures/{_fname}"
    except Exception:
        return None


# ==============================================================================
# Shared mutable refs — passed as mutable containers so threads see updates
# ==============================================================================

class SharedRefs:
    """Thread-safe container for state that crosses the queue boundary."""
    def __init__(self):
        # These are mutated in-place by detect_track_worker and read by render_worker
        self.parking_space_status = []        # list[int]
        self.outside_vehicles = []           # list[dict]
        self.overlapping_vehicles = []        # list[dict]
        self.slots_info = []                 # list[dict]
        self.occupied_spaces = 0
        self.available_spaces = 0
        self.render_seq = 0
        self.current_parking_status = {}      # dict for api reads
        self.last_emit_status = None
        self.last_emit_time = 0.0
        self.last_improper_summary_frame = 0

        # Drawing caches
        self.last_slot_status_draw = None
        self.slot_layer = None
        self.slot_layer_mask = None

        # Per-frame data
        self.frame_counter = 0
        self.all_vehicles_this_frame = []
        self.track_id_to_vehicle = {}

        # State dicts for detect_track_worker (persisted across frames)
        self.parking_crossing_state = {}
        self.slot_matched_plates = {}
        self.previous_slot_status = []
        self.startup_occupied_slots = set()
        self.recently_removed_track_ids = {}
        self.startup_vehicle_centroids = set()
        self.is_startup_phase = True
        self.vehicle_motion_tracks = {}
        self.vehicle_velocities = {}
        self.slot_empty_counters = {}
        self.slot_stopped_counters = {}
        self.slot19_present_frames = 0
        self.slot19_absent_frames = 0
        self.slot19_state = 0
        self.last_detections = None
        self.last_px = pd.DataFrame()
        self.cached_detections_list = []
        self.improper_park_timers = {}
        self.outside_track_history = {}
        self.overlap_track_history = {}


# ==============================================================================
# Thread 1: Detect + Track Worker
# ==============================================================================

def detect_track_worker(
    model,
    video_path: str,
    is_stream: bool,
    stop_event: threading.Event,
    render_queue,
    refs: SharedRefs,
    runtime: ParkingPipelineRuntime,
    class_list: list,
    areas: list,
    area_arrays: list,
    inner_zone_arrays: list,
    entry_line_zones_scaled: list,
    ENTRY_LINE_CENTER: tuple,
    DISPLAY_WIDTH: int,
    DISPLAY_HEIGHT: int,
    ANNOTATION_SCALE_FACTOR: float,
    scale_ratio_x: float,
    scale_ratio_y: float,
    PARKING_CAPTURE_DIR: str,
):
    """
    Reads frames, runs YOLO+ByteTrack, performs slot matching + FIFO pairing,
    updates all shared state, then pushes annotated frame to render_queue.
    """
    print("[PARKING] Detect+Track Worker started")

    # CLAHE
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if PARKING_USE_CLAHE else None

    # ── RTSP/file capture ────────────────────────────────────────────────────
    if is_stream:
        rtsp_cap = RTSPCapture(video_path, buffer_size=2)
        if not rtsp_cap.open():
            print(f"[ERROR] Failed to open RTSP stream: {video_path}")
            return
        cap = None
        signal_camera_ready('parking')
        if not wait_for_camera_sync(timeout=60):
            print("[ERROR] Camera sync timeout!")
        rtsp_cap.flush(wait_seconds=1.0)
        print("[PARKING] ✓ Sync complete, starting detection loop.")
    else:
        cap = cv2.VideoCapture(video_path)
        rtsp_cap = None
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video file: {video_path}")
            return

    total_spaces = len(areas)
    scale_x = DISPLAY_WIDTH / DISPLAY_WIDTH  # 1.0 (native resolution)
    scale_y = DISPLAY_HEIGHT / DISPLAY_HEIGHT

    # Derived constants
    _avg_ratio = (scale_ratio_x + scale_ratio_y) / 2.0
    ENTRY_LINE_EXCLUDE_RADIUS = int(45 * _avg_ratio)
    if entry_line_zones_scaled and entry_line_zones_scaled[0]:
        try:
            _poly = entry_line_zones_scaled[0]
            _cx, _cy = ENTRY_LINE_CENTER
            _max_r = max((((int(x) - _cx) ** 2 + (int(y) - _cy) ** 2) ** 0.5) for x, y in _poly) if _poly else 0.0
            _poly_based_r = int(_max_r * 0.7)
            ENTRY_LINE_EXCLUDE_RADIUS = max(10, min(ENTRY_LINE_EXCLUDE_RADIUS, _poly_based_r))
        except Exception:
            pass
    TRANSIT_RECOVERY_MAX_PX = int(45 * _avg_ratio)
    STARTUP_FRAMES = 30
    SLOT_EMPTY_THRESHOLD = 45
    STOPPED_FRAMES_THRESHOLD = 30
    MOTION_THRESHOLD = 3.0
    MOTION_TRACK_HISTORY = 10
    PERIODIC_REDETECT_INTERVAL = 1
    RECENTLY_REMOVED_TTL = 90
    IMPROPER_PARK_LOG_DELAY_SECONDS = 30
    IMPROPER_PARK_MAX_MOVE_PX = 40
    IMPROPER_PARK_MIN_BBOX_AREA = 2000
    STARTUP_GRACE_FRAMES = 60

    r = refs
    frame_counter = 0
    runtime_io = runtime  # alias

    while not stop_event.is_set():
        frame_start_time = time.time()

        # ── Read frame ──────────────────────────────────────────────────────
        if is_stream:
            ret, frame = rtsp_cap.read(timeout=RTSP_READ_TIMEOUT)
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        if not ret or frame is None:
            continue

        raw_frame = frame.copy()
        frame_counter += 1
        r.frame_counter = frame_counter

        # ── Soft video sync ─────────────────────────────────────────────────
        shared_state.update_parking_frame(frame_counter)
        _sync_wait_start = time.time()
        while shared_state.should_parking_wait():
            if time.time() - _sync_wait_start >= MAX_SYNC_WAIT_SEC:
                break
            time.sleep(0.01)

        # ── CLAHE preprocessing ──────────────────────────────────────────────
        detection_frame = frame
        if PARKING_USE_CLAHE and _clahe is not None:
            lab = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2LAB)
            _l, _a, _b = cv2.split(lab)
            _l = _clahe.apply(_l)
            lab = cv2.merge([_l, _a, _b])
            detection_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # ── YOLO + ByteTrack ─────────────────────────────────────────────────
        results = model.track(
            detection_frame, persist=True, tracker=PARKING_TRACKER_CONFIG,
            conf=0.20, iou=0.5, verbose=False
        )
        detections = []
        _bt_vehicles = []
        _boxes = results[0].boxes if (results and results[0].boxes is not None) else None
        if _boxes is not None and _boxes.id is not None:
            _xyxy = _boxes.xyxy.cpu().numpy()
            _ids = _boxes.id.cpu().numpy().astype(int)
            _confs = _boxes.conf.cpu().numpy()
            _cls_np = _boxes.cls.cpu().numpy().astype(int)
            for _i in range(len(_xyxy)):
                _ci = int(_cls_np[_i])
                if not (0 <= _ci < len(class_list)):
                    continue
                _cn = class_list[_ci]
                if _cn not in ('car', 'bus', 'truck'):
                    continue
                _x1, _y1, _x2, _y2 = _xyxy[_i]
                _x1 = int(_x1 * scale_x); _y1 = int(_y1 * scale_y)
                _x2 = int(_x2 * scale_x); _y2 = int(_y2 * scale_y)
                if _x2 <= _x1 or _y2 <= _y1:
                    continue
                _cf = float(_confs[_i])
                _tid = int(_ids[_i])
                _cx = (_x1 + _x2) // 2
                _cy = (_y1 + _y2) // 2
                detections.append([_x1, _y1, _x2, _y2, _cf, _ci])
                _bt_vehicles.append({
                    'cx': _cx, 'cy': _cy,
                    'bbox': [_x1, _y1, _x2, _y2],
                    'grid_pos': (_cx // 30, _cy // 30),
                    'track_id': _tid,
                    'tsu': 0,
                    'class_name': _cn,
                })

        r.all_vehicles_this_frame = _bt_vehicles
        r.track_id_to_vehicle = {
            v['track_id']: v for v in _bt_vehicles if v.get('track_id') is not None
        }

        # ── Slot occupancy (parked vehicles) ────────────────────────────────
        parking_space_status = [0] * len(areas)
        outside_vehicles = []
        overlapping_vehicles = []

        if _bt_vehicles:
            parked_info = draw_parked_vehicles(
                frame, _bt_vehicles, areas, inner_zone_arrays,
                class_list, ANNOTATION_SCALE_FACTOR
            )
            for slot_num, vehicles in parked_info['slot_occupancy'].items():
                slot_idx = slot_num - 1
                if 0 <= slot_idx < len(parking_space_status):
                    parking_space_status[slot_idx] = len(vehicles)

        r.parking_space_status = parking_space_status
        r.outside_vehicles = outside_vehicles
        r.overlapping_vehicles = overlapping_vehicles

        # ── Tracking: match vehicles from gate ───────────────────────────────
        if TRACKING_ENABLED:
            tracker = get_tracker()
            matched_vehicles_info = []

            # Startup phase
            if r.is_startup_phase:
                if frame_counter <= STARTUP_FRAMES:
                    for row in detections:
                        x1, y1, x2, y2, confidence, class_idx = map(float, row)
                        x1, y1, x2, y2, class_idx = int(x1), int(y1), int(x2), int(y2), int(class_idx)
                        class_name = class_list[class_idx]
                        if class_name in ['car', 'bus', 'truck']:
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            r.startup_vehicle_centroids.add((cx // 50, cy // 50))
                else:
                    r.is_startup_phase = False
                    print(f"[TRACKING] Startup phase complete. {len(r.startup_vehicle_centroids)} existing vehicle positions captured.")

            newly_occupied_slots = set()
            for i, (current, previous) in enumerate(zip(parking_space_status, r.previous_slot_status)):
                if current > 0 and previous == 0:
                    newly_occupied_slots.add(i + 1)

            if frame_counter <= STARTUP_FRAMES:
                for i, status in enumerate(parking_space_status):
                    if status > 0:
                        r.startup_occupied_slots.add(i + 1)

            # Clear matched plates for empty slots
            for i, status in enumerate(parking_space_status):
                slot_num = i + 1
                if status == 0 and slot_num in r.slot_matched_plates:
                    r.slot_empty_counters[slot_num] = r.slot_empty_counters.get(slot_num, 0) + 1
                    if r.slot_empty_counters[slot_num] >= SLOT_EMPTY_THRESHOLD:
                        _still_tracked = any(
                            _minfo.get('assigned_slot') == slot_num
                            for _minfo in matched_vehicles_with_plates.values()
                        )
                        if _still_tracked:
                            r.slot_empty_counters[slot_num] = 0
                            continue
                        _cleared_plate = r.slot_matched_plates[slot_num].get('plate', 'N/A')
                        print(f"[TRACKING] Slot {slot_num} empty for {SLOT_EMPTY_THRESHOLD} frames, clearing matched plate: {_cleared_plate}")
                        del r.slot_matched_plates[slot_num]
                        del r.slot_empty_counters[slot_num]
                        r.startup_occupied_slots.discard(slot_num)
                elif status > 0 and slot_num in r.slot_empty_counters:
                    del r.slot_empty_counters[slot_num]

            # ── Per-detection loop: slot matching + motion ─────────────────
            for row in detections:
                x1, y1, x2, y2, confidence, class_idx = map(float, row)
                x1, y1, x2, y2, class_idx = int(x1), int(y1), int(x2), int(y2), int(class_idx)
                class_name = class_list[class_idx]
                if class_name not in ['car', 'bus', 'truck']:
                    continue

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Position-based match to matched_vehicles_with_plates
                already_matched_info = None
                for _seq, minfo in matched_vehicles_with_plates.items():
                    if minfo.get('assigned_slot') is not None:
                        continue
                    mx, my = minfo.get('cx', 0), minfo.get('cy', 0)
                    if (cx // 50, cy // 50) == (mx // 50, my // 50):
                        already_matched_info = {'plate': minfo.get('plate'), 'ticket_id': minfo.get('ticket_id', '')}
                        break

                slot_number = None
                for i, area_np in enumerate(area_arrays):
                    if cv2.pointPolygonTest(area_np, (float(cx), float(cy)), False) >= 0:
                        slot_number = i + 1
                        break

                if already_matched_info:
                    matched_vehicles_info.append({
                        'bbox': [x1, y1, x2, y2],
                        'plate': already_matched_info.get('plate', 'N/A'),
                        'ticket_id': already_matched_info.get('ticket_id', ''),
                        'slot': f'Slot {slot_number}' if slot_number else 'DRIVING'
                    })

                if slot_number is None:
                    continue

                # Motion tracking
                vehicle_id = f"parking_{slot_number}_{(cx//50)*50}_{(cy//50)*50}"
                if vehicle_id not in r.vehicle_motion_tracks:
                    r.vehicle_motion_tracks[vehicle_id] = []
                r.vehicle_motion_tracks[vehicle_id].append((cx, cy, frame_counter))
                if len(r.vehicle_motion_tracks[vehicle_id]) > MOTION_TRACK_HISTORY:
                    r.vehicle_motion_tracks[vehicle_id] = r.vehicle_motion_tracks[vehicle_id][-MOTION_TRACK_HISTORY:]

                track = r.vehicle_motion_tracks[vehicle_id]
                if len(track) >= 3:
                    total_distance = sum(
                        (((track[i][0] - track[i-1][0]) ** 2 + (track[i][1] - track[i-1][1]) ** 2) ** 0.5)
                        for i in range(1, len(track))
                    )
                    frames_elapsed = track[-1][2] - track[0][2]
                    velocity = total_distance / max(frames_elapsed, 1)
                    r.vehicle_velocities[vehicle_id] = velocity
                else:
                    velocity = r.vehicle_velocities.get(vehicle_id, 999)

                is_stopped = velocity < 1.0

                if is_stopped:
                    r.slot_stopped_counters[slot_number] = r.slot_stopped_counters.get(slot_number, 0) + 1
                else:
                    if slot_number in r.slot_stopped_counters:
                        del r.slot_stopped_counters[slot_number]

                stopped_frames = r.slot_stopped_counters.get(slot_number, 0)
                truly_parked = stopped_frames >= STOPPED_FRAMES_THRESHOLD

                # IoU with tracked vehicles for slot_track_id
                slot_track_id = None
                best_iou = 0.0
                for v in r.all_vehicles_this_frame:
                    vx1, vy1, vx2, vy2 = v['bbox']
                    ix1 = max(x1, vx1); iy1 = max(y1, vy1)
                    ix2 = min(x2, vx2); iy2 = min(y2, vy2)
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    if inter <= 0:
                        continue
                    det_area = (x2 - x1) * (y2 - y1)
                    trk_area = (vx2 - vx1) * (vy2 - vy1)
                    union = det_area + trk_area - inter
                    if union <= 0:
                        continue
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        slot_track_id = v.get('track_id')

                # Direct plate observation
                if (truly_parked and slot_number not in r.slot_matched_plates):
                    for _isq, _minfo in matched_vehicles_with_plates.items():
                        _mp = _minfo.get('plate')
                        if not _mp or _minfo.get('assigned_slot') not in (None, slot_number):
                            continue
                        _mcx = _minfo.get('cx', 0)
                        _mcy = _minfo.get('cy', 0)
                        if cv2.pointPolygonTest(area_arrays[slot_number - 1], (float(_mcx), float(_mcy)), False) >= 0:
                            _minfo['assigned_slot'] = slot_number
                            r.slot_matched_plates[slot_number] = {
                                'plate': _mp, 'conf': _minfo.get('conf', 0.0),
                                'matched_time': datetime.now().isoformat(), 'ingress_seq': _isq,
                            }
                            r.startup_occupied_slots.discard(slot_number)
                            print(f"[DIRECT-OBS] Slot {slot_number} ← {_mp} (direct observation)")
                            runtime_io.submit_high(
                                update_vehicle_slot, _mp, slot_number,
                                coalesce_group="update_vehicle_slot",
                                coalesce_key=f"{_mp}:{slot_number}",
                            )
                            break

                # FIFO gate-handoff: transfer plate to slot
                if truly_parked and slot_track_id is not None and slot_number not in r.slot_matched_plates:
                    if slot_number in r.startup_occupied_slots:
                        continue

                    best_match = None
                    best_dist = float('inf')
                    best_ingress_seq = None
                    for ingress_seq, match_info in matched_vehicles_with_plates.items():
                        if match_info.get('assigned_slot') is not None:
                            continue
                        match_cx = match_info.get('cx', cx)
                        match_cy = match_info.get('cy', cy)
                        dist_inside = cv2.pointPolygonTest(
                            area_arrays[slot_number - 1], (float(match_cx), float(match_cy)), True
                        )
                        if dist_inside < -20:
                            continue
                        dist = ((match_cx - cx) ** 2 + (match_cy - cy) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_match = match_info
                            best_ingress_seq = ingress_seq

                    MAX_TRANSFER_DISTANCE = 120
                    if best_match and best_dist < MAX_TRANSFER_DISTANCE:
                        plate_val = best_match.get('plate', 'N/A')
                        already_bound_elsewhere = False
                        for _isq, _minfo in matched_vehicles_with_plates.items():
                            if _minfo.get('plate') == plate_val:
                                other_slot = _minfo.get('assigned_slot')
                                if other_slot is not None and other_slot != slot_number:
                                    already_bound_elsewhere = True
                                    break
                        if not already_bound_elsewhere:
                            best_match['assigned_slot'] = slot_number
                            r.slot_matched_plates[slot_number] = {
                                'plate': plate_val, 'conf': best_match.get('conf', 0.0),
                                'matched_time': datetime.now().isoformat(), 'ingress_seq': best_ingress_seq,
                            }
                            print(f"[TRANSFER] Slot {slot_number} ← {plate_val} (dist: {best_dist:.1f}px, ingress_seq: {best_ingress_seq})")
                            runtime_io.submit_high(
                                update_vehicle_slot, plate_val, slot_number,
                                coalesce_group="update_vehicle_slot",
                                coalesce_key=f"{plate_val}:{slot_number}",
                            )
                            if slot_number in r.slot_stopped_counters:
                                del r.slot_stopped_counters[slot_number]

                # Slot lock UI
                if slot_number in r.slot_matched_plates:
                    matched_info = r.slot_matched_plates[slot_number]
                    matched_vehicles_info.append({
                        'bbox': [x1, y1, x2, y2],
                        'plate': matched_info.get('plate', 'N/A'),
                        'ticket_id': matched_info.get('ticket_id', ''),
                        'score': matched_info.get('score', 0),
                        'slot': slot_number
                    })

                is_moving = velocity > MOTION_THRESHOLD
                if not is_moving and tracker.get_pending_count() > 0:
                    continue
                if slot_number not in newly_occupied_slots:
                    continue

            r.previous_slot_status = parking_space_status.copy()

            # Draw matched vehicle outlines
            for info in matched_vehicles_info:
                x1, y1, x2, y2 = info['bbox']
                border_thickness = max(2, int(3 * ANNOTATION_SCALE_FACTOR))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), border_thickness)
                cx_match = (x1 + x2) // 2; cy_match = (y1 + y2) // 2
                dot_radius = max(4, int(8 * ANNOTATION_SCALE_FACTOR))
                cv2.circle(frame, (cx_match, cy_match), dot_radius, (255, 0, 255), -1)

            # Draw entry trigger zone
            if entry_line_zones_scaled:
                draw_entry_line_zone(frame, entry_line_zones_scaled[0], ANNOTATION_SCALE_FACTOR)

            # ── Line crossing detection ─────────────────────────────────────
            all_vehicles_this_frame = r.all_vehicles_this_frame

            for vehicle_data in all_vehicles_this_frame:
                cx, cy = vehicle_data['cx'], vehicle_data['cy']
                x1, y1, x2, y2 = vehicle_data['bbox']
                track_id = vehicle_data['track_id']
                _veh_class = (vehicle_data.get('class_name') or vehicle_data.get('det_class') or 'car').lower()
                _is_motor_vehicle = _veh_class in ('car', 'bus', 'truck')
                if not _is_motor_vehicle:
                    continue

                if track_id not in r.parking_crossing_state:
                    r.parking_crossing_state[track_id] = {
                        'last_y': cy, 'last_y2': y2,
                        'crossed_entry_line': False,
                        'first_seen_frame': frame_counter,
                        'in_line_zone_prev': False,
                        'trigger_enqueued': False,
                    }

                state = r.parking_crossing_state[track_id]

                if frame_counter <= STARTUP_GRACE_FRAMES and state['first_seen_frame'] <= STARTUP_GRACE_FRAMES:
                    state['last_y'] = cy; state['last_y2'] = y2
                    continue

                in_line_zone_now = False
                if entry_line_zones_scaled:
                    for _lz in entry_line_zones_scaled:
                        if rect_intersects_polygon([x1, y1, x2, y2], _lz):
                            in_line_zone_now = True
                            break

                if not state['crossed_entry_line']:
                    crossed = False
                    crossing_entering = False
                    is_appeared_entry = False
                    in_line_zone_prev = state.get('in_line_zone_prev', False)

                    if (not in_line_zone_prev) and in_line_zone_now:
                        frames_since_first = frame_counter - state['first_seen_frame']
                        if frames_since_first >= 1:
                            crossed = True; crossing_entering = True; is_appeared_entry = True
                            print(f"[PARKING] Track {track_id} HIT entry trigger zone (cx={cx}, cy={cy})")
                    elif entry_line_zones_scaled and rect_intersects_polygon([x1, y1, x2, y2], entry_line_zones_scaled[0]):
                        _fs = frame_counter - state['first_seen_frame']
                        if _fs <= 30 and _fs >= 1:
                            crossed = True; crossing_entering = True; is_appeared_entry = True

                    if crossed and crossing_entering:
                        _dx = cx - ENTRY_LINE_CENTER[0]; _dy = cy - ENTRY_LINE_CENTER[1]
                        skip_exclude = is_appeared_entry
                        if not skip_exclude and _dx * _dx + _dy * _dy <= ENTRY_LINE_EXCLUDE_RADIUS * ENTRY_LINE_EXCLUDE_RADIUS:
                            state['crossed_entry_line'] = False
                        else:
                            is_inside_slot = False
                            for area in areas:
                                area_np = np.array(area, dtype=np.int32)
                                if cv2.pointPolygonTest(area_np, (float(cx), float(cy)), False) >= 0:
                                    is_inside_slot = True
                                    break
                            if is_inside_slot:
                                state['crossed_entry_line'] = True
                            elif not state.get('trigger_enqueued'):
                                tid_in_matched = any(
                                    info.get('track_id') == track_id
                                    for info in matched_vehicles_with_plates.values()
                                )
                                if tid_in_matched:
                                    state['crossed_entry_line'] = True
                                else:
                                    with parking_trigger_lock:
                                        dup_in_queue = any(
                                            t.get('track_id') == track_id for t in parking_trigger_queue
                                        )
                                    if not dup_in_queue:
                                        with parking_trigger_lock:
                                            parking_trigger_queue.append({
                                                'track_id': track_id, 'cx': cx, 'cy': cy,
                                                'bbox': [x1, y1, x2, y2],
                                                'frame_counter': frame_counter,
                                                'ts': datetime.now(),
                                            })
                                        state['trigger_enqueued'] = True
                                        print(f"[PARKING] Track {track_id} queued parking trigger (FIFO match)")
                                    state['crossed_entry_line'] = True
                    elif crossed:
                        state['crossed_entry_line'] = True

                state['last_y'] = cy; state['last_y2'] = y2
                if not state['crossed_entry_line']:
                    state['in_line_zone_prev'] = in_line_zone_now

            # ── FIFO pairing ───────────────────────────────────────────────
            track_id_to_vehicle = r.track_id_to_vehicle
            _try_pair_plate_fifo_with_parking_trigger(frame_counter)

            # ── Step 5: Update matched vehicles positions ───────────────────
            matched_to_remove = []

            track_id_to_ingress = {}
            for ingress_seq, match_info in list(matched_vehicles_with_plates.items()):
                tracked_id = match_info.get('track_id')
                if tracked_id is not None:
                    if tracked_id in track_id_to_ingress:
                        old_seq = track_id_to_ingress[tracked_id]
                        old_info = matched_vehicles_with_plates[old_seq]
                        old_last_seen = old_info.get('last_seen_frame', 0)
                        new_last_seen = match_info.get('last_seen_frame', 0)
                        if new_last_seen > old_last_seen:
                            print(f"⚠️  [TRACK REUSE] Track ID {tracked_id} reused! Removing old ingress_seq={old_seq}")
                            matched_to_remove.append(old_seq)
                            track_id_to_ingress[tracked_id] = ingress_seq
                        else:
                            matched_to_remove.append(ingress_seq)
                    else:
                        track_id_to_ingress[tracked_id] = ingress_seq

            used_track_ids = {
                info.get('track_id')
                for info in matched_vehicles_with_plates.values()
                if info.get('track_id') is not None and info.get('ingress_seq') not in matched_to_remove
            }

            for ingress_seq, match_info in matched_vehicles_with_plates.items():
                plate_text = match_info.get('plate')
                plate_conf = match_info.get('conf', 0.0)

                last_seen = match_info.get('last_seen_frame', match_info.get('matched_frame', 0))
                frames_missing = frame_counter - last_seen

                found_current_pos = False
                best_match = None
                tracked_id = match_info.get('track_id')

                if tracked_id is not None:
                    if tracked_id in track_id_to_vehicle:
                        best_match = track_id_to_vehicle[tracked_id]
                        found_current_pos = True
                    else:
                        for _dead_tid in list(r.recently_removed_track_ids):
                            if frame_counter - r.recently_removed_track_ids[_dead_tid] > RECENTLY_REMOVED_TTL:
                                del r.recently_removed_track_ids[_dead_tid]

                        assigned_slot = match_info.get('assigned_slot')
                        _is_parked_outside = match_info.get('parked_outside', False)
                        _prev_bbox = match_info.get('bbox')

                        if _is_parked_outside and _prev_bbox is not None:
                            for _v in all_vehicles_this_frame:
                                _iou = bbox_iou(_prev_bbox, _v['bbox'])
                                if _iou > 0.4:
                                    match_info['cx'] = _v['cx']; match_info['cy'] = _v['cy']
                                    match_info['bbox'] = _v['bbox']
                                    match_info['last_seen_frame'] = frame_counter
                                    found_current_pos = True
                                    best_match = _v
                                    _ip_key = f"outside_{ingress_seq}"
                                    if _ip_key in r.improper_park_timers:
                                        r.improper_park_timers[_ip_key]['last_bbox'] = list(_v['bbox'])
                                        r.improper_park_timers[_ip_key]['plate'] = plate_text or r.improper_park_timers[_ip_key]['plate']
                                    break
                        elif assigned_slot is None and _prev_bbox is not None:
                            _hist = match_info.get('pos_hist') or []
                            _speed = 999.0
                            if len(_hist) >= 2:
                                _x0, _y0, _f0 = _hist[0]; _x1, _y1, _f1 = _hist[-1]
                                _df = max(1, _f1 - _f0)
                                _speed = (((_x1 - _x0) ** 2 + (_y1 - _y0) ** 2) ** 0.5) / _df
                            _is_static = _speed < 0.8
                            _r = min(25 if _is_static else 60, TRANSIT_RECOVERY_MAX_PX)
                            _recovery_sq = _r * _r
                            _best_sq = _recovery_sq
                            _recovered = None

                            _prev_area = max(0, (_prev_bbox[2] - _prev_bbox[0])) * max(0, (_prev_bbox[3] - _prev_bbox[1]))

                            for _v in all_vehicles_this_frame:
                                _tid = _v.get('track_id')
                                if _tid is None:
                                    continue
                                if _tid in used_track_ids and _tid != tracked_id:
                                    continue
                                if _tid in r.recently_removed_track_ids:
                                    continue
                                _in_slot = False
                                for _area_np in area_arrays:
                                    if cv2.pointPolygonTest(_area_np, (float(_v['cx']), float(_v['cy'])), False) >= 0:
                                        _in_slot = True; break
                                if _in_slot:
                                    continue
                                _d = (_v['cx'] - match_info['cx']) ** 2 + (_v['cy'] - match_info['cy']) ** 2
                                if _d >= _best_sq:
                                    continue
                                _iou = bbox_iou(_prev_bbox, _v['bbox'])
                                _iou_th = 0.6 if _is_static else 0.4
                                if _iou < _iou_th:
                                    continue
                                _va = max(0, (_v['bbox'][2] - _v['bbox'][0])) * max(0, (_v['bbox'][3] - _v['bbox'][1]))
                                if _va <= 0:
                                    continue
                                _ratio = _va / max(_prev_area, 1)
                                if _ratio < 0.5 or _ratio > 2.0:
                                    continue
                                _best_sq = _d; _recovered = _v

                            if _recovered is not None:
                                _last_rb = match_info.get('last_rebind_frame', -999999)
                                if frame_counter - _last_rb >= 10:
                                    match_info['last_rebind_frame'] = frame_counter
                                    old_tid = match_info.get('track_id')
                                    new_tid = _recovered['track_id']
                                    match_info['track_id'] = new_tid
                                    best_match = _recovered
                                    found_current_pos = True
                                    print(f"[RECOVERY] Transit ingress_seq {ingress_seq}: re-bound track {old_tid}→{new_tid}")
                else:
                    max_legacy_sq = TRANSIT_RECOVERY_MAX_PX * TRANSIT_RECOVERY_MAX_PX
                    min_dist_sq = max_legacy_sq + 1
                    for vehicle in all_vehicles_this_frame:
                        _vcls = (vehicle.get('class_name') or vehicle.get('det_class') or '').lower()
                        if _vcls not in ('car', 'bus', 'truck'):
                            continue
                        dist_sq = (vehicle['cx'] - match_info['cx']) ** 2 + (vehicle['cy'] - match_info['cy']) ** 2
                        if dist_sq < min_dist_sq and dist_sq <= max_legacy_sq:
                            min_dist_sq = dist_sq; best_match = vehicle
                    if best_match:
                        match_info['track_id'] = best_match.get('track_id')
                        found_current_pos = True

                if best_match:
                    match_info['cx'] = best_match['cx']; match_info['cy'] = best_match['cy']
                    match_info['bbox'] = best_match['bbox']
                    match_info['last_seen_frame'] = frame_counter
                    found_current_pos = True

                    _hist = match_info.get('pos_hist')
                    if _hist is None:
                        _hist = []; match_info['pos_hist'] = _hist
                    _hist.append((match_info['cx'], match_info['cy'], frame_counter))
                    if len(_hist) > 15:
                        del _hist[:-15]

                    # Check in-slot parked
                    _in_slot_parked = False
                    for i, area in enumerate(areas):
                        area_np = np.array(area, dtype=np.int32)
                        dist_to_slot = cv2.pointPolygonTest(area_np, (float(match_info['cx']), float(match_info['cy'])), True)
                        if dist_to_slot > 15.0:
                            if 'parked' not in match_info:
                                match_info['parked'] = True
                                match_info['parked_slot'] = i + 1
                                print(f"[TRACKING] Vehicle ingress_seq {ingress_seq} parked in Slot {i+1}")
                                if plate_text:
                                    runtime_io.submit_high(
                                        update_vehicle_slot, plate_text, i + 1,
                                        coalesce_group="update_vehicle_slot",
                                        coalesce_key=f"{plate_text}:{i + 1}",
                                    )
                            _in_slot_parked = True; break

                    _fifo_ip_key = f"outside_{ingress_seq}"
                    if _fifo_ip_key in r.improper_park_timers and not r.improper_park_timers[_fifo_ip_key].get('logged'):
                        _cur_bbox = match_info.get('bbox')
                        if _cur_bbox:
                            r.improper_park_timers[_fifo_ip_key]['last_bbox'] = list(_cur_bbox)
                        if len(_hist) >= 5:
                            _rx0, _ry0, _rf0 = _hist[-5]; _rx1, _ry1, _rf1 = _hist[-1]
                            _rdf = max(1, _rf1 - _rf0)
                            _rspeed = ((_rx1 - _rx0) ** 2 + (_ry1 - _ry0) ** 2) ** 0.5 / _rdf
                            if _rspeed >= 1.5:
                                r.improper_park_timers.pop(_fifo_ip_key, None)
                                match_info['parked_outside'] = False

                    # Check outside-slot parked
                    if not _in_slot_parked and not match_info.get('parked_outside', False):
                        if len(_hist) >= 5:
                            _x0, _y0, _f0 = _hist[0]; _x1o, _y1o, _f1o = _hist[-1]
                            _df = max(1, _f1o - _f0)
                            _total_move = ((_x1o - _x0) ** 2 + (_y1o - _y0) ** 2) ** 0.5
                            _avg_speed = _total_move / _df
                            if _avg_speed < 0.5:
                                match_info['parked_outside'] = True
                                print(f"[TRACKING] Vehicle ingress_seq {ingress_seq} parked OUTSIDE slots (speed={_avg_speed:.2f}px/f)")
                                if plate_text:
                                    runtime_io.submit_high(
                                        update_vehicle_parked_outside, plate_text,
                                        coalesce_group="parked_outside", coalesce_key=plate_text,
                                    )
                                _ip_key = f"outside_{ingress_seq}"
                                if _ip_key not in r.improper_park_timers:
                                    # cancel any generic track_{tid} timer — FIFO owns this vehicle now
                                    _tracked_id = match_info.get('track_id')
                                    if _tracked_id is not None:
                                        r.improper_park_timers.pop(f"track_{_tracked_id}", None)
                                    _fifo_bbox = match_info.get('bbox', [0, 0, 0, 0])
                                    _init_cx = (_fifo_bbox[0] + _fifo_bbox[2]) / 2.0
                                    _init_cy = (_fifo_bbox[1] + _fifo_bbox[3]) / 2.0
                                    r.improper_park_timers[_ip_key] = {
                                        'type': 'outside', 'first_detected': datetime.now(),
                                        'logged': False, 'plate': plate_text,
                                        'last_bbox': list(_fifo_bbox),
                                        'init_center': (_init_cx, _init_cy), 'image_path': None,
                                    }

                # Draw plate UI
                can_draw = found_current_pos
                if not can_draw and match_info.get('bbox') is not None:
                    if frames_missing <= 10:
                        can_draw = True

                if can_draw:
                    x1, y1, x2, y2 = match_info['bbox']
                    plate_status = match_info.get('plate_status', 'confirmed' if plate_text else 'pending')
                    if plate_text:
                        label = plate_text if plate_status == 'confirmed' else f"Tam: {plate_text}"
                        bg_color = (0, 200, 255) if plate_status == 'confirmed' else (0, 165, 255)
                        txt_color = (0, 0, 0)
                    else:
                        label = "Nhan dien bien so..."
                        bg_color = (100, 100, 100); txt_color = (255, 255, 255)
                    font_scale = 0.7 * ANNOTATION_SCALE_FACTOR
                    thickness = max(1, int(2 * ANNOTATION_SCALE_FACTOR))
                    padding = max(6, int(6 * ANNOTATION_SCALE_FACTOR))
                    offset_y = max(10, int(10 * ANNOTATION_SCALE_FACTOR))
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    center_x = (x1 + x2) // 2
                    label_x1 = center_x - tw // 2 - padding
                    label_y1 = max(0, y1 - th - offset_y - padding // 2)
                    cv2.rectangle(frame, (label_x1, label_y1), (label_x1 + tw + padding * 2, y1 - padding), bg_color, -1)
                    cv2.putText(frame, label, (label_x1 + padding, y1 - offset_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thickness)

                is_parked = match_info.get('parked', False)
                is_parked_outside_flag = match_info.get('parked_outside', False)
                if is_parked:
                    grace = 300
                elif is_parked_outside_flag:
                    grace = 300
                else:
                    grace = 240 if plate_text is None else 150

                if frames_missing > grace:
                    matched_to_remove.append(ingress_seq)
                    reason = 'in-slot' if is_parked else ('outside-slot' if is_parked_outside_flag else 'in-transit')
                    print(f"[FIFO] Plate {plate_text} (ingress_seq {ingress_seq}) removed - {reason}, missing {frames_missing}f")

            # Clean up old matched vehicles
            for rm_seq in matched_to_remove:
                if rm_seq in matched_vehicles_with_plates:
                    _removed_info = matched_vehicles_with_plates[rm_seq]
                    _removed_tid = _removed_info.get('track_id')
                    if _removed_tid is not None:
                        r.recently_removed_track_ids[_removed_tid] = frame_counter
                    _parked_slot = _removed_info.get('parked_slot')
                    if _parked_slot:
                        runtime_io.submit_low(
                            release_slot, _parked_slot,
                            coalesce_group="release_slot", coalesce_key=str(_parked_slot),
                        )
                    del matched_vehicles_with_plates[rm_seq]
                r.improper_park_timers.pop(f"outside_{rm_seq}", None)

            # Track ID conflicts resolution
            track_id_conflicts = {}
            plate_conflicts = {}
            for ingress_seq, match_info in list(matched_vehicles_with_plates.items()):
                tracked_id = match_info.get('track_id')
                plate = match_info.get('plate')
                if tracked_id is not None:
                    if tracked_id not in track_id_conflicts:
                        track_id_conflicts[tracked_id] = []
                    track_id_conflicts[tracked_id].append((ingress_seq, match_info))
                if plate and len(plate) >= 6:
                    if plate not in plate_conflicts:
                        plate_conflicts[plate] = []
                    plate_conflicts[plate].append((ingress_seq, match_info))

            for tracked_id, entries in track_id_conflicts.items():
                if len(entries) > 1:
                    entries.sort(key=lambda x: x[1].get('last_seen_frame', 0), reverse=True)
                    keep_seq, keep_info = entries[0]
                    for ingress_seq, match_info in entries[1:]:
                        if ingress_seq in matched_vehicles_with_plates:
                            del matched_vehicles_with_plates[ingress_seq]
                            r.improper_park_timers.pop(f"outside_{ingress_seq}", None)

            for plate, entries in plate_conflicts.items():
                if len(entries) > 1:
                    entries.sort(key=lambda x: x[1].get('last_seen_frame', 0), reverse=True)
                    for ingress_seq, match_info in entries[1:]:
                        if ingress_seq in matched_vehicles_with_plates:
                            del matched_vehicles_with_plates[ingress_seq]
                            r.improper_park_timers.pop(f"outside_{ingress_seq}", None)

            # Draw pending UI labels
            drawn_track_ids = set()
            for ingress_seq, match_info in matched_vehicles_with_plates.items():
                plate_text = match_info.get('plate')
                if plate_text:
                    continue
                tracked_id = match_info.get('track_id')
                if tracked_id is None or tracked_id not in track_id_to_vehicle:
                    continue
                _v = track_id_to_vehicle[tracked_id]
                if _v.get('bbox') is None:
                    continue
                x1, y1, x2, y2 = map(int, _v['bbox'])
                label = "Nhan dien bien so..."
                bg_color = (100, 100, 100); txt_color = (255, 255, 255)
                font_scale = 0.6 * ANNOTATION_SCALE_FACTOR
                thickness = max(1, int(2 * ANNOTATION_SCALE_FACTOR))
                padding = max(6, int(6 * ANNOTATION_SCALE_FACTOR))
                offset_y = max(10, int(10 * ANNOTATION_SCALE_FACTOR))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                center_x = (x1 + x2) // 2
                label_x1 = center_x - tw // 2 - padding
                label_y1 = max(0, y1 - th - offset_y - padding // 2)
                cv2.rectangle(frame, (label_x1, label_y1), (label_x1 + tw + padding * 2, y1 - padding), bg_color, -1)
                cv2.putText(frame, label, (label_x1 + padding, y1 - offset_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thickness)
                drawn_track_ids.add(tracked_id)

            # Pending parking_trigger_queue labels
            pending_draw_triggers = []
            with parking_trigger_lock:
                pending_draw_triggers = list(parking_trigger_queue)
            now_dt = datetime.now()
            for trig in pending_draw_triggers:
                _tid = trig.get('track_id')
                if _tid is None or _tid in drawn_track_ids:
                    continue
                _ts = trig.get('ts')
                if _ts is None:
                    continue
                _age_sec = (now_dt - _ts).total_seconds()
                if _age_sec > HANDOFF_QUEUE_ITEM_MAX_AGE_SEC:
                    continue
                if _tid not in track_id_to_vehicle:
                    continue
                _v = track_id_to_vehicle[_tid]
                if _v.get('bbox') is None:
                    continue
                x1, y1, x2, y2 = map(int, _v['bbox'])
                label = "Nhan dien bien so..."
                bg_color = (100, 100, 100); txt_color = (255, 255, 255)
                font_scale = 0.6 * ANNOTATION_SCALE_FACTOR
                thickness = max(1, int(2 * ANNOTATION_SCALE_FACTOR))
                padding = max(6, int(6 * ANNOTATION_SCALE_FACTOR))
                offset_y = max(10, int(10 * ANNOTATION_SCALE_FACTOR))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                center_x = (x1 + x2) // 2
                label_x1 = center_x - tw // 2 - padding
                label_y1 = max(0, y1 - th - offset_y - padding // 2)
                cv2.rectangle(frame, (label_x1, label_y1), (label_x1 + tw + padding * 2, y1 - padding), bg_color, -1)
                cv2.putText(frame, label, (label_x1 + padding, y1 - offset_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thickness)

            # ── Outside-slot tracker ─────────────────────────────────────────
            _fifo_outside_tids = {
                info.get('track_id') for info in matched_vehicles_with_plates.values()
                if info.get('parked_outside') and info.get('track_id') is not None
            }
            # sweep: cancel generic track_* timers for vehicles now owned by FIFO outside path
            for _tid in _fifo_outside_tids:
                _track_key = f"track_{_tid}"
                if _track_key in r.improper_park_timers and not r.improper_park_timers[_track_key].get('logged'):
                    r.improper_park_timers.pop(_track_key, None)
            _track_id_to_plate = {
                info.get('track_id'): info.get('plate')
                for info in matched_vehicles_with_plates.values()
                if info.get('track_id') is not None
            }
            _current_outside_tids = set()

            for _ov in all_vehicles_this_frame:
                _ovclass = (_ov.get('class_name') or _ov.get('det_class') or '').lower()
                if _ovclass not in ('car', 'bus', 'truck'):
                    continue
                _otid = _ov.get('track_id')
                if _otid is None or _otid in _fifo_outside_tids:
                    continue
                _obbox = _ov['bbox']
                _bbox_area = max(0, _obbox[2] - _obbox[0]) * max(0, _obbox[3] - _obbox[1])
                if _bbox_area < IMPROPER_PARK_MIN_BBOX_AREA:
                    continue
                _ovcx, _ovcy = float(_ov['cx']), float(_ov['cy'])
                _in_any_slot = any(
                    cv2.pointPolygonTest(_ap, (_ovcx, _ovcy), False) >= 0
                    for _ap in area_arrays
                )
                if _in_any_slot:
                    r.outside_track_history.pop(_otid, None)
                    r.improper_park_timers.pop(f"track_{_otid}", None)
                    continue

                _current_outside_tids.add(_otid)
                _oh = r.outside_track_history.setdefault(_otid, [])
                _oh.append((_ovcx, _ovcy, frame_counter))
                if len(_oh) > 30:
                    del _oh[:-30]

                _ospeed = 999.0
                if len(_oh) >= 5:
                    _ox0, _oy0, _of0 = _oh[-5]; _ox1, _oy1, _of1 = _oh[-1]
                    _odf = max(1, _of1 - _of0)
                    _ospeed = ((_ox1 - _ox0) ** 2 + (_oy1 - _oy0) ** 2) ** 0.5 / _odf

                _tip_key = f"track_{_otid}"
                if _ospeed >= 2.0:
                    if _tip_key in r.improper_park_timers and not r.improper_park_timers[_tip_key].get('logged'):
                        r.improper_park_timers.pop(_tip_key, None)
                elif _ospeed < 0.8:
                    if _tip_key not in r.improper_park_timers:
                        _init_cx = (_obbox[0] + _obbox[2]) / 2.0
                        _init_cy = (_obbox[1] + _obbox[3]) / 2.0
                        _plate_candidate = _track_id_to_plate.get(_otid)
                        r.improper_park_timers[_tip_key] = {
                            'type': 'outside', 'first_detected': datetime.now(),
                            'logged': False, 'plate': _plate_candidate,
                            'last_bbox': list(_obbox),
                            'init_center': (_init_cx, _init_cy), 'image_path': None,
                        }

            for _gone_tid in list(r.outside_track_history.keys()):
                if _gone_tid not in _current_outside_tids and _gone_tid not in _fifo_outside_tids:
                    r.improper_park_timers.pop(f"track_{_gone_tid}", None)
                    del r.outside_track_history[_gone_tid]

            # ── Overlapping tracker ─────────────────────────────────────────
            _active_overlap_keys = set()
            for _ov in overlapping_vehicles:
                _ov_bbox = _ov.get('bbox')
                if not _ov_bbox:
                    continue
                _bx1, _by1, _bx2, _by2 = _ov_bbox
                _ovcx = (_bx1 + _bx2) / 2.0; _ovcy = (_by1 + _by2) / 2.0
                _cx_b = round(_bx1 / 60); _cy_b = round(_by1 / 60)
                _ip_key = f"overlap_{_ov['area']}_{_cx_b}_{_cy_b}"
                _active_overlap_keys.add(_ip_key)

                _ovh = r.overlap_track_history.setdefault(_ip_key, [])
                _ovh.append((_ovcx, _ovcy, frame_counter))
                if len(_ovh) > 30:
                    del _ovh[:-30]

                _ovspeed = 999.0
                if len(_ovh) >= 5:
                    _ovx0, _ovy0, _ovf0 = _ovh[-5]; _ovx1, _ovy1, _ovf1 = _ovh[-1]
                    _ovdf = max(1, _ovf1 - _ovf0)
                    _ovspeed = ((_ovx1 - _ovx0) ** 2 + (_ovy1 - _ovy0) ** 2) ** 0.5 / _ovdf

                if _ovspeed < 0.8:
                    if _ip_key not in r.improper_park_timers:
                        _init_cx = (_bx1 + _bx2) / 2.0; _init_cy = (_by1 + _by2) / 2.0
                        r.improper_park_timers[_ip_key] = {
                            'type': 'overlapping', 'first_detected': datetime.now(),
                            'logged': False, 'plate': None,
                            'last_bbox': [_bx1, _by1, _bx2, _by2],
                            'init_center': (_init_cx, _init_cy), 'image_path': None,
                        }

            for _stale_key in list(r.overlap_track_history.keys()):
                if _stale_key not in _active_overlap_keys:
                    del r.overlap_track_history[_stale_key]
                    r.improper_park_timers.pop(_stale_key, None)

            # ── Fire improper parking timers ────────────────────────────────
            _now_dt = datetime.now()
            for _ip_key, _ip_info in list(r.improper_park_timers.items()):
                if _ip_info.get('logged'):
                    continue
                _elapsed = (_now_dt - _ip_info['first_detected']).total_seconds()
                if _elapsed < IMPROPER_PARK_LOG_DELAY_SECONDS:
                    continue
                _cur_bb = _ip_info.get('last_bbox', [0, 0, 0, 0])
                _cur_cx = (_cur_bb[0] + _cur_bb[2]) / 2.0
                _cur_cy = (_cur_bb[1] + _cur_bb[3]) / 2.0
                _init = _ip_info.get('init_center', (_cur_cx, _cur_cy))
                _drift = ((_cur_cx - _init[0]) ** 2 + (_cur_cy - _init[1]) ** 2) ** 0.5
                if _drift > IMPROPER_PARK_MAX_MOVE_PX:
                    r.improper_park_timers.pop(_ip_key, None)
                    continue
                _fire_img = _capture_improper_crop(
                    raw_frame, _cur_bb, PARKING_CAPTURE_DIR,
                    _ip_info['type'], _ip_info['plate'], frame_counter
                )
                slot_num = None
                if _ip_key.startswith('overlap_'):
                    parts = _ip_key.split('_')
                    if len(parts) >= 2:
                        try:
                            slot_num = int(parts[1]) + 1
                        except ValueError:
                            pass
                runtime_io.submit_low(
                    log_improper_parking,
                    _ip_info['plate'], _ip_info['type'],
                    image_path=_fire_img, slot_number=slot_num,
                    coalesce_group="improper_parking",
                    coalesce_key=_ip_key,   # "outside_{ingress_seq}" | "track_{otid}" | "overlap_{area}_{bx}_{by}"
                )
                _ip_info['logged'] = True

            # ── Slot occupancy summary ─────────────────────────────────────
            occupied_spaces = sum(1 for s in parking_space_status if s > 0)
            available_spaces = total_spaces - occupied_spaces

            slots_info = []
            for i, status in enumerate(parking_space_status):
                slot_num = i + 1
                matched_info = r.slot_matched_plates.get(slot_num, {})
                slots_info.append({
                    'slot_number': slot_num,
                    'status': 'occupied' if status > 0 else 'available',
                    'vehicle_count': status,
                    'plate': matched_info.get('plate'),
                    'ticket_id': matched_info.get('ticket_id'),
                })

            runtime_io.submit_low(
                check_and_notify_slot_hijacked, slots_info,
                coalesce_group="slot_hijacked_check", coalesce_key="latest",
            )

            r.slots_info = slots_info
            r.occupied_spaces = occupied_spaces
            r.available_spaces = available_spaces
            r.render_seq += 1

            # ── Slot overlay (cached) ────────────────────────────────────────
            r.slot_layer, r.slot_layer_mask, r.last_slot_status_draw = draw_slot_overlay_cached(
                frame, parking_space_status,
                area_arrays, inner_zone_arrays,
                [_a[0] for _a in areas],
                r.slot_layer, r.slot_layer_mask, r.last_slot_status_draw,
                ANNOTATION_SCALE_FACTOR,
            )

        else:
            # Non-tracking mode: just slot occupancy
            r.slots_info = []
            for i, status in enumerate(parking_space_status):
                r.slots_info.append({
                    'slot_number': i + 1,
                    'status': 'occupied' if status > 0 else 'available',
                    'vehicle_count': status,
                    'plate': None, 'ticket_id': None,
                })
            occupied_spaces = sum(1 for s in parking_space_status if s > 0)
            r.occupied_spaces = occupied_spaces
            r.available_spaces = total_spaces - occupied_spaces
            r.render_seq += 1

            r.slot_layer, r.slot_layer_mask, r.last_slot_status_draw = draw_slot_overlay_cached(
                frame, parking_space_status,
                area_arrays, inner_zone_arrays,
                [_a[0] for _a in areas],
                r.slot_layer, r.slot_layer_mask, r.last_slot_status_draw,
                ANNOTATION_SCALE_FACTOR,
            )

        # ── Push to render queue ───────────────────────────────────────────
        render_job = {
            'frame': frame,
            'raw_frame': raw_frame,
            'total_spaces': total_spaces,
            'frame_counter': frame_counter,
        }
        try:
            render_queue.put_nowait(render_job)
        except Exception:
            try:
                render_queue.get_nowait()
                render_queue.put_nowait(render_job)
            except Exception:
                pass

        # ── Adaptive sleep + metrics ────────────────────────────────────────
        elapsed = time.time() - frame_start_time
        with plate_fifo_lock:
            _pf_depth = len(plate_fifo_queue)
        with parking_trigger_lock:
            _pt_depth = len(parking_trigger_queue)
        runtime.mark_loop(
            elapsed * 1000.0, _pf_depth, _pt_depth, len(matched_vehicles_with_plates)
        )
        time.sleep(max(0.0, TARGET_FPS_INTERVAL - elapsed))

    # Cleanup
    if rtsp_cap:
        rtsp_cap.release()
    if cap:
        cap.release()
    print("[PARKING] Detect+Track Worker stopped")


# ==============================================================================
# Thread 2: Render Worker
# ==============================================================================

def render_worker(
    render_queue,
    refs: SharedRefs,
    socketio,
    runtime: ParkingPipelineRuntime,
    stop_event: threading.Event,
):
    """
    Consumes annotated frames from the render_queue, encodes JPEG,
    and writes to shared_state.parking_latest_jpeg.
    Also handles SocketIO status emissions.
    """
    print("[PARKING] Render Worker started")

    r = refs

    while not stop_event.is_set():
        try:
            job = render_queue.get(timeout=0.5)
        except Empty:
            continue

        frame = job.get('frame')
        if frame is None:
            continue

        raw_frame = job.get('raw_frame')
        total_spaces = job.get('total_spaces', 0)
        frame_counter = job.get('frame_counter', 0)

        # Slot overlay is already baked into frame by detect_track_worker
        # (via draw_slot_overlay_cached modifying in-place)

        # Upscale for display
        display_frame = frame
        if PARKING_UPSCALE_DISPLAY:
            current_height, current_width = frame.shape[:2]
            if current_width < PARKING_UPSCALE_WIDTH or current_height < PARKING_UPSCALE_HEIGHT:
                display_frame = cv2.resize(
                    frame, (PARKING_UPSCALE_WIDTH, PARKING_UPSCALE_HEIGHT),
                    interpolation=cv2.INTER_CUBIC
                )

        # JPEG encode
        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        with parking_jpeg_lock:
            parking_latest_jpeg = frame_bytes
            shared_state.parking_latest_jpeg = frame_bytes
            setattr(shared_state, 'parking_render_seq', r.render_seq)
            setattr(shared_state, 'parking_render_ts', time.time())

        # Update global status
        if r.current_parking_status is not None:
            r.current_parking_status.update({
                'total_spaces': total_spaces,
                'occupied_spaces': r.occupied_spaces,
                'available_spaces': r.available_spaces,
                'slots': r.slots_info,
                'outside_vehicles_count': len(r.outside_vehicles),
                'overlapping_vehicles_count': len(r.overlapping_vehicles),
                'last_update': datetime.now().isoformat(),
                'render_seq': r.render_seq,
            })

        # SocketIO emit (throttled)
        if socketio is not None:
            _now = time.time()
            _status_key = (r.occupied_spaces, len(r.outside_vehicles), len(r.overlapping_vehicles))
            if _now - r.last_emit_time >= 0.33 or _status_key != r.last_emit_status:
                _payload = {
                    'total_spaces': total_spaces,
                    'occupied_spaces': r.occupied_spaces,
                    'available_spaces': r.available_spaces,
                    'outside_vehicles_count': len(r.outside_vehicles),
                    'overlapping_vehicles_count': len(r.overlapping_vehicles),
                    'render_seq': r.render_seq,
                    'timestamp': datetime.now().isoformat(),
                }
                socketio.start_background_task(socketio.emit, 'update_status', _payload)
                runtime.mark_emit()
                r.last_emit_time = _now
                r.last_emit_status = _status_key

    print("[PARKING] Render Worker stopped")
