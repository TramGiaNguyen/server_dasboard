# Global state management for Smart Parking System
from datetime import datetime
import collections
import json
import os
import threading
import time

# ========== CAMERA STARTUP SYNCHRONIZATION ==========
# Ensures both cameras connect and start processing at the same time
# Each camera signals "ready" after connecting, then waits for both to be ready
# After both ready, a warmup buffer runs before processing begins

CAMERA_WARMUP_SECONDS = 5  # Seconds to buffer after both cameras connect

_gate_ready = threading.Event()
_parking_ready = threading.Event()
_cameras_synced = threading.Event()  # Set when both cameras are synced and ready to go
_sync_lock = threading.Lock()

def signal_camera_ready(camera_name: str):
    """Signal that a camera has connected and is ready to process."""
    if camera_name == 'gate':
        _gate_ready.set()
        print(f"[SYNC] Gate camera READY, waiting for parking camera...")
    elif camera_name == 'parking':
        _parking_ready.set()
        print(f"[SYNC] Parking camera READY, waiting for gate camera...")
    
    # Check if both are ready
    if _gate_ready.is_set() and _parking_ready.is_set():
        with _sync_lock:
            if not _cameras_synced.is_set():
                print(f"[SYNC] Both cameras connected! Starting {CAMERA_WARMUP_SECONDS}s warmup buffer...")
                # Warmup: discard initial frames for stabilization
                time.sleep(CAMERA_WARMUP_SECONDS)
                _cameras_synced.set()
                print(f"[SYNC] ✓ Warmup complete. Both cameras starting NOW!")

def wait_for_camera_sync(timeout=60):
    """Block until both cameras are synced. Returns True if synced, False if timeout."""
    return _cameras_synced.wait(timeout=timeout)

def reset_camera_sync():
    """Reset sync state (used when streams reconnect)."""
    _gate_ready.clear()
    _parking_ready.clear()
    _cameras_synced.clear()
    print("[SYNC] Camera sync state reset")

# ========== MJPEG FRAME BUFFERS ==========
# Background processing threads store the latest JPEG here;
# /video_feed routes relay these to any number of browser clients.
parking_latest_jpeg = None      # raw JPEG bytes from parking camera
parking_jpeg_lock = threading.Lock()

gate_latest_jpeg = None         # raw JPEG bytes from gate camera
gate_jpeg_lock = threading.Lock()

# Global variable to store current parking status
current_parking_status = {
    'total_spaces': 0,
    'occupied_spaces': 0,
    'available_spaces': 0,
    'slots': [],  # List of slot statuses
    'outside_vehicles_count': 0,
    'overlapping_vehicles_count': 0,
    'last_update': None
}



# Global variable to store OCR results
gate_ocr_results = {
    'vehicles': [],
    'last_detection_time': None,
    'latest_plate': None,
    'latest_plate_confidence': 0.0,
    'vehicle_movements': []  # Track vehicle movements (in/out)
}

# Global variable for cross-camera vehicle tracking
vehicle_tracking_state = {
    'pending_count': 0,          # Vehicles waiting to appear in parking
    'matched_count': 0,          # Successfully matched vehicles
    'expired_count': 0,          # Vehicles that timed out
    'average_transit_time': 0.0, # Average time from gate to parking
    'recent_tickets': [],        # Last 10 tickets issued
    'recent_matches': [],        # Last 10 matches
    'last_update': None
}

# ========== VEHICLE ANCHORED HANDOFF SYSTEM ==========
# 1. Gate camera detects vehicle crossing line -> pushes entry record here (plate may be None)
# 2. Parking camera sees vehicle -> grabs oldest entry record and attaches it to a "Line"
# 3. Gate camera finishes OCR -> updates the same entry record with plate text
# 4. Parking camera sees plate update in its attached record and displays it
plate_handoff_queue = []
plate_handoff_lock = threading.Lock()


# ========== PLATE FIFO QUEUE ==========
# Gate assigns monotonic ingress_seq per vehicle entering (direction in). OCR may fill plate later.
# Parking pairs FIFO head with parking_trigger_queue head when both exist (order match).
plate_fifo_queue = []
plate_fifo_lock = threading.Lock()

_ingress_seq_lock = threading.Lock()
next_ingress_seq = 0


def allocate_ingress_seq() -> int:
    """Monotonic id for each gate ingress event (thread-safe)."""
    global next_ingress_seq
    with _ingress_seq_lock:
        next_ingress_seq += 1
        return next_ingress_seq


# Parking: first bbox crossing entry trigger per track (FIFO), paired with plate_fifo head.
parking_trigger_queue = []
parking_trigger_lock = threading.Lock()

# Pair plate_fifo entry with parking trigger (optional: single lock for atomic pair)
handoff_match_lock = threading.Lock()

# Max age (seconds) for orphan items in FIFO / trigger queue before drop + log
HANDOFF_QUEUE_ITEM_MAX_AGE_SEC = 180.0


def append_plate_fifo(entry: dict):
    """Thread-safe append to plate FIFO queue."""
    with plate_fifo_lock:
        plate_fifo_queue.append(entry)


def update_plate_fifo_entry(gate_track_id, new_plate, new_conf):
    """Thread-safe update of a plate FIFO entry.

    Trước đây hàm chỉ update khi entry chưa được `assigned`.
    Với luồng FIFO gate->parking, parking side có thể mark `assigned=True`
    để ngăn consume lại entry đó (tránh "gắn nhầm biển xe trước").
    Tuy nhiên vẫn cần OCR backfill cập nhật plate/conf cho entry đã được
    reserve (`reserved_ingress_seq`) để parking side nhận trạng thái pending.
    """
    updated = False
    reserved_seqs_touched = []
    with plate_fifo_lock:
        for p in plate_fifo_queue:
            if p.get('gate_track_id') != gate_track_id:
                continue

            # Update nếu entry chưa assigned, hoặc đã assigned nhưng đã được reserve
            # cho một ingress_seq ở phía parking side.
            if (not p.get('assigned')) or (p.get('reserved_ingress_seq') is not None):
                p['plate'] = new_plate
                p['conf'] = new_conf
                updated = True
                _rs = p.get("reserved_ingress_seq")
                if _rs is not None:
                    reserved_seqs_touched.append(int(_rs))
                    # Sync plate từ FIFO → matched_vehicles_with_plates (UI overlay)
                    with matched_vehicles_lock:
                        _ui = matched_vehicles_with_plates.get(_rs)
                        if _ui is not None and not _ui.get("plate"):
                            _ui["plate"] = new_plate
                            _ui["conf"] = float(new_conf or 0.0)
                            _ui["plate_status"] = (
                                "pending" if not new_plate
                                else ("confirmed" if float(new_conf or 0.0) >= 0.80 else "provisional")
                            )

    return updated


def pop_oldest_unassigned_plate():
    """Thread-safe: return oldest unassigned plate entry or None."""
    with plate_fifo_lock:
        for entry in plate_fifo_queue:
            if not entry.get('assigned'):
                return entry
    return None


def remove_plate_fifo(entry: dict):
    """Thread-safe removal from plate FIFO queue."""
    with plate_fifo_lock:
        try:
            plate_fifo_queue.remove(entry)
        except ValueError:
            pass

# Deprecated — kept empty for backward compatibility during refactor (do not use)
entry_zone_pending_vehicles = {}
entry_zone_next_line_number = 1
entry_zone_vehicle_lines = {}
entry_zone_lock = threading.Lock()


# ========== MATCHED VEHICLES WITH PLATES ==========
matched_vehicles_with_plates = {}
matched_vehicles_lock = threading.Lock()

# ========== GATE CAMERA PIPELINE QUEUES ==========
# Multi-threaded pipeline: Detect+Track Worker → OCR Worker / Render Worker
#
#  Fair OCR scheduler (replaces dict+popitem LIFO):
#    gate_ocr_latest_jobs[track_id] = latest crop job (overwrite each frame).
#    gate_ocr_pending_queue: deque of track_id (FIFO); gate_ocr_pending_enqueued avoids duplicates.
#    gate_ocr_track_db_ctx[track_id]: DB backfill after handoff removed — gate_log_id, session_id, direction, created_ts.
#
#  gate_render_queue: Detect+Track Worker pushes annotated-frame info here.
#    Render Worker consumes and writes to gate_latest_jpeg.
#    Item: {'frame': np.ndarray, 'frame_count': int, 'crossing_events': list}
#    maxsize=2: only latest frame matters for MJPEG; older frames are discarded.

GATE_OCR_BACKFILL_TTL_SEC = 300.0
GATE_OCR_PROVISIONAL_CONF = 0.60  # was 0.85 — lowered to capture more single-frame results

gate_ocr_scheduler_lock = threading.Lock()
gate_ocr_latest_jobs = {}
gate_ocr_pending_queue = collections.deque()
gate_ocr_pending_enqueued = set()
gate_ocr_track_db_ctx = {}
gate_ocr_artifacts = {}

# Legacy: kept empty; detect now uses gate_ocr_enqueue_job
gate_ocr_mailbox = {}
gate_ocr_condition = threading.Condition()


def gate_ocr_enqueue_job(track_id, crop_job: dict):
    """Enqueue latest crop for track_id (FIFO fair scheduling)."""
    if track_id is None:
        return
    with gate_ocr_scheduler_lock:
        gate_ocr_latest_jobs[track_id] = crop_job
        if track_id not in gate_ocr_pending_enqueued:
            gate_ocr_pending_queue.append(track_id)
            gate_ocr_pending_enqueued.add(track_id)
    with gate_ocr_condition:
        gate_ocr_condition.notify()


def gate_ocr_merge_ctx_from_handoff(track_id, handoff_record: dict, gate_vehicle_handoffs: dict = None):
    """Persist gate_log_id from handoff for OCR backfill after track removed."""
    if not track_id or not handoff_record:
        return
    # do NOT overwrite with stale ctx if an active handoff already owns this track_id
    if gate_vehicle_handoffs is not None and track_id in gate_vehicle_handoffs:
        return
    rs = handoff_record.get('result_store') or {}
    # IN: log_gate_entry writes gate_log_id onto the handoff dict passed as result_store.
    # OUT: gate_log_id may live only in nested result_store.
    gid = handoff_record.get('gate_log_id') or rs.get('gate_log_id')
    sid = handoff_record.get('session_id') or rs.get('session_id')
    if not gid:
        return
    with gate_ocr_scheduler_lock:
        gate_ocr_track_db_ctx[track_id] = {
            'gate_log_id': gid,
            'session_id': sid,
            'direction': handoff_record.get('direction', 'in'),
            'created_ts': time.time(),
        }


def gate_ocr_persist_ctx_before_handoff_drop(vehicle_id, handoff_record: dict, gate_vehicle_handoffs: dict = None):
    """Call before removing handoff so OCR can still update GateLog."""
    gate_ocr_merge_ctx_from_handoff(vehicle_id, handoff_record, gate_vehicle_handoffs)


def gate_ocr_prune_stale_ctx_and_jobs(now=None):
    """Remove expired backfill ctx and latest job stubs."""
    now = now or time.time()
    with gate_ocr_scheduler_lock:
        dead = [
            tid
            for tid, ctx in list(gate_ocr_track_db_ctx.items())
            if now - float(ctx.get('created_ts', 0)) > GATE_OCR_BACKFILL_TTL_SEC
        ]
        for tid in dead:
            gate_ocr_track_db_ctx.pop(tid, None)
            gate_ocr_latest_jobs.pop(tid, None)
            gate_ocr_artifacts.pop(tid, None)


def gate_ocr_scheduler_depth():
    with gate_ocr_scheduler_lock:
        return len(gate_ocr_pending_queue), len(gate_ocr_latest_jobs)

from queue import Queue as _Queue
gate_render_queue:   _Queue = _Queue(maxsize=2)

# ========== VIDEO SYNCHRONIZATION ==========
# Sync gate camera and parking camera video streams
# Both cameras will check sync state before processing each frame

video_sync_state = {
    'gate_frame': 0,        # Current frame number of gate camera
    'parking_frame': 0,     # Current frame number of parking camera
    'max_frame_diff': 5,    # Allow max 5 frames difference
    'enabled': False,       # DISABLED: Frame sync causes slowdown. FIFO plate queue handles matching.
}

# Lock for thread-safe frame counter updates
video_sync_lock = threading.Lock()

def update_gate_frame(frame_num):
    """Update gate camera frame counter"""
    with video_sync_lock:
        video_sync_state['gate_frame'] = frame_num

def update_parking_frame(frame_num):
    """Update parking camera frame counter"""
    with video_sync_lock:
        video_sync_state['parking_frame'] = frame_num

def should_gate_wait():
    """Check if gate camera should wait for parking camera to catch up"""
    if not video_sync_state['enabled']:
        return False
    with video_sync_lock:
        diff = video_sync_state['gate_frame'] - video_sync_state['parking_frame']
        return diff > video_sync_state['max_frame_diff']

def should_parking_wait():
    """Check if parking camera should wait for gate camera to catch up"""
    if not video_sync_state['enabled']:
        return False
    with video_sync_lock:
        diff = video_sync_state['parking_frame'] - video_sync_state['gate_frame']
        return diff > video_sync_state['max_frame_diff']
