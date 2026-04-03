"""Gate camera worker threads: OCR, render, detect+track."""
import cv2
import os
import sys
import time
import threading
from datetime import datetime
from queue import Empty

import numpy as np

_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

_ocr_path = os.path.join(_base_dir, "services", "ocr", "LicensePlate_OCR_Standalone")
if _ocr_path not in sys.path:
    sys.path.insert(0, _ocr_path)

try:
    from plate_detector import VehicleInfo  # type: ignore
    from ocr_utils import enhanced_plate_preprocessing  # type: ignore
except ImportError:
    VehicleInfo = None  # type: ignore

    def enhanced_plate_preprocessing(img, scale=6):
        if img is None:
            return img
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

from shared.state import (
    vehicle_tracking_state,
    plate_handoff_queue,
    plate_fifo_queue,
    plate_fifo_lock,
    allocate_ingress_seq,
    update_plate_fifo_entry,
    gate_ocr_condition,
    gate_render_queue,
    gate_ocr_enqueue_job,
    gate_ocr_merge_ctx_from_handoff,
    gate_ocr_persist_ctx_before_handoff_drop,
    gate_ocr_prune_stale_ctx_and_jobs,
    gate_ocr_scheduler_depth,
    GATE_OCR_PROVISIONAL_CONF,
)
import shared.state as shared_state

from config import TRACKING_ENABLED
from shared.models import initialize_model
from services.vehicle_tracking.tracker import get_tracker
from database.operations import log_gate_entry, log_gate_exit, update_gate_exit_plate, update_gate_entry_media
from services.vehicle_tracking.models import VehicleTicket
from shared.rtsp_capture import RTSPCapture


def _upscale_plate_for_cache(plate_img: 'np.ndarray') -> 'np.ndarray':
    """Apply same preprocessing as OCR pipeline: Upscale + Wiener + CLAHE + Sharpen."""
    try:
        import numpy as np
        if plate_img is None or plate_img.size == 0:
            return plate_img
        h = plate_img.shape[0]
        if h < 40:
            scale = 10
        elif h < 80:
            scale = 8
        else:
            scale = 4
        return enhanced_plate_preprocessing(plate_img.copy(), scale=scale)
    except Exception:
        return plate_img  # fallback to raw crop if preprocessing fails


def _save_ocr_plate_image(gate_capture_dir: str, track_id: str, plate_text: str, plate_img: 'np.ndarray'):
    """Persist OCR plate crop and return public image path."""
    if not gate_capture_dir or plate_img is None or plate_img.size == 0:
        return None
    try:
        plate_tag = (plate_text or "noplate").strip() or "noplate"
        plate_tag = "".join(ch for ch in plate_tag if ch.isalnum() or ch in ("-", "_"))[:24]
        ts = int(time.time() * 1000)
        # reject geometrically implausible crops (vertical stripes / horizontally-compressed)
        h, w = plate_img.shape[:2]
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        min_aspect = 1.0
        max_aspect = 10.0
        min_width_px = 24
        if w >= min_width_px and min_aspect <= aspect_ratio <= max_aspect:
            fname = f"gate_{track_id}_{plate_tag}_{ts}.jpg"
            fpath = os.path.join(gate_capture_dir, fname)
            cv2.imwrite(fpath, plate_img)
            return f"/static/gate_captures/{fname}"
        print(f"[GATE OCR] Skipping implausible plate crop "
              f"{w}x{h} ar={aspect_ratio:.2f} track={track_id}")
        return None
    except Exception as img_exc:
        print(f"[GATE OCR] Failed to save OCR artifact image for {track_id}: {img_exc}")
        return None


def _ctc_canonical(plate_text: str, existing_plates) -> str:
    """
    CTC decoding collapses consecutive identical tokens that have no blank
    between them. Redirects votes to longer canonical form when a plate is
    a CTC-collapsed subset of an existing recognised plate.
    """
    for existing in existing_plates:
        if len(existing) != len(plate_text) + 1:
            continue
        for i in range(len(existing)):
            if existing[:i] + existing[i + 1:] == plate_text:
                is_left_dup  = i > 0 and existing[i] == existing[i - 1]
                is_right_dup = i < len(existing) - 1 and existing[i] == existing[i + 1]
                if is_left_dup or is_right_dup:
                    return existing
    return plate_text


# ---------------------------------------------------------------------------
# OCR Worker Thread
# ---------------------------------------------------------------------------

def ocr_worker(detector, plate_votes_by_track, best_plate_by_track,
                track_plate_images, stable_plate_cache, best_plate_img_cache,
                gate_vehicle_handoffs, _upsert_fn, stop_event,
                submit_low=None, gate_capture_dir=None):
    """
    Fair FIFO queue of track_ids; latest crop per track in gate_ocr_latest_jobs.
    Backfill DB via gate_ocr_track_db_ctx when handoff already removed.
    """
    print("[GATE] OCR Worker started")
    empty_media_saved = set()

    while not stop_event.is_set():
        # Clean up memory occasionally
        if len(empty_media_saved) > 50:
            stale_tids = [tid for tid in empty_media_saved if tid not in shared_state.gate_ocr_track_db_ctx]
            for tid in stale_tids:
                empty_media_saved.discard(tid)

        with gate_ocr_condition:
            gate_ocr_condition.wait_for(
                lambda: len(shared_state.gate_ocr_pending_queue) > 0 or stop_event.is_set(),
                timeout=0.5,
            )
            if stop_event.is_set():
                break
            if not shared_state.gate_ocr_pending_queue:
                continue
            with shared_state.gate_ocr_scheduler_lock:
                if not shared_state.gate_ocr_pending_queue:
                    continue
                track_id = shared_state.gate_ocr_pending_queue.popleft()
                shared_state.gate_ocr_pending_enqueued.discard(track_id)
                job = shared_state.gate_ocr_latest_jobs.get(track_id)
        if job is None:
            continue

        crop_frame = job.get('crop_frame')
        frame_count = job.get('frame_count', 0)

        if crop_frame is None or track_id is None:
            continue

        _has_db_ctx = track_id in shared_state.gate_ocr_track_db_ctx
        if (
            track_id not in gate_vehicle_handoffs
            and track_id not in stable_plate_cache
            and not _has_db_ctx
        ):
            continue

        try:
            v = VehicleInfo()
            v.vehicle_image = crop_frame
            v.track_id = track_id
            v.bbox = job.get('bbox')

            detector.detect_plates([v])
            detector.recognize_plates([v])

            ocr_image_path = None
            ocr_ts = int(time.time() * 1000)

            # Phase 5: provisional single-frame high confidence
            if v.plate_text and len(v.plate_text.strip()) >= 3 and float(v.plate_conf or 0) >= GATE_OCR_PROVISIONAL_CONF:
                pt = v.plate_text.strip()
                prev = best_plate_by_track.get(track_id)
                if prev is None or float(v.plate_conf) > float(prev.get('conf', 0)):
                    if v.plate_image is not None and v.plate_image.shape[0] >= 10:
                        ocr_image_path = _save_ocr_plate_image(
                            gate_capture_dir, track_id, pt, v.plate_image
                        )
                    best_plate_by_track[track_id] = {'plate': pt, 'conf': float(v.plate_conf)}
                    if ocr_image_path:
                        with shared_state.gate_ocr_scheduler_lock:
                            shared_state.gate_ocr_artifacts[track_id] = {
                                'plate': pt,
                                'conf': float(v.plate_conf),
                                'image_path': ocr_image_path,
                                'ocr_ts': ocr_ts,
                            }
                    if track_id in stable_plate_cache or track_id in gate_vehicle_handoffs or _has_db_ctx:
                        if track_id not in stable_plate_cache:
                            stable_plate_cache[track_id] = {'cx': 0, 'cy': 0}
                        stable_plate_cache[track_id]['plate'] = pt
                        stable_plate_cache[track_id]['conf'] = float(v.plate_conf)
                        if ocr_image_path:
                            _upsert_fn(track_id, pt, float(v.plate_conf), image_path=ocr_image_path, ocr_ts=ocr_ts)
                        update_plate_fifo_entry(track_id, pt, float(v.plate_conf))

            if not v.plate_text:
                if v.plate_image is not None and v.plate_image.shape[0] >= 8:
                    if track_id not in track_plate_images:
                        track_plate_images[track_id] = _upscale_plate_for_cache(v.plate_image)
                    if submit_low and gate_capture_dir and _has_db_ctx and track_id not in empty_media_saved:
                        ctx = shared_state.gate_ocr_track_db_ctx.get(track_id) or {}
                        gid = ctx.get('gate_log_id')
                        sid = ctx.get('session_id')
                        if gid and track_id in track_plate_images:
                            empty_media_saved.add(track_id)
                            try:
                                _img = track_plate_images[track_id]
                                _fname = f"gate_{track_id}_crop_{int(time.time() * 1000)}.jpg"
                                _p = os.path.join(gate_capture_dir, _fname)
                                cv2.imwrite(_p, _img)
                                _url = f"/static/gate_captures/{_fname}"
                                submit_low(
                                    update_gate_entry_media,
                                    gid,
                                    sid,
                                    _url,
                                    coalesce_group="gate_entry_media",
                                    coalesce_key=str(gid),
                                )
                            except Exception as _me:
                                print(f"[GATE OCR] Media-only update failed: {_me}")
                continue

            if v.plate_image is not None and v.plate_image.shape[0] < 10:
                print(f"[GATE OCR] Skipping very small plate crop ({v.plate_image.shape[0]}px): {v.plate_text}")
                continue

            if v.plate_image is not None and v.track_id is not None:
                track_plate_images[track_id] = _upscale_plate_for_cache(v.plate_image)

            # --- Per-track majority voting ---
            if len(v.plate_text) < 3:
                continue

            if track_id not in plate_votes_by_track:
                plate_votes_by_track[track_id] = {}
            track_votes = plate_votes_by_track[track_id]
            plate_text_final = _ctc_canonical(v.plate_text, track_votes.keys())

            if plate_text_final not in track_votes:
                track_votes[plate_text_final] = {'votes': 0, 'best_conf': 0.0}
            r = track_votes[plate_text_final]
            r['votes'] += 1
            if v.plate_conf > r['best_conf']:
                r['best_conf'] = v.plate_conf

            if track_votes:
                best_text = max(track_votes, key=lambda t: track_votes[t]['votes'] * track_votes[t]['best_conf'])
                best_conf = track_votes[best_text]['best_conf']
                prev = best_plate_by_track.get(track_id)
                if prev is None or prev['plate'] != best_text or best_conf > prev['conf']:
                    if ocr_image_path is None and v.plate_image is not None and v.plate_image.shape[0] >= 10:
                        ocr_image_path = _save_ocr_plate_image(
                            gate_capture_dir, track_id, best_text, v.plate_image
                        )
                    if prev is None:
                        print(f"[GATE OCR] Track {track_id}: new plate {best_text} (conf={best_conf:.2f}, votes={track_votes[best_text]['votes']})")
                    elif prev['plate'] != best_text:
                        print(f"[GATE OCR] Track {track_id}: plate changed {prev['plate']} → {best_text}")
                    best_plate_by_track[track_id] = {'plate': best_text, 'conf': best_conf}
                    if ocr_image_path:
                        with shared_state.gate_ocr_scheduler_lock:
                            shared_state.gate_ocr_artifacts[track_id] = {
                                'plate': best_text,
                                'conf': best_conf,
                                'image_path': ocr_image_path,
                                'ocr_ts': ocr_ts,
                            }
                    if (
                        track_id in stable_plate_cache
                        or track_id in gate_vehicle_handoffs
                        or track_id in shared_state.gate_ocr_track_db_ctx
                    ):
                        if track_id not in stable_plate_cache:
                            stable_plate_cache[track_id] = {'cx': 0, 'cy': 0}
                        stable_plate_cache[track_id]['plate'] = best_text
                        stable_plate_cache[track_id]['conf'] = best_conf
                        if ocr_image_path:
                            _upsert_fn(track_id, best_text, best_conf, image_path=ocr_image_path, ocr_ts=ocr_ts)
                        update_plate_fifo_entry(track_id, best_text, best_conf)

        except Exception as exc:
            print(f"[GATE OCR] Worker error for track {track_id}: {exc}")

    print("[GATE] OCR Worker stopped")


# ---------------------------------------------------------------------------
# Render Worker Thread
# ---------------------------------------------------------------------------

def render_worker(detector, stable_plate_cache, vehicle_line_crossings_ref,
                   vehicle_tracks_ref, line1_y_ref, line2_y_ref, line3_y_ref,
                   gate_ocr_results_dict, socketio, stop_event, line_thickness, runtime=None):
    """
    Consumes (frame, active_tracks, crossing_events) from gate_render_queue.
    Draws annotations (lines, bboxes, plate labels), encodes JPEG, stores to
    shared_state.gate_latest_jpeg — this is what /video_feed_gate streams.
    """
    print("[GATE] Render Worker started")
    _clahe_render = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    LINE_THICKNESS_RENDER = line_thickness

    while not stop_event.is_set():
        try:
            job = gate_render_queue.get(timeout=0.5)
        except Empty:
            continue

        frame      = job.get('frame')
        if frame is None:
            continue

        active_tracks   = job.get('active_tracks', [])
        crossing_events = job.get('crossing_events', [])
        frame_count     = job.get('frame_count', 0)
        line1_y = job.get('line1_y', line1_y_ref[0])
        line2_y = job.get('line2_y', line2_y_ref[0])
        line3_y = job.get('line3_y', line3_y_ref[0])

        try:
            # --- Draw detection lines ---
            cv2.line(frame, (0, line1_y), (frame.shape[1], line1_y), (0, 0, 255), LINE_THICKNESS_RENDER)
            cv2.line(frame, (0, line2_y), (frame.shape[1], line2_y), (0, 0, 255), LINE_THICKNESS_RENDER)
            cv2.line(frame, (0, line3_y), (frame.shape[1], line3_y), (0, 0, 255), LINE_THICKNESS_RENDER)
            cv2.putText(frame, "1", (10, line1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(frame, "2", (10, line2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(frame, "3", (10, line3_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # --- Draw vehicle bboxes + plate labels from stable_plate_cache ---
            for trk in active_tracks:
                vehicle_id = trk.get('track_id')
                bbox       = trk.get('bbox')
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Draw vehicle box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Plate label (from stable_plate_cache — updated by OCR Worker asynchronously)
                plate_info = stable_plate_cache.get(vehicle_id)
                if plate_info and plate_info.get('plate'):
                    plate_text = plate_info['plate']
                    (tw, th), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    label_x1 = cx - tw // 2 - 5
                    label_y1 = max(0, y1 - th - 14)
                    cv2.rectangle(frame, (label_x1, label_y1),
                                  (label_x1 + tw + 10, label_y1 + th + 10),
                                  (0, 200, 255), -1)
                    cv2.putText(frame, plate_text, (label_x1 + 5, label_y1 + th + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                # Draw center point
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

                # Movement trail
                track_hist = vehicle_tracks_ref.get(vehicle_id, [])
                if len(track_hist) > 1:
                    pts = [(p[0], p[1]) for p in track_hist]
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i-1], pts[i], (255, 0, 255), 2)

                # Direction label
                cross_state = vehicle_line_crossings_ref.get(vehicle_id, {})
                direction = cross_state.get('direction')
                if direction:
                    d_text  = "XE RA" if direction == 'out' else "XE VAO"
                    d_color = (0, 255, 0) if direction == 'out' else (0, 165, 255)
                    cv2.putText(frame, d_text, (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, d_color, 2)

            # --- Encode and store JPEG ---
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = buf.tobytes()
            with shared_state.gate_jpeg_lock:
                shared_state.gate_latest_jpeg = jpeg_bytes

            # --- Emit SocketIO on crossing events ---
            if socketio is not None and crossing_events:
                for evt in crossing_events:
                    _payload = {
                        'plate_count': 1 if evt.get('plate') else 0,
                        'latest_plate': evt.get('plate'),
                        'latest_confidence': evt.get('conf', 0.0),
                        'vehicle_count': len(active_tracks),
                        'timestamp': datetime.now().isoformat(),
                    }
                    socketio.start_background_task(socketio.emit, 'gate_ocr_update', _payload)
                    if runtime is not None:
                        runtime.mark_emit()

        except Exception as exc:
            print(f"[GATE Render] Error: {exc}")

    print("[GATE] Render Worker stopped")


# ---------------------------------------------------------------------------
# Detect + Track Worker Thread  (main processing loop)
# ---------------------------------------------------------------------------

def detect_track_worker(video_url, socketio, gate_ocr_results_dict,
                          process_interval, stop_event,
                          plate_votes_by_track, best_plate_by_track,
                          stable_plate_cache, best_plate_img_cache,
                          track_plate_images, gate_vehicle_handoffs,
                          _upsert_fn, io_submit=None, runtime=None,
                          *, base_dir, gate_capture_dir, get_ocr_detector,
                          coco_file_path, gate_line_1_y, gate_line_2_y, gate_line_3_y,
                          vehicle_line_crossings, vehicle_tracks):
    """
    Thread 1 — Detect+Track Worker.
    Runs YOLO+ByteTrack every frame. Handles line-crossing logic and FIFO push.
    Sends crop jobs to OCR Worker and frames to Render Worker asynchronously.
    """
    print("[GATE] Detect+Track Worker started")

    # Convert video path
    if video_url.startswith('/static/'):
        video_path = os.path.join(base_dir, video_url.lstrip('/'))
    else:
        video_path = video_url

    is_stream = video_path.lower().startswith(('rtsp://', 'http://', 'https://'))

    if is_stream:
        rtsp_cap = RTSPCapture(video_path, buffer_size=3)
        if not rtsp_cap.open():
            print(f"[ERROR] Gate Detect+Track: Failed to open RTSP stream: {video_path}")
            return
        cap = None
        from shared.state import signal_camera_ready, wait_for_camera_sync
        signal_camera_ready('gate')
        print("[GATE] Waiting for camera sync...")
        if not wait_for_camera_sync(timeout=60):
            print("[ERROR] Camera sync timeout! Starting gate camera anyway.")
        rtsp_cap.flush(wait_seconds=1.0)
        print("[GATE] ✓ Sync complete, starting processing.")
    else:
        cap = cv2.VideoCapture(video_path)
        rtsp_cap = None
        if not cap.isOpened():
            print(f"[ERROR] Gate Detect+Track: Failed to open video: {video_path}")
            return

    # Initialize models
    detector = get_ocr_detector()
    gate_vehicle_model = initialize_model()
    with open(coco_file_path, 'r', encoding='utf-8') as f:
        gate_coco_classes = [line.strip() for line in f if line.strip()]
    COCO_CAR_BUS_TRUCK_IDS = [2, 5, 7]  # car, bus, truck

    # Read native resolution & scale LINE_Y values
    if is_stream:
        ret, test_frame = rtsp_cap.read(timeout=5.0)
    else:
        ret, test_frame = cap.read()

    line_1_y = gate_line_1_y
    line_2_y = gate_line_2_y
    line_3_y = gate_line_3_y

    if ret and test_frame is not None:
        frame_height, frame_width = test_frame.shape[:2]
        _h_ratio = frame_height / 500.0
        line_1_y = int(gate_line_1_y * _h_ratio)
        line_2_y = int(gate_line_2_y * _h_ratio)
        line_3_y = int(gate_line_3_y * _h_ratio)
        print(f"[GATE] Native resolution: {frame_width}x{frame_height} "
              f"| LINE_1_Y={line_1_y} LINE_2_Y={line_2_y} LINE_3_Y={line_3_y} (scale={_h_ratio:.2f}x)")
    if not is_stream and cap is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    no_vehicle_frames = 0
    MAX_CONSECUTIVE_ERRORS = 50
    consecutive_errors = 0
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Local crossing state (shared with render via dicts passed from camera module)
    _line_crossings = vehicle_line_crossings
    _tracks_hist    = vehicle_tracks

    while not stop_event.is_set():
        frame_start = time.time()

        # --- Read frame ---
        if is_stream:
            ret, frame = rtsp_cap.read()
        else:
            ret, frame = cap.read()
            if not ret:
                # Loop video file
                plate_votes_by_track.clear()
                best_plate_by_track.clear()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        if not ret or frame is None:
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"[ERROR] Gate camera: {MAX_CONSECUTIVE_ERRORS} consecutive read failures, stopping.")
                break
            continue
        consecutive_errors = 0

        # Video sync (only for file-based testing)
        from shared.state import update_gate_frame, should_gate_wait
        update_gate_frame(frame_count)
        while should_gate_wait():
            time.sleep(0.01)

        frame_count += 1
        raw_frame = frame.copy()  # Clean frame for evidence crops

        # --- Run detection + tracking on the latest frame ---
        crossing_events = []  # Accumulated crossing events for this frame
        active_tracks   = []  # List of {'track_id', 'bbox'} for render

        if True:
            try:
                gate_ocr_prune_stale_ctx_and_jobs()
                for _tid, _hrec in list(gate_vehicle_handoffs.items()):
                    gate_ocr_merge_ctx_from_handoff(_tid, _hrec, gate_vehicle_handoffs)

                # Bypass CLAHE for Gate Camera to maximize FPS and prevent tracking loss
                detection_frame = frame.copy()

                results = gate_vehicle_model.track(
                    detection_frame,
                    classes=COCO_CAR_BUS_TRUCK_IDS,
                    conf=0.15,  # Lower confidence to prevent flickering/track drop
                    iou=0.40,
                    persist=True,
                    tracker='botsort.yaml',  # robust tracking for moving vehicles
                    verbose=False,
                    imgsz=640,
                )
                boxes = results[0].boxes
                all_vehicles_raw = []
                if boxes is not None and len(boxes) > 0:
                    has_ids = boxes.id is not None
                    for idx in range(len(boxes.xyxy)):
                        bbox = boxes.xyxy[idx].cpu().numpy().astype(int)
                        v_tmp = VehicleInfo()
                        v_tmp.bbox = bbox
                        v_tmp.vehicle_conf = float(boxes.conf[idx])
                        cls_idx = int(boxes.cls[idx])
                        v_tmp.vehicle_type = (
                            gate_coco_classes[cls_idx]
                            if cls_idx < len(gate_coco_classes)
                            else f"vehicle_{cls_idx}"
                        )
                        v_tmp.vehicle_image = raw_frame[
                            bbox[1]: bbox[3], bbox[0]: bbox[2], :
                        ].copy()
                        v_tmp.track_id = f"track_{int(boxes.id[idx])}" if has_ids else None
                        all_vehicles_raw.append(v_tmp)

                allowed_types = ['car', 'bus', 'truck']
                filtered_vehicles = [v for v in all_vehicles_raw if v.vehicle_type.lower() in allowed_types]

                # Vehicles around LINE_1..LINE_2 (OCR zone)
                # Widened zone to better capture EXIT vehicles that move fast and
                # may only appear for a few frames near LINE_2.
                OCR_ZONE_MARGIN_TOP = 20
                OCR_ZONE_MARGIN_BOTTOM = 40
                vehicles_in_detection_zone = []
                for v in filtered_vehicles:
                    if v.bbox is not None:
                        x1, y1, x2, y2 = v.bbox
                        cy = (y1 + y2) // 2
                        in_basic_zone = (y2 >= (line_1_y - OCR_ZONE_MARGIN_TOP) and y1 <= (line_2_y + OCR_ZONE_MARGIN_BOTTOM))
                        force_exit_ocr = False
                        _rec = gate_vehicle_handoffs.get(v.track_id)
                        if _rec and _rec.get('direction') == 'out' and not _rec.get('exit_updated'):
                            force_exit_ocr = True
                        if in_basic_zone or force_exit_ocr:
                            vehicles_in_detection_zone.append(v)

                # --- Push OCR jobs (non-blocking) ---
                # Only crop vehicles that are moving (not frozen) and are in detection zone
                if detector is not None:
                    for v in vehicles_in_detection_zone:
                        vid = v.track_id
                        if vid is None:
                            continue
                        x1, y1, x2, y2 = v.bbox
                        v_cx, v_cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Push crop job — Overwrite mailbox to keep only newest frame
                        crop_job = {
                            'crop_frame': v.vehicle_image,
                            'bbox': list(v.bbox),
                            'frame_count': frame_count,
                        }
                        gate_ocr_enqueue_job(vid, crop_job)

                # --- Clear per-track state when no vehicles ---
                if not filtered_vehicles:
                    no_vehicle_frames += 1
                    if no_vehicle_frames >= 30:
                        plate_votes_by_track.clear()
                        best_plate_by_track.clear()
                else:
                    no_vehicle_frames = 0

                # --- ByteTrack tracking + line-crossing ---
                for vehicle in filtered_vehicles:
                    if vehicle.bbox is None or vehicle.track_id is None:
                        continue

                    vehicle_id = vehicle.track_id
                    x1, y1, x2, y2 = vehicle.bbox
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Build active_tracks list for Render Worker
                    active_tracks.append({'track_id': vehicle_id, 'bbox': list(vehicle.bbox)})

                    # Update stable_plate_cache position (for render)
                    if vehicle_id in stable_plate_cache:
                        stable_plate_cache[vehicle_id]['cx'] = cx
                        stable_plate_cache[vehicle_id]['cy'] = cy
                    elif vehicle_id in best_plate_by_track:
                        stable_plate_cache[vehicle_id] = {
                            'plate': best_plate_by_track[vehicle_id]['plate'],
                            'conf':  best_plate_by_track[vehicle_id]['conf'],
                            'cx': cx, 'cy': cy,
                        }

                    # Track history
                    _tracks_hist[vehicle_id].append((cx, cy, frame_count))
                    if len(_tracks_hist[vehicle_id]) > 30:
                        _tracks_hist[vehicle_id].pop(0)

                    # Initialize crossing state
                    if vehicle_id not in _line_crossings:
                        _line_crossings[vehicle_id] = {
                            'line1_crossed': False,
                            'line2_crossed': False,
                            'line3_crossed': False,
                            'line1_cross_frame': None,
                            'line2_cross_frame': None,
                            'line3_cross_frame': None,
                            # Direction when crossing LINE_1:
                            # - 'up'   : từ dưới -> lên trên (vào bãi)
                            # - 'down' : từ trên -> xuống dưới (ra khỏi bãi)
                            'line1_cross_dir': None,
                            'direction': None,
                            'last_y': cy,
                            'last_y1': y1,
                            'last_y2': y2,
                        }

                    # Update handoff record with plate info (if available)
                    if vehicle_id in gate_vehicle_handoffs:
                        record = gate_vehicle_handoffs[vehicle_id]
                        bp = best_plate_by_track.get(vehicle_id)
                        if bp and bp['conf'] >= record.get('conf', 0):
                            record['plate'] = bp['plate']
                            record['conf']  = bp['conf']
                            record['status'] = 'ocr_done' if bp['conf'] >= 0.50 else 'waiting_ocr'
                            if record.get('direction', 'in') == 'in':
                                artifact = shared_state.gate_ocr_artifacts.get(vehicle_id, {})
                                _upsert_fn(
                                    vehicle_id,
                                    bp['plate'],
                                    bp['conf'],
                                    image_path=artifact.get('image_path'),
                                    ocr_ts=artifact.get('ocr_ts'),
                                )
                            else:
                                if record['status'] == 'ocr_done' and not record.get('exit_updated'):
                                    r_store = record.get('result_store', {})
                                    gate_log_id = r_store.get('gate_log_id')
                                    if gate_log_id:
                                        artifact = shared_state.gate_ocr_artifacts.get(vehicle_id, {})
                                        artifact_img = artifact.get('image_path')
                                        artifact_ts = artifact.get('ocr_ts')
                                        if not artifact_img:
                                            # Keep pending until OCR artifact image exists for text-image coupling.
                                            continue
                                        record['exit_updated'] = True
                                        if io_submit is not None:
                                            print(f"[GATE OCR] OUT submit log_id={gate_log_id} track={vehicle_id} plate={bp['plate']} image={artifact_img}")
                                            io_submit(
                                                update_gate_exit_plate,
                                                gate_log_id,
                                                bp['plate'],
                                                bp['conf'],
                                                image_path=artifact_img,
                                                coalesce_group="gate_exit_update",
                                                coalesce_key=f"{gate_log_id}:{artifact_ts}" if artifact_ts else str(gate_log_id),
                                            )
                                        else:
                                            threading.Thread(
                                                target=update_gate_exit_plate,
                                                args=(gate_log_id, bp['plate'], bp['conf']),
                                                kwargs={'image_path': artifact_img},
                                                daemon=True
                                            ).start()

                    crossing_state = _line_crossings[vehicle_id]
                    last_y  = crossing_state['last_y']
                    last_y2 = crossing_state.get('last_y2', y2)

                    # ===== Determine line crossing state =====
                    # Robust crossing: accept either
                    # 1) bbox overlap with line, OR
                    # 2) center moved across the line between two frames.
                    # This reduces missed handoff when FPS drops / tracker jumps.
                    overlap_line1 = (y1 <= line_1_y <= y2)
                    overlap_line2 = (y1 <= line_2_y <= y2)
                    overlap_line3 = (y1 <= line_3_y <= y2)
                    crossed_line1 = ((last_y < line_1_y <= cy) or (last_y > line_1_y >= cy))
                    crossed_line2 = ((last_y < line_2_y <= cy) or (last_y > line_2_y >= cy))
                    crossed_line3 = ((last_y < line_3_y <= cy) or (last_y > line_3_y >= cy))
                    touches_line1 = overlap_line1 or crossed_line1
                    touches_line2 = overlap_line2 or crossed_line2
                    touches_line3 = overlap_line3 or crossed_line3

                    # ===== LINE 2 TOUCH =====
                    if not crossing_state['line2_crossed'] and touches_line2:
                        crossing_state['line2_crossed'] = True
                        crossing_state['line2_cross_frame'] = frame_count
                        print(f"[GATE] Vehicle {vehicle_id} touched LINE 2 at frame {frame_count}")

                    # ===== LINE 3 TOUCH =====
                    if not crossing_state['line3_crossed'] and touches_line3:
                        crossing_state['line3_crossed'] = True
                        crossing_state['line3_cross_frame'] = frame_count
                        print(f"[GATE] Vehicle {vehicle_id} touched LINE 3 at frame {frame_count}")

                    # ===== LINE 1 TOUCH =====
                    if not crossing_state['line1_crossed'] and touches_line1:
                        crossing_state['line1_crossed'] = True
                        crossing_state['line1_cross_frame'] = frame_count
                        # Infer enter/exit direction from how LINE_1 was crossed.
                        # Note: In image coordinates, y increases downward.
                        # Only infer direction when the center actually crosses the line
                        # between two frames. If we only detect overlap/touch (bbox spans
                        # the line) but the motion direction is unclear (e.g. first frame
                        # of the track), keep it as None and let the fallback rule decide.
                        if crossed_line1:
                            if last_y > line_1_y and cy <= line_1_y:
                                crossing_state['line1_cross_dir'] = 'up'
                            elif last_y < line_1_y and cy >= line_1_y:
                                crossing_state['line1_cross_dir'] = 'down'
                        print(f"[GATE] Vehicle {vehicle_id} touched LINE 1 at frame {frame_count}")

                    # ===== Determine direction and trigger crossing events =====
                    if crossing_state['direction'] is None:
                        
                        l1f = crossing_state.get('line1_cross_frame')
                        l2f = crossing_state.get('line2_cross_frame')
                        l3f = crossing_state.get('line3_cross_frame')
                        line1_dir = crossing_state.get('line1_cross_dir')
                        
                        _new_dir = None
                        
                        # 1. Xe Vào (Entry): Line 2 to Line 1 (moving up)
                        if l1f is not None and l2f is not None:
                            if line1_dir == 'up':
                                _new_dir = 'in'
                            elif line1_dir is None and l2f < l1f:
                                _new_dir = 'in'
                                
                        # 2. Xe Ra (Exit): Line 1 to Line 3 (moving down)
                        if _new_dir is None and l1f is not None and l3f is not None:
                            if line1_dir == 'down':
                                _new_dir = 'out'
                            elif line1_dir is None and l1f < l3f:
                                _new_dir = 'out'
                                
                        # 3. Fallbacks for simultaneous touch without explicit cross_dir
                        if _new_dir is None and crossing_state['line1_crossed']:
                            hist = _tracks_hist.get(vehicle_id, [])
                            if len(hist) >= 2:
                                dy = cy - hist[0][1]
                                if dy < -2 and crossing_state['line2_crossed']:
                                    _new_dir = 'in'
                                elif dy > 2 and crossing_state['line3_crossed']:
                                    _new_dir = 'out'

                        if _new_dir is None:
                            continue  # Wait for clear direction

                        crossing_state['direction'] = _new_dir

                        if _new_dir == 'in':
                            print(f"[GATE] Vehicle {vehicle_id} - XE VÀO")
                            
                            # 1. Handoff Initialization
                            if vehicle_id not in gate_vehicle_handoffs:
                                vehicle_type = vehicle.vehicle_type if vehicle.vehicle_type else 'car'
                                handoff_record = {
                                    'entry_id': f"entry_{frame_count}_{vehicle_id}",
                                    'plate': None,
                                    'conf': 0.0,
                                    'timestamp': datetime.now(),
                                    'vehicle_type': vehicle_type,
                                    'status': 'waiting_ocr',
                                    'assigned': False,
                                    'ready_to_handover': True,
                                    'handover_trigger_frame': frame_count,
                                    'direction': 'in',
                                }
                                plate_handoff_queue.append(handoff_record)
                                gate_vehicle_handoffs[vehicle_id] = handoff_record
                                print(f"[HANDOFF] Physical vehicle {vehicle_id} entered (Waiting OCR). Queue: {len(plate_handoff_queue)}")
                            else:
                                gate_vehicle_handoffs[vehicle_id]['ready_to_handover'] = True
                                gate_vehicle_handoffs[vehicle_id]['handover_trigger_frame'] = frame_count

                            # 2. Prepare Plate Data
                            best_plate_text = None
                            best_plate_conf = 0.0
                            if vehicle_id in best_plate_by_track:
                                best_plate_text = best_plate_by_track[vehicle_id]['plate']
                                best_plate_conf = best_plate_by_track[vehicle_id]['conf']
                            
                            # 3. Push to FIFO Queue for immediate matching availability
                            plate_entry = {
                                'ingress_seq': allocate_ingress_seq(),
                                'plate': best_plate_text or None,
                                'conf': best_plate_conf,
                                'timestamp': datetime.now(),
                                'assigned': False,
                                'reserved_ingress_seq': None,
                                'gate_track_id': vehicle_id,
                            }
                            with plate_fifo_lock:
                                plate_fifo_queue.append(plate_entry)
                                unassigned_count = len([p for p in plate_fifo_queue if not p.get('assigned')])
                            print(
                                f"[GATE→QUEUE] Added plate: {best_plate_text or 'NO_PLATE'} "
                                f"(ingress_seq={plate_entry.get('ingress_seq')}, conf={best_plate_conf:.2f}, "
                                f"unassigned in queue: {unassigned_count})"
                            )

                            crossing_events.append({'plate': best_plate_text, 'conf': best_plate_conf, 'direction': 'in'})

                            # 4. Log Entry to Database with Initial Image Crop
                            _entry_result_store = gate_vehicle_handoffs[vehicle_id]
                            _img_path = None
                            _cached_img = track_plate_images.get(vehicle_id)
                            if _cached_img is not None:
                                try:
                                    _fname = f"gate_{best_plate_text or 'noplate'}_{frame_count}.jpg"
                                    _fpath = os.path.join(gate_capture_dir, _fname)
                                    cv2.imwrite(_fpath, _cached_img)
                                    _img_path = f"/static/gate_captures/{_fname}"
                                except Exception as _e:
                                    print(f"[GATE] Failed to save plate image: {_e}")
                            
                            if io_submit is not None:
                                io_submit(
                                    log_gate_entry,
                                    best_plate_text,
                                    best_plate_conf,
                                    image_path=_img_path,
                                    result_store=_entry_result_store,
                                    coalesce_group="gate_entry_log",
                                    coalesce_key=str(vehicle_id),
                                )
                            else:
                                threading.Thread(
                                    target=log_gate_entry,
                                    args=(best_plate_text, best_plate_conf),
                                    kwargs={'image_path': _img_path, 'result_store': _entry_result_store},
                                    daemon=True
                                ).start()
                            continue

                        elif _new_dir == 'out':
                            print(f"[GATE] Vehicle {vehicle_id} - XE RA")
                            
                            # Validate plate
                            _exit_plate = None
                            _exit_conf  = 0.0
                            if vehicle_id in best_plate_by_track:
                                _exit_plate = best_plate_by_track[vehicle_id]['plate']
                                _exit_conf  = best_plate_by_track[vehicle_id]['conf']
                            if not _exit_plate and vehicle_id in stable_plate_cache:
                                _exit_plate = stable_plate_cache[vehicle_id]['plate']
                                _exit_conf  = stable_plate_cache[vehicle_id]['conf']
                            if not _exit_plate and vehicle_id in gate_vehicle_handoffs:
                                _exit_plate = gate_vehicle_handoffs[vehicle_id].get('plate')
                                _exit_conf  = gate_vehicle_handoffs[vehicle_id].get('conf', 0.0)
                            
                            _exit_img_path = None
                            _cached_exit = track_plate_images.get(vehicle_id)
                            if _cached_exit is not None:
                                try:
                                    _fname_plate = _exit_plate or "noplate"
                                    _fname_ex = f"gate_{_fname_plate}_{frame_count}_out.jpg"
                                    cv2.imwrite(os.path.join(gate_capture_dir, _fname_ex), _cached_exit)
                                    _exit_img_path = f"/static/gate_captures/{_fname_ex}"
                                except Exception:
                                    pass
                            
                            if vehicle_id not in gate_vehicle_handoffs:
                                vehicle_type = vehicle.vehicle_type if vehicle.vehicle_type else 'car'
                                handoff_record = {
                                    'entry_id': f"exit_{frame_count}_{vehicle_id}",
                                    'plate': _exit_plate,
                                    'conf': _exit_conf,
                                    'timestamp': datetime.now(),
                                    'vehicle_type': vehicle_type,
                                    'status': 'waiting_ocr' if not _exit_plate else 'ocr_done',
                                    'assigned': False,
                                    'ready_to_handover': True,
                                    'handover_trigger_frame': frame_count,
                                    'direction': 'out',
                                    'result_store': {} 
                                }
                                gate_vehicle_handoffs[vehicle_id] = handoff_record
                            else:
                                gate_vehicle_handoffs[vehicle_id]['ready_to_handover'] = True
                                gate_vehicle_handoffs[vehicle_id]['handover_trigger_frame'] = frame_count
                                gate_vehicle_handoffs[vehicle_id]['direction'] = 'out'
                                if 'result_store' not in gate_vehicle_handoffs[vehicle_id]:
                                    gate_vehicle_handoffs[vehicle_id]['result_store'] = {}

                            _exit_result_store = gate_vehicle_handoffs[vehicle_id]['result_store']

                            if _exit_plate:
                                crossing_events.append({'plate': _exit_plate, 'conf': _exit_conf, 'direction': 'out'})

                            # Emergency OCR retry for EXIT:
                                # if plate not ready at crossing moment, push a fresh crop immediately
                            # so OCR worker can still update Gate OUT log afterwards.
                            if detector is not None and not _exit_plate and vehicle.vehicle_image is not None:
                                try:
                                    gate_ocr_enqueue_job(
                                        vehicle_id,
                                        {
                                            'crop_frame': vehicle.vehicle_image,
                                            'bbox': list(vehicle.bbox),
                                            'frame_count': frame_count,
                                        },
                                    )
                                    print(f"[GATE OCR] EXIT retry queued for {vehicle_id}")
                                except Exception as _ocrq_e:
                                    print(f"[GATE OCR] EXIT retry queue failed for {vehicle_id}: {_ocrq_e}")
                                
                            if io_submit is not None:
                                io_submit(
                                    log_gate_exit,
                                    _exit_plate,
                                    _exit_conf,
                                    image_path=_exit_img_path,
                                    result_store=_exit_result_store,
                                    coalesce_group="gate_exit_log",
                                    # Coalesce dựa trên plate để hạn chế việc track_id bị re-id
                                    # gây submit Gate OUT lặp nhiều lần.
                                    coalesce_key=f"plate:{_exit_plate or 'noplate'}",
                                )
                            else:
                                threading.Thread(
                                    target=log_gate_exit,
                                    args=(_exit_plate, _exit_conf),
                                    kwargs={'image_path': _exit_img_path, 'result_store': _exit_result_store},
                                    daemon=True
                                ).start()
                            
                            # Do NOT clean per-track state immediately, allow OCR to continue
                            # (Cleaned up in vehicles_to_remove after 60 frames instead)

                    # Update last_y
                    crossing_state['last_y']  = cy
                    crossing_state['last_y1'] = y1
                    crossing_state['last_y2'] = y2

                # --- Clean up old tracks (not seen for 60 frames) ---
                vehicles_to_remove = [
                    vid for vid in _tracks_hist
                    if _tracks_hist[vid] and frame_count - _tracks_hist[vid][-1][2] > 60
                ]
                for vehicle_id in vehicles_to_remove:
                    cross_st = _line_crossings.get(vehicle_id, {})
                    if cross_st.get('needs_ticket') and TRACKING_ENABLED:
                        try:
                            _p_text = ''
                            _p_conf = 0.0
                            if vehicle_id in best_plate_by_track:
                                _p_text = best_plate_by_track[vehicle_id]['plate']
                                _p_conf = best_plate_by_track[vehicle_id]['conf']
                            ticket = VehicleTicket.create(
                                plate_text=_p_text, plate_conf=_p_conf,
                                vehicle_type=cross_st.get('ticket_vehicle_type', 'car'),
                                vehicle_bbox=cross_st.get('ticket_bbox', [0, 0, 0, 0])
                            )
                            tracker = get_tracker()
                            tracker.register_vehicle(ticket)
                            stats = tracker.get_stats()
                            vehicle_tracking_state['pending_count'] = stats['pending']
                            vehicle_tracking_state['matched_count'] = stats['total_matched']
                            vehicle_tracking_state['last_update'] = datetime.now().isoformat()
                            vehicle_tracking_state['recent_tickets'].append(ticket.to_dict())
                            vehicle_tracking_state['recent_tickets'] = vehicle_tracking_state['recent_tickets'][-10:]
                            if socketio is not None:
                                socketio.emit('tracking_update', {
                                    'event': 'vehicle_entered',
                                    'ticket_id': ticket.ticket_id[:8],
                                    'plate': _p_text or 'N/A',
                                    'pending_count': stats['pending'],
                                    'timestamp': datetime.now().isoformat()
                                })
                        except Exception as _te:
                            print(f"[TRACKING] Error creating ticket: {_te}")

                    # Handoff cleanup
                    if vehicle_id in gate_vehicle_handoffs:
                        record = gate_vehicle_handoffs[vehicle_id]
                        bp = best_plate_by_track.get(vehicle_id)
                        if bp:
                            record['plate'] = bp['plate']
                            record['conf']  = bp['conf']
                        
                        direction = record.get('direction', 'in')
                        _final_plate = record.get('plate')
                        
                        if direction == 'in':
                            if _final_plate:
                                record['status'] = 'ocr_done'
                                artifact = shared_state.gate_ocr_artifacts.get(vehicle_id, {})
                                if not _upsert_fn(
                                    vehicle_id,
                                    _final_plate,
                                    record.get('conf', 0),
                                    image_path=artifact.get('image_path'),
                                    ocr_ts=artifact.get('ocr_ts'),
                                ):
                                    if not update_plate_fifo_entry(vehicle_id, _final_plate, record.get('conf', 0.0)):
                                        with plate_fifo_lock:
                                            plate_fifo_queue.append({
                                                'ingress_seq': allocate_ingress_seq(),
                                                'plate': _final_plate,
                                                'conf': record.get('conf', 0.0),
                                                'timestamp': datetime.now(),
                                                'assigned': False,
                                                'reserved_ingress_seq': None,
                                                'gate_track_id': vehicle_id,
                                            })
                                        print(f"[GATE→QUEUE] Added plate {_final_plate} on disappear (track={vehicle_id})")
                        elif direction == 'out' and not record.get('exit_updated'):
                            if _final_plate:
                                record['status'] = 'ocr_done'
                            r_store = record.get('result_store', {})
                            gate_log_id = r_store.get('gate_log_id')
                            if gate_log_id:
                                _img_path = None
                                _cached_exit = track_plate_images.get(vehicle_id)
                                if _cached_exit is not None:
                                    try:
                                        _fname_plate = _final_plate or "noplate"
                                        _fname_ex = f"gate_{_fname_plate}_{frame_count}_out.jpg"
                                        cv2.imwrite(os.path.join(gate_capture_dir, _fname_ex), _cached_exit)
                                        _img_path = f"/static/gate_captures/{_fname_ex}"
                                    except Exception:
                                        pass
                                artifact = shared_state.gate_ocr_artifacts.get(vehicle_id, {})
                                artifact_img = artifact.get('image_path') or _img_path
                                artifact_ts = artifact.get('ocr_ts')
                                if io_submit is not None:
                                    print(f"[GATE OCR] OUT cleanup submit log_id={gate_log_id} track={vehicle_id} plate={_final_plate} image={artifact_img}")
                                    io_submit(
                                        update_gate_exit_plate,
                                        gate_log_id,
                                        _final_plate,
                                        record.get('conf', 0.0),
                                        image_path=artifact_img,
                                        coalesce_group="gate_exit_update",
                                        coalesce_key=f"{gate_log_id}:{artifact_ts}" if artifact_ts else str(gate_log_id),
                                    )
                                else:
                                    threading.Thread(
                                        target=update_gate_exit_plate,
                                        args=(gate_log_id, _final_plate, record.get('conf', 0.0)),
                                        kwargs={'image_path': artifact_img},
                                        daemon=True
                                    ).start()
                            record['exit_updated'] = True

                        if not record.get('ready_to_handover'):
                            record['ready_to_handover'] = True
                            record['handover_trigger_frame'] = frame_count

                        gate_ocr_persist_ctx_before_handoff_drop(vehicle_id, record)
                        # Cleanup to prevent memory leak — pop BEFORE persist so the guard skips
                        # (vehicle no longer in gate_vehicle_handoffs → no stale-ctx overwrite risk)
                        gate_vehicle_handoffs.pop(vehicle_id, None)
                        with shared_state.gate_ocr_scheduler_lock:
                            shared_state.gate_ocr_latest_jobs.pop(vehicle_id, None)
                            shared_state.gate_ocr_track_db_ctx.pop(vehicle_id, None)
                            shared_state.gate_ocr_artifacts.pop(vehicle_id, None)

                    del _tracks_hist[vehicle_id]
                    _line_crossings.pop(vehicle_id, None)
                    stable_plate_cache.pop(vehicle_id, None)
                    track_plate_images.pop(vehicle_id, None)
                    plate_votes_by_track.pop(vehicle_id, None)
                    best_plate_by_track.pop(vehicle_id, None)

                # --- Re-anchor plate label on ByteTrack ID switch ---
                # When tracker swaps `track_id`, the OCR plate is still associated to
                # the old track id, so the label can appear "off" or disappear.
                # If a new track (missing plate) is spatially close to a track
                # that already has an OCR plate in the *current* frame, copy plate/conf.
                try:
                    # Build bbox map for current tracks
                    _bboxes_by_tid = {}
                    _plate_tids = []
                    _missing_tids = []
                    for _v in filtered_vehicles:
                        if _v.track_id is None or _v.bbox is None:
                            continue
                        _tid = _v.track_id
                        _bboxes_by_tid[_tid] = list(_v.bbox)
                        _pi = stable_plate_cache.get(_tid) if stable_plate_cache is not None else None
                        if _pi and _pi.get('plate'):
                            _plate_tids.append(_tid)
                        else:
                            _missing_tids.append(_tid)

                    if _plate_tids and _missing_tids:
                        import math

                        def _iou(_a, _b):
                            ax1, ay1, ax2, ay2 = _a
                            bx1, by1, bx2, by2 = _b
                            inter_x1 = max(ax1, bx1)
                            inter_y1 = max(ay1, by1)
                            inter_x2 = min(ax2, bx2)
                            inter_y2 = min(ay2, by2)
                            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                                return 0.0
                            inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                            area_a = (ax2 - ax1) * (ay2 - ay1)
                            area_b = (bx2 - bx1) * (by2 - by1)
                            denom = float(area_a + area_b - inter) or 1.0
                            return inter / denom

                        for _t in _missing_tids:
                            _tb = _bboxes_by_tid.get(_t)
                            if not _tb:
                                continue
                            tx1, ty1, tx2, ty2 = _tb
                            tcx = (tx1 + tx2) // 2
                            tcy = (ty1 + ty2) // 2

                            # Dynamic distance threshold based on bbox size
                            t_diag = math.hypot(tx2 - tx1, ty2 - ty1)
                            dist_thresh = 0.75 * t_diag + 20

                            best_cand = None
                            best_score = -1.0
                            for _c in _plate_tids:
                                _cb = _bboxes_by_tid.get(_c)
                                if not _cb:
                                    continue
                                _pi_c = stable_plate_cache.get(_c, {})
                                cand_conf = float(_pi_c.get('conf') or 0.0)
                                if cand_conf < 0.30:
                                    continue

                                cx1, cy1, cx2, cy2 = _cb
                                ccx = (cx1 + cx2) // 2
                                ccy = (cy1 + cy2) // 2
                                dist = math.hypot(ccx - tcx, ccy - tcy)
                                if dist > dist_thresh:
                                    continue

                                iou = _iou(_tb, _cb)
                                # Score: prioritize IoU first, then confidence
                                score = iou * 2.0 + cand_conf
                                if score > best_score:
                                    best_score = score
                                    best_cand = _c

                            # Require at least some overlap OR a reasonable proximity
                            if best_cand is not None:
                                iou_best = _iou(_tb, _bboxes_by_tid[best_cand])
                                if iou_best >= 0.05 or best_score >= 0.35:
                                    _pi_best = stable_plate_cache.get(best_cand, {})
                                    plate = _pi_best.get('plate')
                                    conf = float(_pi_best.get('conf') or 0.0)
                                    if plate:
                                        stable_plate_cache[_t] = {
                                            'plate': plate,
                                            'conf': conf,
                                            'cx': tcx,
                                            'cy': tcy,
                                        }
                                        # Also mirror into best_plate_by_track so gate entry logic can use it.
                                        if _t not in best_plate_by_track:
                                            best_plate_by_track[_t] = {'plate': plate, 'conf': conf}
                except Exception:
                    # Never break pipeline due to re-anchor failure
                    pass

                # Update shared OCR results dict for API polling
                gate_ocr_results_dict['vehicles'] = filtered_vehicles
                gate_ocr_results_dict['last_detection_time'] = datetime.now().isoformat()
                _latest_plate = None
                _latest_conf  = 0.0
                for veh in filtered_vehicles:
                    if veh.plate_text and veh.plate_conf > _latest_conf:
                        _latest_plate = veh.plate_text
                        _latest_conf  = veh.plate_conf
                gate_ocr_results_dict['latest_plate'] = _latest_plate
                gate_ocr_results_dict['latest_plate_confidence'] = _latest_conf

            except Exception as exc:
                print(f"[GATE Detect+Track] Error: {exc}")
                import traceback
                traceback.print_exc()

        # --- Push frame to Render Worker (non-blocking) ---
        render_job = {
            'frame': frame.copy(),
            'active_tracks': list(active_tracks),
            'crossing_events': list(crossing_events),
            'frame_count': frame_count,
            'line1_y': line_1_y,
            'line2_y': line_2_y,
            # IMPORTANT: pass scaled LINE_3_Y to render so it doesn't rely on
            # unscaled line3_y_ref[0] (which can visually appear near LINE 1).
            'line3_y': line_3_y,
        }
        # Discard oldest if full (only latest frame matters for display)
        try:
            gate_render_queue.put_nowait(render_job)
        except Exception:
            try:
                gate_render_queue.get_nowait()
                gate_render_queue.put_nowait(render_job)
            except Exception:
                pass

        # Adaptive sleep — target ~30fps for detect+track loop
        elapsed = time.time() - frame_start
        if runtime is not None:
            _qpend, _qlatest = gate_ocr_scheduler_depth()
            runtime.mark_loop(
                elapsed * 1000.0,
                _qpend + _qlatest,
                gate_render_queue.qsize(),
                len(plate_fifo_queue),
            )
        time.sleep(max(0.0, 0.033 - elapsed))

    # Cleanup
    if rtsp_cap:
        rtsp_cap.release()
    if cap:
        cap.release()
    print("[GATE] Detect+Track Worker stopped")
