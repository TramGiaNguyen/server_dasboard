#!/usr/bin/env python3
"""
eval_temporal_analysis.py

Trien khai Section C (4 che do Hysteresis x Inner-Core) va Section E (do tre
cap nhat trang thai) voi bieu do matplotlib, bao cao ASCII day du, va
JSON/CSV exports.

Su dung cache detections (pickle) de chi can chay YOLO inference 1 lan duy nhat
cho toan bo video, roi reuse cho tat ca mode.

Usage:
    python eval_temporal_analysis.py --section ce --save-charts --device cuda

    # Chi Section C
    python eval_temporal_analysis.py --section c --device cuda

    # Chi Section E
    python eval_temporal_analysis.py --section e --device cuda
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_VIDEO        = "static/video/CAM_PARKING.mp4"
DEFAULT_SLOT_GT     = "annotations/CAM_PARKING_slot_state_gt.csv"
DEFAULT_BBOX_GT     = "annotations/CAM_PARKING_bbox_outside_gt.csv"
DEFAULT_FRAME_LIST  = "annotations/frame_numbers.txt"
DEFAULT_OUTPUT      = "eval_results/temporal_analysis"

STATE_CATEGORIES    = ["available", "occupied", "overlapping", "outside"]
PRED_STATES        = ["available", "occupied", "overlapping", "outside"]
GT_STATES          = ["available", "occupied", "overlapping"]

VEHICLE_CLASS_IDS   = [2, 5, 7]

DEFAULT_HYST_THRESHOLD = 45
DEFAULT_VIDEO_FPS      = 30.0
DEFAULT_CONF           = 0.30

VIDEO_START_FRAME  = 6
VIDEO_END_FRAME    = 41319
FULL_VIDEO_FRAMES  = VIDEO_END_FRAME - VIDEO_START_FRAME + 1


# ---------------------------------------------------------------------------
# Helpers (copied from eval_occupancy_comprehensive.py)
# ---------------------------------------------------------------------------

def resolve_path(base, path_arg, default):
    p = Path(path_arg)
    if p.is_absolute():
        return str(p)
    candidates = [path_arg, PROJECT_ROOT / path_arg, PROJECT_ROOT / default]
    for c in candidates:
        if Path(c).exists():
            return str(Path(c).resolve())
    return str(PROJECT_ROOT / default)


def parse_args():
    p = argparse.ArgumentParser(
        description="Section C + E: Temporal analysis (Hysteresis x Inner-Core) "
                    "and state update latency evaluation.")
    p.add_argument("--section",      default="ce",
                   help="Sections to run: 'c'=Section C, 'e'=Section E, "
                        "'ce'=both (default: ce)")
    p.add_argument("--save-charts",  action="store_true",
                   help="Save matplotlib charts as PNG")
    p.add_argument("--video",        default=DEFAULT_VIDEO,
                   help="Path to video file")
    p.add_argument("--slot-gt",     default=DEFAULT_SLOT_GT,
                   help="Path to slot state GT CSV")
    p.add_argument("--bbox-gt",     default=DEFAULT_BBOX_GT,
                   help="Path to outside bbox GT CSV")
    p.add_argument("--frame-list",  default=DEFAULT_FRAME_LIST,
                   help="Path to frame numbers list")
    p.add_argument("--output",      default=DEFAULT_OUTPUT,
                   help="Output directory")
    p.add_argument("--conf",        type=float, default=DEFAULT_CONF,
                   help="YOLO confidence threshold (default: 0.30)")
    p.add_argument("--model-size", default="l",
                   choices=["n", "s", "m", "l"],
                   help="YOLO model size: n=nano, s=small, m=medium, l=large (default: l)")
    p.add_argument("--device",      default="cpu",
                   help="YOLO device: 'cuda' (or '0') for GPU, 'cpu' for CPU (default: cpu)")
    p.add_argument("--hyst-threshold", type=int, default=DEFAULT_HYST_THRESHOLD,
                   help="Hysteresis threshold in frames (default: 45)")
    p.add_argument("--no-cache",    action="store_true",
                   help="Force re-run YOLO inference (ignore cache)")
    p.add_argument("--warmup-frames", type=int, default=100,
                   help="Number of warm-up frames before GT period (default: 100)")
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="IoU threshold for bbox matching in outside detection (default: 0.30)")
    return p.parse_args()


def _normalize_device(device):
    """Normalize device string for YOLO."""
    d = device.lower().strip()
    if d == "cuda":
        return "0"
    if d == "cpu":
        return "cpu"
    if d.isdigit():
        return d
    return device


# ---------------------------------------------------------------------------
# Parking zones (copied from eval_occupancy_comprehensive.py)
# ---------------------------------------------------------------------------

def define_parking_areas():
    return [
        [(305,101),(360,101),(353,153),(288,151),(305,99)],
        [(362,101),(421,101),(426,159),(354,154)],
        [(421,103),(486,102),(504,163),(426,158)],
        [(488,104),(553,108),(583,167),(505,163)],
        [(554,107),(623,112),(666,174),(585,168)],
        [(625,112),(692,117),(747,180),(668,174)],
        [(694,119),(761,125),(822,189),(749,180)],
        [(288,151),(352,155),(347,232),(270,225)],
        [(353,155),(426,160),(436,240),(348,233)],
        [(428,161),(505,165),(528,248),(437,241)],
        [(505,165),(584,168),(626,257),(529,248)],
        [(585,170),(668,176),(720,265),(628,257)],
        [(669,175),(752,181),(813,272),(721,266)],
        [(750,180),(820,188),(899,281),(814,273)],
        [(250,341),(343,347),(370,499),(251,495)],
        [(344,348),(445,353),(504,499),(373,497)],
        [(504,353),(606,355),(720,486),(589,497)],
        [(765,124),(823,190),(889,197),(837,132)],
        [(839,134),(887,139),(954,205),(891,199)],
    ]


def create_inner_zones(areas, shrink_percentage=0.20):
    _FAR_SLOTS = set(range(0, 14)) | {17, 18}
    _FAR_SLOT_SHRINK = 0.35
    inner_zones = []
    for idx, area in enumerate(areas):
        slot_shrink = _FAR_SLOT_SHRINK if idx in _FAR_SLOTS else shrink_percentage
        points = np.array(area, dtype=np.float32)
        centroid = np.mean(points, axis=0)
        inner_points = []
        for point in points:
            vector = point - centroid
            new_point = centroid + vector * (1 - slot_shrink)
            inner_points.append(tuple(new_point.astype(int)))
        inner_zones.append(inner_points)
    return inner_zones


def scale_areas(raw_areas, sx, sy):
    return [[(int(x * sx), int(y * sy)) for x, y in area] for area in raw_areas]


def load_parking_zones(video_path=None):
    """Load and scale parking zones to match video resolution.

    Parking zones are defined for baseline 1020x500.
    We scale them to match the actual video resolution.
    """
    video_path = video_path or DEFAULT_VIDEO
    video_path_resolved = resolve_path("static/video", video_path, DEFAULT_VIDEO.lstrip("static/"))

    sample_frame = None
    if Path(video_path_resolved).exists():
        cap = cv2.VideoCapture(str(video_path_resolved))
        if cap.isOpened():
            ret, sample_frame = cap.read()
            cap.release()
    if sample_frame is None:
        sample_frame = np.zeros((1080, 1910, 3), dtype=np.uint8)

    h, w = sample_frame.shape[:2]
    sx, sy = w / 1020.0, h / 500.0

    raw_areas = define_parking_areas()
    raw_inner = create_inner_zones(raw_areas)
    areas = scale_areas(raw_areas, sx, sy)
    inner_areas = scale_areas(raw_inner, sx, sy)

    print(f"[Parking] {len(areas)} slots, baseline 1020x500 -> scaled to {w}x{h} "
          f"(sx={sx:.3f}, sy={sy:.3f})")
    return areas, inner_areas


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_slot_state_gt(path):
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[(int(row["frame"]), int(row["slot"]))] = row["state"].strip()
    return gt


def load_bbox_gt(path):
    gt = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[int(row["frame"])].append({
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
                "class_name": row.get("class_name", "outside").strip(),
            })
    return dict(gt)


def load_frame_numbers(path):
    with open(path, encoding="utf-8") as f:
        return sorted(int(l.strip()) for l in f if l.strip())


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

def init_yolo(device="cpu", model_size="l"):
    model_name = f"yolov8{model_size}.pt"
    model_path = PROJECT_ROOT / "static" / "models" / model_name
    normalized = _normalize_device(device)
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print(f"[YOLO] Model loaded: {model_path} | device={normalized}")
        return model, normalized
    except Exception as e:
        print(f"[YOLO] ERROR: cannot load model ({e})")
        return None, "cpu"


def detect_yolo(model, frame, conf=0.30, device="0"):
    if model is None:
        return []
    try:
        results = model(frame, conf=conf, iou=0.45,
                        classes=VEHICLE_CLASS_IDS, verbose=False, device=device)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                conf_s = float(box.conf[0].cpu().numpy())
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy,
                    "class_id": cls_id, "conf": conf_s,
                })
        return detections
    except Exception as e:
        print(f"[YOLO] Inference error: {e}")
        return []


# ---------------------------------------------------------------------------
# Slot state inference
# ---------------------------------------------------------------------------

def infer_slot_state_instant(detections, areas, inner_areas, num_slots,
                              use_inner_core=True):
    slot_states = ["available"] * num_slots

    for det in detections:
        cx, cy = det["cx"], det["cy"]
        for s in range(num_slots):
            in_outer = cv2.pointPolygonTest(
                np.array(areas[s], np.int32), (cx, cy), False) >= 0
            if not in_outer:
                continue

            if use_inner_core:
                in_inner = cv2.pointPolygonTest(
                    np.array(inner_areas[s], np.int32), (cx, cy), False) >= 0
                state = "overlapping" if not in_inner else "occupied"
            else:
                state = "occupied"

            if slot_states[s] == "available":
                slot_states[s] = state
            elif slot_states[s] == "occupied" and state == "overlapping":
                slot_states[s] = "overlapping"

    return slot_states


def _hysteresis_step(detections, areas, inner_areas, num_slots,
                      threshold, use_inner_core, counters, current_states):
    """
    Simulate hysteresis matching the actual project logic:
      - entering a slot (detection found) → transition IMMEDIATELY (0 frames delay)
      - leaving a slot (no detection) → wait threshold frames without detection → transition

    Counters[s] tracks consecutive frames WITH OUT detection.
    Transition to occupied/overlapping: immediate (no counter needed).
    Transition to available: wait threshold frames of no detection.
    """
    overlap_threshold = 8
    instant_states = infer_slot_state_instant(
        detections, areas, inner_areas, num_slots, use_inner_core=use_inner_core)

    new_states = list(current_states)

    # Pre-compute which slots have a detection this frame
    detected_slots = set()
    for det in detections:
        cx, cy = det["cx"], det["cy"]
        for s in range(num_slots):
            in_outer = cv2.pointPolygonTest(
                np.array(areas[s], np.int32), (cx, cy), False) >= 0
            if not in_outer:
                continue
            if use_inner_core:
                in_inner = cv2.pointPolygonTest(
                    np.array(inner_areas[s], np.int32), (cx, cy), False) >= 0
            else:
                in_inner = False
            if in_outer:
                detected_slots.add(s)

    for s in range(num_slots):
        inst = instant_states[s]
        curr = current_states[s]

        if inst == curr:
            counters[s] = 0
            continue

        # Detect direction:
        #   entering = instant is more "occupied" than current
        #   leaving  = instant is more "available" than current
        #
        # Priority (highest to lowest occupancy):
        #   available < overlapping < occupied
        STATE_RANK = {"available": 0, "overlapping": 1, "occupied": 2}

        entering = STATE_RANK.get(inst, 0) > STATE_RANK.get(curr, 0)
        leaving = STATE_RANK.get(inst, 0) < STATE_RANK.get(curr, 0)

        if entering:
            # Vehicle entering the slot → transition immediately
            new_states[s] = inst
            counters[s] = 0

        elif leaving:
            # No detection for this slot → start/continue empty counter
            # (the slot was previously occupied/overlapping, now no detection)
            if s not in detected_slots:
                counters[s] += 1
            else:
                counters[s] = 0

            # Determine effective threshold for overlapping transitions
            is_overlap_trans = (inst == "available" and curr == "overlapping") or \
                               (inst == "overlapping" and curr == "occupied")
            eff_thresh = overlap_threshold if is_overlap_trans else threshold

            if counters[s] >= eff_thresh:
                new_states[s] = inst
                counters[s] = 0

    for s in range(num_slots):
        current_states[s] = new_states[s]

    return new_states


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_confusion_matrix(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        ti = idx.get(t, -1)
        pi = idx.get(p, -1)
        if ti >= 0 and pi >= 0:
            cm[ti][pi] += 1
    res = {}
    total = len(y_true)
    for c in classes:
        i = idx[c]
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c_idx] for c_idx in range(n)) - tp
        p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        res[c] = {
            "precision": round(p_val * 100, 2),
            "recall":    round(r_val * 100, 2),
            "f1":        round(f_val * 100, 2),
            "support":   sum(cm[i]),
            "tp": tp, "fp": fp, "fn": fn,
        }
    total_tp = sum(cm[i][i] for i in range(n))
    acc = total_tp / total if total > 0 else 0.0

    core_classes = ["available", "occupied", "overlapping"]
    core_present = [c for c in core_classes if c in res and res[c]["support"] > 0]
    if core_present:
        core_f1_sum = sum(res[c]["f1"] for c in core_present)
        core_weighted_f1 = sum(
            res[c]["f1"] * res[c]["support"] for c in core_present
        ) / sum(res[c]["support"] for c in core_present)
    else:
        core_f1_sum = 0.0
        core_weighted_f1 = 0.0

    res["_overall"] = {
        "accuracy":        round(acc * 100, 2),
        "macro_precision": round(np.mean([res[c]["precision"] for c in classes]), 2),
        "macro_recall":    round(np.mean([res[c]["recall"]    for c in classes]), 2),
        "macro_f1":        round(np.mean([res[c]["f1"]        for c in classes]), 2),
        "total": total,
        "macro_f1_3class": round(core_f1_sum / len(core_present), 2) if core_present else 0.0,
        "weighted_f1_3class": round(core_weighted_f1, 2),
    }
    return res, cm


def detect_outside(detections, areas, num_slots):
    outside = []
    for det in detections:
        cx, cy = det["cx"], det["cy"]
        in_any = False
        for s in range(num_slots):
            if cv2.pointPolygonTest(
                    np.array(areas[s], np.int32), (cx, cy), False) >= 0:
                in_any = True
                break
        if not in_any:
            outside.append(det)
    return outside


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    return float(inter / (ua + ub - inter))


def match_detections_to_gt(preds, gt_bboxes, iou_thresh):
    if not gt_bboxes:
        return 0, len(preds), 0
    if not preds:
        return 0, 0, len(gt_bboxes)
    gt_matched = [False] * len(gt_bboxes)
    tp, fp = 0, 0
    for p in preds:
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gt_bboxes):
            if gt_matched[gi]:
                continue
            iou = bbox_iou(p, g)
            if iou >= iou_thresh and iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0:
            gt_matched[best_gi] = True
            tp += 1
        else:
            fp += 1
    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Cache management — run full video YOLO inference once
# ---------------------------------------------------------------------------

def _make_cache_key(video_path, model, conf, device):
    vid_stat = os.stat(video_path)
    key = f"{video_path}_{vid_stat.st_size}_{vid_stat.st_mtime:.0f}_{model}_{conf}_{device}"
    return hashlib.md5(key.encode()).hexdigest()


def run_full_video_inference(args, yolo_model, areas, inner_areas, num_slots):
    """
    Chay YOLO inference tren toan bo video (frame 6 → 41319).
    Tra ve dict: {frame_idx: detections}
    Chi chay 1 lan, cache lai cho tat ca mode.
    """
    cache_dir = PROJECT_ROOT / ".cache" / "eval_temporal"
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_path = resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/"))
    device_norm = _normalize_device(args.device)
    model_name = f"yolov8{args.model_size}.pt"
    cache_key = _make_cache_key(video_path, model_name, args.conf, device_norm)
    cache_file = cache_dir / f"detections_{cache_key}.pkl"

    if not args.no_cache and cache_file.exists():
        print(f"[Cache] Loading cached detections from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"\n[Inference] Running YOLO on full video: {total_vid_frames} frames @ {fps_vid:.1f} FPS")
    print(f"           Range: frame {VIDEO_START_FRAME} → {VIDEO_END_FRAME} ({FULL_VIDEO_FRAMES} frames)")
    print(f"           Device: {device_norm} | Model: {model_name} | Conf: {args.conf}")

    all_detections = {}
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    t0 = time.perf_counter()
    last_print = t0

    while frame_idx <= VIDEO_END_FRAME:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx < VIDEO_START_FRAME:
            continue

        detections = detect_yolo(yolo_model, frame, conf=args.conf, device=device_norm)
        all_detections[frame_idx] = detections

        now = time.perf_counter()
        if now - last_print >= 5.0:
            elapsed = now - t0
            progress = (frame_idx - VIDEO_START_FRAME + 1) / FULL_VIDEO_FRAMES * 100
            eta = elapsed / (frame_idx - VIDEO_START_FRAME + 1) * (FULL_VIDEO_FRAMES - (frame_idx - VIDEO_START_FRAME + 1))
            print(f"  [{progress:5.1f}%] Frame {frame_idx}/{VIDEO_END_FRAME} | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
            last_print = now

    cap.release()
    elapsed_total = time.perf_counter() - t0
    print(f"[Inference] Done in {elapsed_total:.1f}s | "
          f"{len(all_detections)} frames cached")

    with open(cache_file, "wb") as f:
        pickle.dump(all_detections, f)
    print(f"[Cache] Saved to {cache_file}")

    return all_detections


# ---------------------------------------------------------------------------
# Section C: 4 Temporal Modes (Hysteresis x Inner-Core)
# ---------------------------------------------------------------------------

def run_section_c(args, all_detections, areas, inner_areas, num_slots,
                   frame_numbers, slot_state_gt, bbox_gt):
    """
    Chay 4 mode tren full video da suy luan.
    Chi thu thap predictions tai cac frame co GT.
    """
    eval_frames = sorted(set(frame_numbers) & set(f for f, s in slot_state_gt))
    if len(eval_frames) < 2:
        print("[Section C] Can it nhat 2 frame co GT. Bo qua.")
        return None

    gt_frame_set = set(eval_frames)
    warm_frames = min(args.warmup_frames, eval_frames[0])

    print(f"\n[Section C] Full video: {FULL_VIDEO_FRAMES} frames, "
          f"{len(eval_frames)} frames co GT")
    print(f"           Warm-up: {warm_frames} frames | "
          f"Hysteresis threshold: {args.hyst_threshold} frames")

    modes = [
        ("Hyst_OFF_Inner_OFF", False, False),
        ("Hyst_OFF_Inner_ON",  False, True),
        ("Hyst_ON_Inner_OFF",  True,  False),
        ("Hyst_ON_Inner_ON",   True,  True),
    ]

    results = {}
    for mode_name, use_hyst, use_inner in modes:
        print(f"  [{mode_name}] ...", end="", flush=True)
        t0 = time.perf_counter()

        state_true, state_pred = [], []
        flickers = defaultdict(lambda: None)
        flicker_trans = 0
        trans_count = 0
        hyst_counters = [0] * num_slots

        hyst_current = [
            slot_state_gt.get((eval_frames[0], s), "available")
            for s in range(num_slots)
        ]

        outside_tp = 0
        outside_fp = 0
        outside_fn = 0

        for frame_idx in range(VIDEO_START_FRAME, VIDEO_END_FRAME + 1):
            detections = all_detections.get(frame_idx, [])
            prev_states = list(hyst_current)

            if use_hyst:
                slot_states_pred = _hysteresis_step(
                    detections, areas, inner_areas, num_slots,
                    args.hyst_threshold, use_inner,
                    hyst_counters, hyst_current)
            else:
                slot_states_pred = infer_slot_state_instant(
                    detections, areas, inner_areas, num_slots, use_inner_core=use_inner)

            if use_hyst:
                for s in range(num_slots):
                    if prev_states[s] != hyst_current[s]:
                        trans_count += 1

            if frame_idx in gt_frame_set:
                for s in range(num_slots):
                    key = (frame_idx, s)
                    if key not in slot_state_gt:
                        continue
                    gt_state = slot_state_gt[key]
                    pred_state = hyst_current[s] if use_hyst else slot_states_pred[s]
                    state_true.append(gt_state)
                    state_pred.append(pred_state)

                    prev = flickers[s]
                    if prev is not None and prev != pred_state:
                        flicker_trans += 1
                    flickers[s] = pred_state

                gt_bboxes_outside = bbox_gt.get(frame_idx, [])
                pred_outside = detect_outside(detections, areas, num_slots)
                otp, ofp, ofn = match_detections_to_gt(
                    pred_outside, gt_bboxes_outside, args.iou_threshold)
                outside_tp += otp
                outside_fp += ofp
                outside_fn += ofn

        total = len(state_true)
        flick_rate = flicker_trans / total * 100 if total > 0 else 0.0
        correct = sum(1 for t, p in zip(state_true, state_pred) if t == p)
        accuracy = correct / total * 100 if total > 0 else 0.0

        slot_res, cm = compute_confusion_matrix(
            state_true, state_pred, PRED_STATES)

        total_outside = outside_tp + outside_fn
        out_p = outside_tp / (outside_tp + outside_fp) if (outside_tp + outside_fp) > 0 else 0.0
        out_r = outside_tp / (outside_tp + outside_fn) if (outside_tp + outside_fn) > 0 else 0.0
        out_f1 = 2 * out_p * out_r / (out_p + out_r) if (out_p + out_r) > 0 else 0.0

        results[mode_name] = {
            "mode": mode_name,
            "hysteresis": use_hyst,
            "inner_core": use_inner,
            "accuracy": round(accuracy, 2),
            "macro_precision": slot_res["_overall"]["macro_precision"],
            "macro_recall":    slot_res["_overall"]["macro_recall"],
            "macro_f1":        slot_res["_overall"]["macro_f1"],
            "macro_f1_3class": slot_res["_overall"].get("macro_f1_3class", 0.0),
            "weighted_f1_3class": slot_res["_overall"].get("weighted_f1_3class", 0.0),
            "flickering_rate": round(flick_rate, 4),
            "flickering_transitions": flicker_trans,
            "total_transitions": trans_count,
            "stability_score": round(correct / total * 100, 4) if total > 0 else 0.0,
            "per_class": {c: slot_res.get(c, {
                "precision": 0, "recall": 0, "f1": 0, "support": 0
            }) for c in PRED_STATES},
            "outside_metrics": {
                "precision": round(out_p * 100, 2),
                "recall":    round(out_r * 100, 2),
                "f1":        round(out_f1 * 100, 2),
                "tp": outside_tp, "fp": outside_fp, "fn": outside_fn,
                "support": total_outside,
            },
            "confusion_matrix": {"classes": PRED_STATES, "matrix": cm},
            "frames_evaluated": len(eval_frames),
        }
        elapsed = time.perf_counter() - t0
        print(f" done in {elapsed:.1f}s | Acc={results[mode_name]['accuracy']:.2f}% "
              f"F1={results[mode_name]['macro_f1']:.2f}% "
              f"WF1_3c={results[mode_name]['weighted_f1_3class']:.2f}% "
              f"FlickRate={results[mode_name]['flickering_rate']:.4f}%")

    print(f"\n  [Section C Summary]")
    print(f"  {'Mode':<25} {'Accuracy':>10} {'Macro-F1':>10} "
          f"{'WF1-3cls':>10} {'Out-F1':>10} {'FlickRate':>10} {'Trans':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for name, res in results.items():
        out_f1 = res.get("outside_metrics", {}).get("f1", 0.0)
        print(f"  {name:<25} {res['accuracy']:>9.2f}% "
              f"{res['macro_f1']:>9.2f}% {res['weighted_f1_3class']:>9.2f}% "
              f"{out_f1:>9.2f}% {res['flickering_rate']:>9.4f}% "
              f"{res['total_transitions']:>8}")

    return results


# ---------------------------------------------------------------------------
# Section E: State Update Latency
# ---------------------------------------------------------------------------

def find_gt_transitions(slot_state_gt, num_slots, frame_numbers):
    """Tim tat ca GT transitions tu slot_state_gt."""
    gt_by_slot = defaultdict(dict)
    for (f, s), state in slot_state_gt.items():
        gt_by_slot[s][f] = state

    gt_transitions = []
    for s in range(num_slots):
        frames_s = sorted(gt_by_slot[s].keys())
        for i in range(1, len(frames_s)):
            prev_f, prev_st = frames_s[i-1], gt_by_slot[s][frames_s[i-1]]
            curr_f, curr_st = frames_s[i], gt_by_slot[s][frames_s[i]]
            if prev_st != curr_st:
                gt_transitions.append({
                    "slot": s,
                    "frame": curr_f,
                    "from_state": prev_st,
                    "to_state": curr_st,
                })
    return gt_transitions


def find_pred_transitions(pred_states_by_frame, num_slots, eval_frames):
    """Tim cac Pred transition tu prediction log."""
    pred_transitions = []
    sorted_frames = sorted(pred_states_by_frame.keys())
    if not sorted_frames:
        return pred_transitions

    for fi in range(1, len(sorted_frames)):
        prev_f, curr_f = sorted_frames[fi-1], sorted_frames[fi]
        prev_states = pred_states_by_frame[prev_f]
        curr_states = pred_states_by_frame[curr_f]
        for s in range(num_slots):
            if prev_states.get(s) != curr_states.get(s):
                pred_transitions.append({
                    "frame": curr_f,
                    "slot": s,
                    "from_state": prev_states.get(s, "available"),
                    "to_state": curr_states.get(s, "available"),
                })
    return pred_transitions


def match_transitions(gt_transitions, pred_transitions, num_slots):
    """Match GT vs Pred transitions, tinh latency."""
    gt_by_slot = defaultdict(list)
    for t in gt_transitions:
        gt_by_slot[t["slot"]].append(t)

    pred_by_slot = defaultdict(list)
    for t in pred_transitions:
        pred_by_slot[t["slot"]].append(t)

    latencies = []
    matched_gt = set()

    for s in range(num_slots):
        gt_list = sorted(gt_by_slot[s], key=lambda x: x["frame"])
        pred_list = sorted(pred_by_slot[s], key=lambda x: x["frame"])

        for gt_t in gt_list:
            for pred_t in pred_list:
                if (gt_t["to_state"] == pred_t["to_state"] and
                        pred_t["frame"] >= gt_t["frame"]):
                    latency_frames = pred_t["frame"] - gt_t["frame"]
                    latencies.append({
                        "slot": s,
                        "gt_frame": gt_t["frame"],
                        "pred_frame": pred_t["frame"],
                        "from_state": gt_t["from_state"],
                        "to_state": gt_t["to_state"],
                        "latency_frames": latency_frames,
                        "latency_sec": round(latency_frames / DEFAULT_VIDEO_FPS, 3),
                    })
                    matched_gt.add((s, gt_t["frame"]))
                    break

    return latencies, matched_gt


def run_section_e(args, all_detections, areas, inner_areas, num_slots,
                  frame_numbers, slot_state_gt):
    """
    Do tre cap nhat trang thai: ON_ON vs ON_OFF.
    """
    eval_frames = sorted(set(frame_numbers) & set(f for f, s in slot_state_gt))
    if len(eval_frames) < 2:
        print("[Section E] Can it nhat 2 frame co GT. Bo qua.")
        return None

    gt_transitions = find_gt_transitions(slot_state_gt, num_slots, frame_numbers)
    if not gt_transitions:
        print("[Section E] Khong co transition nao trong GT. Bo qua.")
        return None

    gt_frame_set = set(eval_frames)

    print(f"\n[Section E] GT transitions: {len(gt_transitions)} | "
          f"Eval frames: {len(eval_frames)} | "
          f"Range: {eval_frames[0]}–{eval_frames[-1]}")
    print(f"           Hysteresis threshold: {args.hyst_threshold} frames")

    sub_experiments = {
        "ON_ON":  (True,  True),
        "ON_OFF": (True,  False),
    }

    results = {}
    for sub_name, (use_hyst, use_inner) in sub_experiments.items():
        print(f"  [{sub_name}] ...", end="", flush=True)
        t0 = time.perf_counter()

        hyst_counters = [0] * num_slots
        hyst_current = [
            slot_state_gt.get((eval_frames[0], s), "available")
            for s in range(num_slots)
        ]

        pred_states_by_frame = {}

        for frame_idx in range(VIDEO_START_FRAME, VIDEO_END_FRAME + 1):
            detections = all_detections.get(frame_idx, [])

            if use_hyst:
                _hysteresis_step(
                    detections, areas, inner_areas, num_slots,
                    args.hyst_threshold, use_inner,
                    hyst_counters, hyst_current)
            else:
                infer_slot_state_instant(
                    detections, areas, inner_areas, num_slots, use_inner_core=use_inner)

            if frame_idx in gt_frame_set:
                pred_states_by_frame[frame_idx] = dict(enumerate(hyst_current if use_hyst
                                                                  else infer_slot_state_instant(
                                                                      detections, areas, inner_areas,
                                                                      num_slots, use_inner_core=use_inner)))

        pred_transitions = find_pred_transitions(
            pred_states_by_frame, num_slots, eval_frames)
        latencies, matched_gt = match_transitions(
            gt_transitions, pred_transitions, num_slots)

        if latencies:
            lat_vals = [l["latency_frames"] for l in latencies]
            lat_arr = np.array(lat_vals)
            match_pct = len(matched_gt) / len(gt_transitions) * 100 if gt_transitions else 0

            lat_by_transition = defaultdict(list)
            for l in latencies:
                key = f"{l['from_state']} -> {l['to_state']}"
                lat_by_transition[key].append(l["latency_frames"])

            lat_by_slot = defaultdict(list)
            for l in latencies:
                lat_by_slot[l["slot"]].append(l["latency_frames"])

            lat_dist_bins = [
                ("0-30",    lambda x: 0 <= x < 30),
                ("30-60",   lambda x: 30 <= x < 60),
                ("60-90",   lambda x: 60 <= x < 90),
                ("90-180",  lambda x: 90 <= x < 180),
                ("180-300", lambda x: 180 <= x < 300),
                ("300+",    lambda x: x >= 300),
            ]
            dist_bins = {}
            for bin_name, cond in lat_dist_bins:
                count = sum(1 for v in lat_vals if cond(v))
                dist_bins[bin_name] = {
                    "count": count,
                    "pct": round(count / len(lat_vals) * 100, 2) if lat_vals else 0,
                }

            results[sub_name] = {
                "gt_transitions": len(gt_transitions),
                "pred_transitions": len(pred_transitions),
                "matched_transitions": len(matched_gt),
                "match_pct": round(match_pct, 2),
                "min_frames": int(np.min(lat_arr)),
                "max_frames": int(np.max(lat_arr)),
                "mean_frames": round(float(np.mean(lat_arr)), 2),
                "median_frames": round(float(np.median(lat_arr)), 2),
                "std_frames": round(float(np.std(lat_arr)), 2),
                "p95_frames": int(np.percentile(lat_arr, 95)),
                "p99_frames": int(np.percentile(lat_arr, 99)),
                "min_sec": round(float(np.min(lat_arr)) / DEFAULT_VIDEO_FPS, 3),
                "max_sec": round(float(np.max(lat_arr)) / DEFAULT_VIDEO_FPS, 3),
                "mean_sec": round(float(np.mean(lat_arr)) / DEFAULT_VIDEO_FPS, 3),
                "median_sec": round(float(np.median(lat_arr)) / DEFAULT_VIDEO_FPS, 3),
                "p95_sec": round(float(np.percentile(lat_arr, 95)) / DEFAULT_VIDEO_FPS, 3),
                "p99_sec": round(float(np.percentile(lat_arr, 99)) / DEFAULT_VIDEO_FPS, 3),
                "latency_by_transition": {
                    k: {
                        "count": len(v),
                        "mean_frames": round(float(np.mean(v)), 2),
                        "median_frames": round(float(np.median(v)), 2),
                        "min_frames": int(np.min(v)),
                        "max_frames": int(np.max(v)),
                    }
                    for k, v in lat_by_transition.items()
                },
                "latency_by_slot": {
                    s: {
                        "count": len(v),
                        "mean_frames": round(float(np.mean(v)), 2),
                        "median_frames": round(float(np.median(v)), 2),
                    }
                    for s, v in lat_by_slot.items()
                },
                "latency_distribution": dist_bins,
                "below_threshold_pct": round(
                    sum(1 for v in lat_vals if v < args.hyst_threshold) / len(lat_vals) * 100
                    if lat_vals else 0, 2),
                "hysteresis_threshold": args.hyst_threshold,
                "fps": DEFAULT_VIDEO_FPS,
            }
        else:
            results[sub_name] = {
                "gt_transitions": len(gt_transitions),
                "pred_transitions": len(pred_transitions),
                "matched_transitions": 0,
                "match_pct": 0.0,
                "error": "No matched transitions found",
            }

        elapsed = time.perf_counter() - t0
        matched = results[sub_name].get("matched_transitions", 0)
        mean_f = results[sub_name].get("mean_frames", "N/A")
        print(f" done in {elapsed:.1f}s | Matched: {matched}/{len(gt_transitions)} "
              f"({results[sub_name].get('match_pct', 0):.1f}%) | "
              f"Mean latency: {mean_f} frames")

    print(f"\n  [Section E Latency Summary]")
    print(f"  {'Mode':>10} {'GT':>6} {'Pred':>6} {'Match%':>8} "
          f"{'Mean':>8} {'Median':>8} {'P95':>6} "
          f"{'<Thresh%':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*8} {'-'*8} "
          f"{'-'*8} {'-'*6} {'-'*10}")
    for name, res in results.items():
        print(f"  {name:>10} {res['gt_transitions']:>6} "
              f"{res['pred_transitions']:>6} "
              f"{res.get('match_pct', 0):>7.1f}% "
              f"{res.get('mean_frames', 'N/A'):>8} "
              f"{res.get('median_frames', 'N/A'):>8} "
              f"{res.get('p95_frames', 'N/A'):>6} "
              f"{res.get('below_threshold_pct', 0):>9.1f}%")

    return results


# ---------------------------------------------------------------------------
# Output formatters (ASCII)
# ---------------------------------------------------------------------------

def fmt_section_c_ascii(results, hyst_threshold):
    W = 72
    out = []
    out.append("")
    out.append("=" * W)
    out.append("SECTION C: 4 Che Do Thoi Gian (Hysteresis x Inner-Core)")
    out.append("=" * W)
    out.append(f"  Video: 41363 frames @ 30 FPS, 1000 frames co GT")
    out.append(f"  Hysteresis threshold: {hyst_threshold} frames ({hyst_threshold/30:.1f}s)")
    out.append("")

    out.append("  [So sanh 4 Mode — Main Metrics]")
    out.append(f"  {'Mode':<25} {'Accuracy':>10} {'Macro-F1':>10} "
               f"{'WF1-3cls':>10} {'Out-F1':>10} {'FlickRate':>10}")
    out.append("  " + "-" * W)
    for name, res in sorted(results.items()):
        out_f1 = res.get("outside_metrics", {}).get("f1", 0.0)
        out.append(f"  {name:<25} {res['accuracy']:>9.2f}% "
                   f"{res['macro_f1']:>9.2f}% {res['weighted_f1_3class']:>9.2f}% "
                   f"{out_f1:>9.2f}% {res['flickering_rate']:>9.4f}%")

    out.append("")
    out.append("  [Per-Class F1 Score — available / occupied / overlapping]")
    out.append(f"  {'Mode':<25} {'available':>12} {'occupied':>12} "
               f"{'overlapping':>12} {'outside':>12}")
    out.append("  " + "-" * W)
    core_classes = ["available", "occupied", "overlapping", "outside"]
    for name, res in sorted(results.items()):
        pc = res.get("per_class", {})
        vals = "".join(f"{pc.get(c,{}).get('f1',0):>11.1f}%" for c in core_classes)
        out.append(f"  {name:<25}{vals}")

    out.append("")
    out.append("  [Weighted-F1 3-Class (available/occupied/overlapping)]")
    out.append("  " + "-" * W)
    for name, res in sorted(results.items()):
        wf1 = res.get("weighted_f1_3class", 0.0)
        mf1 = res.get("macro_f1_3class", 0.0)
        out.append(f"    {name:<25} Weighted-F1={wf1:.2f}%  Macro-F1(3c)={mf1:.2f}%")

    out.append("")
    out.append("  [Outside Detection (bbox IoU matching)]")
    out.append("  " + "-" * W)
    for name, res in sorted(results.items()):
        om = res.get("outside_metrics", {})
        out.append(f"    {name:<25} P={om.get('precision',0):.1f}%  "
                  f"R={om.get('recall',0):.1f}%  F1={om.get('f1',0):.1f}%  "
                  f"TP={om.get('tp',0)} FP={om.get('fp',0)} FN={om.get('fn',0)}")

    out.append("")
    out.append("  [Ghi chu ve hysteresis]")
    out.append("  - Hyst_ON_Inner_ON co the yeu hon Hyst_OFF_Inner_ON o class "
               "overlapping vi hysteresis threshold={}f lam tre tran")
    out.append(f"    gian overlapping (xe phai giu trang thai moi trong {hyst_threshold} frames")
    out.append("    lien tiep truoc khi chuyen trang thai, gay mat cac frame overlapping).")
    out.append("  - Khuyen nghi: giam hyst_threshold hoac tach danh gia overlapping rieng.")

    out.append("")
    out.append("  [Confusion Matrix — OFF_ON vs ON_ON (rows=GT, cols=Pred)]")
    off_on = results.get("Hyst_OFF_Inner_ON", {})
    on_on = results.get("Hyst_ON_Inner_ON", {})
    if off_on and on_on:
        for label, res in [("Hyst_OFF_Inner_ON", off_on), ("Hyst_ON_Inner_ON", on_on)]:
            cm = res.get("confusion_matrix", {})
            out.append(f"  {label}:")
            out.append("  " + "".join(f"{'':>8}{c:>12}" for c in PRED_STATES))
            for i, row_c in enumerate(PRED_STATES):
                row_vals = cm.get("matrix", [[]] * len(PRED_STATES))
                out.append("  " + f"{row_c:<{8+12}}"
                           + "".join(f"{v:>12}" for v in row_vals[i]))

    out.append("=" * W)
    return "\n".join(out)


def fmt_section_e_ascii(results, hyst_threshold):
    W = 72
    out = []
    out.append("")
    out.append("=" * W)
    out.append("SECTION E: Do Tre Cap Nhat Trang Thai (State Update Latency)")
    out.append("=" * W)
    out.append(f"  Chay tren video lien tuc: frame {VIDEO_START_FRAME}-{VIDEO_END_FRAME} "
               f"({FULL_VIDEO_FRAMES} frames)")
    out.append(f"  Hysteresis threshold: {hyst_threshold} frames ({hyst_threshold/30:.1f}s)")
    out.append("")

    if not results:
        out.append("  No results.")
        return "\n".join(out)

    modes = sorted(results.keys())
    out.append("  [So sanh Latency: Inner_ON vs Inner_OFF]")
    out.append("  " + "-" * W)
    out.append(f"  {'':>20} " + "".join(f"{m:>15}" for m in modes))
    out.append("  " + "-" * W)

    for key in ["GT trans", "Pred trans", "Match%", "Min", "Max",
                "Mean", "Median", "Std", "P95", "P99"]:
        field_map = {
            "GT trans": "gt_transitions",
            "Pred trans": "pred_transitions",
            "Match%": "match_pct",
            "Min": "min_frames",
            "Max": "max_frames",
            "Mean": "mean_frames",
            "Median": "median_frames",
            "Std": "std_frames",
            "P95": "p95_frames",
            "P99": "p99_frames",
        }
        field = field_map.get(key, "")
        vals = []
        for m in modes:
            v = results[m].get(field, "N/A")
            vals.append(f"{v:>15}")
        out.append(f"  {key:>20} " + "".join(vals))

    out.append("")
    out.append("  [Thoi gian (giay)]")
    out.append("  " + "-" * W)
    for key, field in [("Mean (s)", "mean_sec"), ("Median (s)", "median_sec"),
                        ("P95 (s)", "p95_sec")]:
        vals = []
        for m in modes:
            v = results[m].get(field, "N/A")
            vals.append(f"{v:>15}")
        out.append(f"  {key:>20} " + "".join(vals))

    out.append("")
    out.append("  [% duoi threshold = {} frames]".format(hyst_threshold))
    for m in modes:
        v = results[m].get("below_threshold_pct", 0)
        out.append(f"    {m}: {v:.1f}%")

    if modes and "latency_by_transition" in results[modes[0]]:
        out.append("")
        out.append("  [Latency theo loai Transition]")
        out.append("  " + "-" * W)
        out.append(f"  {'Transition':<25} {'Count':>6} {'Mean(f)':>10} {'Median(f)':>10}")
        out.append("  " + "-" * W)
        lat_by_trans = results[modes[0]].get("latency_by_transition", {})
        for trans, data in sorted(lat_by_trans.items()):
            out.append(f"  {trans:<25} {data['count']:>6} "
                       f"{data['mean_frames']:>10} {data['median_frames']:>10}")

    if modes and "latency_distribution" in results[modes[0]]:
        out.append("")
        out.append(f"  [Latency Distribution — {modes[0]}]")
        out.append("  " + "-" * W)
        dist = results[modes[0]].get("latency_distribution", {})
        for bin_name, data in dist.items():
            out.append(f"    {bin_name:>10}: {data['count']:>5} ({data['pct']:.1f}%)")

    out.append("=" * W)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Matplotlib charts
# ---------------------------------------------------------------------------

def _get_agg(results, key):
    vals = []
    for name in sorted(results.keys()):
        v = results[name].get(key, 0)
        vals.append(v)
    return vals


def plot_section_c_charts(results, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[Warn] matplotlib not available. Skipping Section C charts.")
        return

    mode_labels = sorted(results.keys())
    mode_labels_short = [m.replace("Hyst_", "H_").replace("Inner_", "I_")
                          for m in mode_labels]

    acc = _get_agg(results, "accuracy")
    f1 = _get_agg(results, "macro_f1")
    wf1 = _get_agg(results, "weighted_f1_3class")
    flick = _get_agg(results, "flickering_rate")
    trans = _get_agg(results, "total_transitions")
    out_f1 = [results[m].get("outside_metrics", {}).get("f1", 0.0) for m in mode_labels]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Section C: 4 Che Do Thoi Gian (Hysteresis x Inner-Core)",
                 fontsize=14, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    ax = axes[0, 0]
    bars = ax.bar(mode_labels_short, acc, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Accuracy (%)", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[0, 1]
    x = np.arange(len(mode_labels_short))
    width = 0.35
    bars1 = ax.bar(x - width/2, f1, width, label="Macro-F1 (4-class)", color="#9b59b6",
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, wf1, width, label="Weighted-F1 (3-class)", color="#27ae60",
                   edgecolor="black", linewidth=0.5)
    ax.set_title("F1 Score Comparison", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels_short, fontsize=8)
    ax.legend(fontsize=8)
    for bar, val in zip(bars1, f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, wf1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1, 0]
    bars = ax.bar(mode_labels_short, flick, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Flickering Rate (%)", fontweight="bold")
    ax.set_ylabel("Flickering Rate (%)")
    for bar, val in zip(bars, flick):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(flick)*0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1, 1]
    bars = ax.bar(mode_labels_short, out_f1, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Outside Detection F1 (bbox IoU)", fontweight="bold")
    ax.set_ylabel("F1 (%)")
    for bar, val in zip(bars, out_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    path = output_dir / "section_c_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mode_labels_short))
    width = 0.2
    classes_plot = PRED_STATES
    class_colors = {"available": "#2ecc71", "occupied": "#e74c3c",
                    "overlapping": "#f39c12", "outside": "#95a5a6"}

    for i, cls in enumerate(classes_plot):
        vals = [results[m].get("per_class", {}).get(cls, {}).get("f1", 0)
                for m in mode_labels]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=cls, color=class_colors.get(cls, "gray"),
                      edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Mode")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("Section C: Per-Class F1 Score — 4 Modes", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels_short, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "section_c_grouped_per_class_f1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_section_e_charts(results, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warn] matplotlib not available. Skipping Section E charts.")
        return

    if not results:
        return

    modes = sorted(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Section E: State Update Latency — ON_ON vs ON_OFF",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    lat_data = []
    labels = []
    for m in modes:
        if "mean_frames" in results[m]:
            lat_data.append(results[m]["mean_frames"])
            labels.append(m)
    if lat_data:
        colors = ["#3498db", "#e74c3c"]
        bars = ax.bar(labels, lat_data, color=colors[:len(labels)],
                      edgecolor="black", linewidth=0.5)
        ax.set_title("Mean Latency (frames)", fontweight="bold")
        ax.set_ylabel("Frames")
        for bar, val in zip(bars, lat_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    lat_by_trans = results[modes[0]].get("latency_by_transition", {}) if modes else {}
    if lat_by_trans:
        trans_labels = list(lat_by_trans.keys())
        mean_vals = [lat_by_trans[t]["mean_frames"] for t in trans_labels]
        median_vals = [lat_by_trans[t]["median_frames"] for t in trans_labels]
        x = np.arange(len(trans_labels))
        width = 0.35
        ax.bar(x - width/2, mean_vals, width, label="Mean", color="#3498db", edgecolor="black")
        ax.bar(x + width/2, median_vals, width, label="Median", color="#2ecc71", edgecolor="black")
        ax.set_title("Latency by Transition Type", fontweight="bold")
        ax.set_ylabel("Frames")
        ax.set_xticks(x)
        ax.set_xticklabels(trans_labels, rotation=20, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "section_e_latency_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")

    if modes and "latency_distribution" in results[modes[0]]:
        fig, ax = plt.subplots(figsize=(10, 5))
        dist = results[modes[0]]["latency_distribution"]
        bin_names = list(dist.keys())
        counts = [dist[b]["count"] for b in bin_names]
        pcts = [dist[b]["pct"] for b in bin_names]

        bars = ax.bar(bin_names, counts, color="#9b59b6", edgecolor="black", linewidth=0.5)
        ax.set_title(f"Latency Distribution — {modes[0]}", fontweight="bold")
        ax.set_xlabel("Latency (frames)")
        ax.set_ylabel("Count")
        for bar, cnt, pct in zip(bars, counts, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = output_dir / "section_e_latency_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {path}")


# ---------------------------------------------------------------------------
# Output saves (JSON + CSV)
# ---------------------------------------------------------------------------

def save_outputs(results_c, results_e, output_dir, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if results_c:
        full_json = {
            "config": {
                "video": resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/")),
                "slot_gt": resolve_path("annotations", args.slot_gt, DEFAULT_SLOT_GT),
                "frame_list": resolve_path("annotations", args.frame_list, DEFAULT_FRAME_LIST),
                "yolo_confidence": args.conf,
                "model_size": args.model_size,
                "hysteresis_threshold": args.hyst_threshold,
                "sections": args.section,
                "fps": DEFAULT_VIDEO_FPS,
                "full_video_frames": FULL_VIDEO_FRAMES,
                "video_range": f"{VIDEO_START_FRAME}-{VIDEO_END_FRAME}",
            },
            "section_c": results_c,
        }
        with open(output_dir / "section_c_temporal_comparison_full.json", "w",
                  encoding="utf-8") as f:
            json.dump(full_json, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {output_dir / 'section_c_temporal_comparison_full.json'}")

        with open(output_dir / "section_c_temporal_comparison.csv", "w",
                  newline="", encoding="utf-8") as f:
            cols = ["mode", "accuracy", "macro_precision", "macro_recall",
                    "macro_f1", "macro_f1_3class", "weighted_f1_3class",
                    "flickering_rate", "total_transitions", "frames_evaluated"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for name, res in sorted(results_c.items()):
                om = res.get("outside_metrics", {})
                row = {c: res.get(c, "") for c in cols}
                row["macro_f1_3class"] = res.get("macro_f1_3class", "")
                row["weighted_f1_3class"] = res.get("weighted_f1_3class", "")
                w.writerow(row)
        print(f"[Saved] {output_dir / 'section_c_temporal_comparison.csv'}")

        with open(output_dir / "section_c_per_class_f1.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mode"] + PRED_STATES)
            for name, res in sorted(results_c.items()):
                pc = res.get("per_class", {})
                row = [name] + [pc.get(c, {}).get("f1", 0) for c in PRED_STATES]
                w.writerow(row)
        print(f"[Saved] {output_dir / 'section_c_per_class_f1.csv'}")

        with open(output_dir / "section_c_outside_metrics.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mode", "precision", "recall", "f1",
                        "tp", "fp", "fn", "support"])
            for name, res in sorted(results_c.items()):
                om = res.get("outside_metrics", {})
                w.writerow([
                    name,
                    om.get("precision", 0),
                    om.get("recall", 0),
                    om.get("f1", 0),
                    om.get("tp", 0),
                    om.get("fp", 0),
                    om.get("fn", 0),
                    om.get("support", 0),
                ])
        print(f"[Saved] {output_dir / 'section_c_outside_metrics.csv'}")

    if results_e:
        full_json_e = {
            "config": {
                "video": resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/")),
                "hysteresis_threshold": args.hyst_threshold,
                "fps": DEFAULT_VIDEO_FPS,
                "video_range": f"{VIDEO_START_FRAME}-{VIDEO_END_FRAME}",
            },
            "section_e": results_e,
        }
        with open(output_dir / "section_e_latency_full.json", "w",
                  encoding="utf-8") as f:
            json.dump(full_json_e, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {output_dir / 'section_e_latency_full.json'}")

        with open(output_dir / "section_e_latency_summary.csv", "w",
                  newline="", encoding="utf-8") as f:
            cols = ["mode", "gt_transitions", "pred_transitions", "matched_transitions",
                    "match_pct", "min_frames", "max_frames", "mean_frames",
                    "median_frames", "std_frames", "p95_frames", "p99_frames",
                    "mean_sec", "median_sec", "below_threshold_pct"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for name, res in sorted(results_e.items()):
                w.writerow({c: res.get(c, "") for c in cols})
        print(f"[Saved] {output_dir / 'section_e_latency_summary.csv'}")

        with open(output_dir / "section_e_latency_by_transition_type.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mode", "transition", "count", "mean_frames",
                        "median_frames", "min_frames", "max_frames"])
            for m, res in sorted(results_e.items()):
                lat_by_t = res.get("latency_by_transition", {})
                for trans, data in sorted(lat_by_t.items()):
                    w.writerow([m, trans, data["count"], data["mean_frames"],
                                data["median_frames"], data["min_frames"], data["max_frames"]])
        print(f"[Saved] {output_dir / 'section_e_latency_by_transition_type.csv'}")

        with open(output_dir / "section_e_latency_per_slot.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mode", "slot", "count", "mean_frames", "median_frames"])
            for m, res in sorted(results_e.items()):
                lat_by_s = res.get("latency_by_slot", {})
                for slot, data in sorted(lat_by_s.items()):
                    w.writerow([m, slot, data["count"], data["mean_frames"],
                                data["median_frames"]])
        print(f"[Saved] {output_dir / 'section_e_latency_per_slot.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args is None:
        print("[Error] Failed to parse arguments.")
        sys.exit(1)

    sections = args.section.lower()
    if "c" not in sections and "e" not in sections:
        print("[Error] --section must contain 'c' or 'e' (e.g. 'ce', 'c', 'e')")
        sys.exit(1)

    print("=" * 72)
    print("  eval_temporal_analysis.py — Section C + E Temporal Evaluation")
    print("=" * 72)
    print(f"  Sections: {sections}")
    print(f"  Video:    {args.video}")
    print(f"  Device:   {args.device}")
    print(f"  Conf:     {args.conf}")
    print(f"  Hyst thresh: {args.hyst_threshold} frames ({args.hyst_threshold/30:.1f}s)")
    print(f"  Output:   {args.output}")

    slot_gt_path = resolve_path("annotations", args.slot_gt, DEFAULT_SLOT_GT)
    bbox_gt_path = resolve_path("annotations", args.bbox_gt, DEFAULT_BBOX_GT)
    frame_list_path = resolve_path("annotations", args.frame_list, DEFAULT_FRAME_LIST)

    print(f"\n[Loading data]")
    slot_state_gt = load_slot_state_gt(slot_gt_path)
    bbox_gt = load_bbox_gt(bbox_gt_path)
    frame_numbers = load_frame_numbers(frame_list_path)
    areas, inner_areas = load_parking_zones(args.video)
    num_slots = len(areas)

    print(f"  GT slot states: {len(slot_state_gt)} entries")
    print(f"  GT bbox outside: {sum(len(v) for v in bbox_gt.values())} entries")
    print(f"  Frame numbers: {len(frame_numbers)} frames (range: "
          f"{frame_numbers[0]}–{frame_numbers[-1]})")
    print(f"  Parking slots: {num_slots}")

    print(f"\n[Init YOLO]")
    yolo_model, device_norm = init_yolo(args.device, args.model_size)
    if yolo_model is None:
        print("[Error] Cannot init YOLO. Exiting.")
        sys.exit(1)

    t_total = time.perf_counter()

    print(f"\n[Step 1/2] Running full video YOLO inference (once)...")
    all_detections = run_full_video_inference(
        args, yolo_model, areas, inner_areas, num_slots)

    results_c = None
    results_e = None

    if "c" in sections:
        t0 = time.perf_counter()
        results_c = run_section_c(
            args, all_detections, areas, inner_areas, num_slots,
            frame_numbers, slot_state_gt, bbox_gt)
        print(f"[Section C] Done in {time.perf_counter()-t0:.1f}s")
        if results_c:
            print(fmt_section_c_ascii(results_c, args.hyst_threshold))

    if "e" in sections:
        t0 = time.perf_counter()
        results_e = run_section_e(
            args, all_detections, areas, inner_areas, num_slots,
            frame_numbers, slot_state_gt)
        print(f"[Section E] Done in {time.perf_counter()-t0:.1f}s")
        if results_e:
            print(fmt_section_e_ascii(results_e, args.hyst_threshold))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 2/2] Saving outputs to {output_dir}/")
    save_outputs(results_c, results_e, output_dir, args)

    if args.save_charts and (results_c or results_e):
        print(f"\n[Charts]")
        if results_c:
            plot_section_c_charts(results_c, output_dir)
        if results_e:
            plot_section_e_charts(results_e, output_dir)

    total_elapsed = time.perf_counter() - t_total
    print(f"\n{'='*72}")
    print(f"  DONE. Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
