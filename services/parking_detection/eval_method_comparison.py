#!/usr/bin/env python3
"""
eval_method_comparison.py

So sanh hieu suat phat hien cua 4 phuong phap suy luan cho bai toan do tom tat
trang thai o do xe:

    1. Rectangular ROI
       - Kiem tra bbox centroid nam trong hinh chu nhat bao (axis-aligned)
       - Khong co Inner-Core, khong co Hysteresis

    2. Polygon Only
       - Kiem tra bbox centroid nam trong polygon cua o do
       - Khong co Inner-Core, khong co Hysteresis

    3. Polygon + Inner-Core
       - Kiem tra bbox centroid nam trong polygon va inner-zone
       - Co Inner-Core, khong co Hysteresis

    4. Proposed Framework
       - Kiem tra bbox centroid nam trong polygon va inner-zone
       - Co Inner-Core + Temporal Hysteresis (threshold = 45 frames)

Tat ca 4 phuong phap deu chay YOLO inference 1 lan duy nhat tren 1000 frame co
ground truth, roi tai su dung ket qua detection cho tat ca.

Usage:
    python eval_method_comparison.py --device cuda
    python eval_method_comparison.py --device cpu --output eval_results/method_comparison
    python eval_method_comparison.py --use-cache  # Chi doc tu cache, khong chay YOLO
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

DEFAULT_VIDEO       = "static/video/CAM_PARKING.mp4"
DEFAULT_SLOT_GT     = "annotations/CAM_PARKING_slot_state_gt.csv"
DEFAULT_BBOX_GT     = "annotations/CAM_PARKING_bbox_outside_gt.csv"
DEFAULT_FRAME_LIST  = "annotations/frame_numbers.txt"
DEFAULT_OUTPUT      = "eval_results/method_comparison"

STATE_CLASSES      = ["available", "occupied", "overlapping"]
PRED_STATES        = ["available", "occupied", "overlapping"]
VEHICLE_CLASS_IDS  = [2, 5, 7]
DEFAULT_HYST_THRESH = 45
DEFAULT_VIDEO_FPS   = 30.0
DEFAULT_CONF        = 0.30
VIDEO_START_FRAME   = 6
VIDEO_END_FRAME     = 41319


# ---------------------------------------------------------------------------
# Helpers
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
        description="Compare 4 parking detection methods: Rectangular ROI, "
                    "Polygon Only, Polygon+Inner-Core, Proposed Framework.")
    p.add_argument("--video",       default=DEFAULT_VIDEO,  help="Path to video")
    p.add_argument("--slot-gt",    default=DEFAULT_SLOT_GT,  help="Path to slot state GT CSV")
    p.add_argument("--bbox-gt",    default=DEFAULT_BBOX_GT,  help="Path to outside bbox GT CSV")
    p.add_argument("--frame-list", default=DEFAULT_FRAME_LIST, help="Path to frame numbers list")
    p.add_argument("--output",     default=DEFAULT_OUTPUT,  help="Output directory")
    p.add_argument("--conf",       type=float, default=DEFAULT_CONF,
                   help="YOLO confidence threshold (default: 0.30)")
    p.add_argument("--model-size", default="l",
                   choices=["n", "s", "m", "l"],
                   help="YOLO model size (default: l)")
    p.add_argument("--device",     default="cpu",
                   help="'cuda' (or '0') for GPU, 'cpu' for CPU (default: cpu)")
    p.add_argument("--hyst-threshold", type=int, default=DEFAULT_HYST_THRESH,
                   help="Hysteresis threshold in frames (default: 45)")
    p.add_argument("--warmup-frames",  type=int, default=100,
                   help="Warm-up frames before GT period (default: 100)")
    p.add_argument("--iou-threshold",  type=float, default=0.3,
                   help="IoU threshold for outside detection (default: 0.30)")
    p.add_argument("--no-cache",   action="store_true",
                   help="Force re-run YOLO inference (ignore cache)")
    p.add_argument("--use-cache",  action="store_true",
                   help="Only read from cache, skip YOLO inference")
    return p.parse_args()


def _normalize_device(device):
    d = device.lower().strip()
    if d == "cuda":
        return "0"
    if d == "cpu":
        return "cpu"
    if d.isdigit():
        return d
    return device


# ---------------------------------------------------------------------------
# Parking zones
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
    _FAR_SLOTS      = set(range(0, 14)) | {17, 18}
    _FAR_SLOT_SHRINK = 0.35
    inner_zones = []
    for idx, area in enumerate(areas):
        slot_shrink = _FAR_SLOT_SHRINK if idx in _FAR_SLOTS else shrink_percentage
        points = np.array(area, dtype=np.float32)
        centroid = np.mean(points, axis=0)
        inner_points = []
        for pt in points:
            vec = pt - centroid
            new_pt = centroid + vec * (1 - slot_shrink)
            inner_points.append(tuple(new_pt.astype(int)))
        inner_zones.append(inner_points)
    return inner_zones


def make_slot_rects(areas):
    """Build axis-aligned bounding boxes for each slot polygon."""
    rects = []
    for area in areas:
        xs = [p[0] for p in area]
        ys = [p[1] for p in area]
        rects.append((min(xs), min(ys), max(xs), max(ys)))
    return rects


def scale_areas(raw_areas, sx, sy):
    return [[(int(x * sx), int(y * sy)) for x, y in area] for area in raw_areas]


def load_parking_zones(video_path=None):
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

    raw_areas  = define_parking_areas()
    raw_inner   = create_inner_zones(raw_areas)
    raw_rects   = make_slot_rects(raw_areas)
    areas       = scale_areas(raw_areas, sx, sy)
    inner_areas = scale_areas(raw_inner, sx, sy)
    slot_rects  = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                   for (x1,y1,x2,y2) in raw_rects]

    print(f"[Parking] {len(areas)} slots, baseline 1020x500 -> scaled to {w}x{h} "
          f"(sx={sx:.3f}, sy={sy:.3f})")
    return areas, inner_areas, slot_rects


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
    if not Path(path).exists():
        return dict(gt)
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
# 4 Method classifiers
# ---------------------------------------------------------------------------

def classify_rectangular_roi(detections, slot_rects, num_slots):
    """Method 1: Rectangular ROI — centroid inside axis-aligned bounding box."""
    states = ["available"] * num_slots
    for det in detections:
        cx, cy = det["cx"], det["cy"]
        for s in range(num_slots):
            x1, y1, x2, y2 = slot_rects[s]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                states[s] = "occupied"
    return states


def classify_polygon_only(detections, areas, num_slots):
    """Method 2: Polygon Only — centroid inside polygon, no inner-core."""
    states = ["available"] * num_slots
    for det in detections:
        cx, cy = det["cx"], det["cy"]
        for s in range(num_slots):
            if cv2.pointPolygonTest(
                    np.array(areas[s], np.int32), (cx, cy), False) >= 0:
                states[s] = "occupied"
    return states


def classify_polygon_inner_core(detections, areas, inner_areas, num_slots):
    """Method 3: Polygon + Inner-Core — centroid in outer polygon, check inner zone."""
    states = ["available"] * num_slots
    for det in detections:
        cx, cy = det["cx"], det["cy"]
        for s in range(num_slots):
            in_outer = cv2.pointPolygonTest(
                np.array(areas[s], np.int32), (cx, cy), False) >= 0
            if not in_outer:
                continue
            in_inner = cv2.pointPolygonTest(
                np.array(inner_areas[s], np.int32), (cx, cy), False) >= 0
            state = "overlapping" if not in_inner else "occupied"
            states[s] = state
    return states


def classify_proposed(detections, areas, inner_areas, num_slots,
                       threshold, counters, current_states):
    """Method 4: Proposed Framework — matches actual project logic.

    State machine mirrors workers.py:
      - entering a slot (detection found) → transition IMMEDIATELY
      - leaving a slot (no detection) → wait threshold frames without detection
    """
    instant = classify_polygon_inner_core(detections, areas, inner_areas, num_slots)
    new_states = list(current_states)
    overlap_thresh = 8

    # Pre-compute which slots have a detection this frame
    detected_slots = set()
    for det in detections:
        cx, cy = det["cx"], det["cy"]
        for s in range(num_slots):
            in_outer = cv2.pointPolygonTest(
                np.array(areas[s], np.int32), (cx, cy), False) >= 0
            if not in_outer:
                continue
            in_inner = cv2.pointPolygonTest(
                np.array(inner_areas[s], np.int32), (cx, cy), False) >= 0
            state = "overlapping" if not in_inner else "occupied"
            if state in ("occupied", "overlapping"):
                detected_slots.add(s)

    # State rank: available(0) < overlapping(1) < occupied(2)
    STATE_RANK = {"available": 0, "overlapping": 1, "occupied": 2}

    for s in range(num_slots):
        inst = instant[s]
        curr = current_states[s]

        if inst == curr:
            counters[s] = 0
            continue

        entering = STATE_RANK.get(inst, 0) > STATE_RANK.get(curr, 0)
        leaving = STATE_RANK.get(inst, 0) < STATE_RANK.get(curr, 0)

        if entering:
            # Vehicle entering → transition immediately
            new_states[s] = inst
            counters[s] = 0

        elif leaving:
            # No detection for this slot
            if s not in detected_slots:
                counters[s] += 1
            else:
                counters[s] = 0

            is_overlap_trans = (inst == "available" and curr == "overlapping") or \
                               (inst == "overlapping" and curr == "occupied")
            eff_thresh = overlap_thresh if is_overlap_trans else threshold

            if counters[s] >= eff_thresh:
                new_states[s] = inst
                counters[s] = 0

    for s in range(num_slots):
        current_states[s] = new_states[s]
    return current_states


# ---------------------------------------------------------------------------
# Outside detection (shared across all methods)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, classes):
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
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        }

    total_tp = sum(cm[i][i] for i in range(n))
    acc = total_tp / total if total > 0 else 0.0

    present = [c for c in classes if c in res and res[c]["support"] > 0]
    macro_f1_3 = (sum(res[c]["f1"] for c in present) / len(present)) if present else 0.0
    weighted_f1_3 = (
        sum(res[c]["f1"] * res[c]["support"] for c in present)
        / sum(res[c]["support"] for c in present)
    ) if present else 0.0

    res["_overall"] = {
        "accuracy":       round(acc * 100, 2),
        "macro_f1_3class": round(macro_f1_3, 2),
        "weighted_f1_3class": round(weighted_f1_3, 2),
        "total": total,
        "correct": total_tp,
    }
    return res, cm


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


def match_outside(preds, gt_bboxes, iou_thresh):
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
# Cache management
# ---------------------------------------------------------------------------

def _make_cache_key(video_path, model, conf, device):
    vid_stat = os.stat(video_path)
    key = f"{video_path}_{vid_stat.st_size}_{vid_stat.st_mtime:.0f}_{model}_{conf}_{device}"
    return hashlib.md5(key.encode()).hexdigest()


def run_full_video_inference(args, yolo_model, areas, inner_areas, slot_rects, num_slots):
    cache_dir = PROJECT_ROOT / ".cache" / "eval_method"
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

    # Try reuse cache from eval_temporal (same video/model/conf/device)
    temporal_cache_dir = PROJECT_ROOT / ".cache" / "eval_temporal"
    if temporal_cache_dir.exists():
        for cache_f in temporal_cache_dir.glob("detections_*.pkl"):
            try:
                with open(cache_f, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and len(data) > 1000:
                    # Verify cache was created with matching model+device
                    # by checking cache key matches what we would compute
                    expected_key = cache_key  # computed above with correct model/device
                    if cache_f.name == f"detections_{expected_key}.pkl":
                        print(f"[Cache] Reusing detections from {cache_f.name} (exact match)")
                        return data
            except Exception:
                pass

    if not Path(video_path).exists():
        print(f"[ERROR] Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    full_frames = VIDEO_END_FRAME - VIDEO_START_FRAME + 1

    print(f"\n[Inference] Running YOLO on full video: {total_frames} frames @ {fps_vid:.1f} FPS")
    print(f"           Range: frame {VIDEO_START_FRAME} -> {VIDEO_END_FRAME} ({full_frames} frames)")
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
            done = frame_idx - VIDEO_START_FRAME + 1
            progress = done / full_frames * 100
            eta = elapsed / done * (full_frames - done)
            print(f"  [{progress:5.1f}%] Frame {frame_idx}/{VIDEO_END_FRAME} | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
            last_print = now

    cap.release()
    elapsed_total = time.perf_counter() - t0
    print(f"[Inference] Done in {elapsed_total:.1f}s | {len(all_detections)} frames cached")

    with open(cache_file, "wb") as f:
        pickle.dump(all_detections, f)
    print(f"[Cache] Saved to {cache_file}")

    return all_detections


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args, all_detections, areas, inner_areas, slot_rects, num_slots,
                   frame_numbers, slot_state_gt, bbox_gt):
    eval_frames = sorted(set(frame_numbers) & set(f for f, s in slot_state_gt))
    if len(eval_frames) < 2:
        print("[Eval] Can it nhat 2 frame co GT. Bo qua.")
        return None

    gt_frame_set = set(eval_frames)
    warm_frames = min(args.warmup_frames, eval_frames[0])

    print(f"\n[Eval] {len(eval_frames)} frames co GT | "
          f"Range: {eval_frames[0]}->{eval_frames[-1]} | "
          f"Warmup: {warm_frames} frames")

    # -------------------------------------------------------------------------
    # Define 4 methods with their parameters
    # -------------------------------------------------------------------------
    # Each entry: (name, has_hysteresis, has_inner_core)
    method_configs = [
        ("Rectangular ROI",       False, False, "rectangular"),
        ("Polygon Only",          False, False, "polygon"),
        ("Polygon + Inner-Core",  False, True,  "inner_core"),
        ("Proposed Framework",     True,  True,  "proposed"),
    ]

    results = {}

    # -------------------------------------------------------------------------
    # Pre-allocate per-method temporal state
    # -------------------------------------------------------------------------
    # prev_pred_states[mname][s] = state string of slot s at previous GT frame
    prev_pred_states = {cfg[0]: {} for cfg in method_configs}
    # hyst state tracking for Proposed Framework only
    hyst_counters   = [0] * num_slots
    hyst_current    = [
        slot_state_gt.get((eval_frames[0], s), "available")
        for s in range(num_slots)
    ]

    # -------------------------------------------------------------------------
    # Collect predictions at GT frames, measure flickering on full video
    # -------------------------------------------------------------------------
    # We do a SINGLE pass over all GT frames (in order) and build up
    # the temporal state for each method as we go.
    #
    # For Methods 1-3: the state is just the instant classification at frame f.
    #   We track prev_pred_states[f-1] vs instant at f to detect flickering.
    #
    # For Method 4: we simulate full hysteresis by iterating over ALL frames
    #   from VIDEO_START_FRAME up to eval_frames[-1]. We maintain hyst_current
    #   and hyst_counters across frames.
    #
    # Because GT frames are a SUBSET of the full video, we need to "fill in"
    # intermediate non-GT frames for the hysteresis simulation.
    # Strategy: iterate frame by frame from VIDEO_START_FRAME to eval_frames[-1].
    #   - At each frame, run hysteresis update
    #   - At GT frames, record predictions for all 4 methods

    # Collect results per method
    method_y_true      = {cfg[0]: [] for cfg in method_configs}
    method_y_pred      = {cfg[0]: [] for cfg in method_configs}
    method_flick_trans = {cfg[0]: 0  for cfg in method_configs}
    method_trans_count = {cfg[0]: 0  for cfg in method_configs}
    method_outside_tp  = {cfg[0]: 0  for cfg in method_configs}
    method_outside_fp  = {cfg[0]: 0  for cfg in method_configs}
    method_outside_fn  = {cfg[0]: 0  for cfg in method_configs}

    method_elapsed = {cfg[0]: 0.0 for cfg in method_configs}

    # ---- Loop 1: Methods 1-3 (iterate only over GT frames) ----
    for frame_idx in eval_frames:
        t_frame = time.perf_counter()
        detections = all_detections.get(frame_idx, [])

        for mname, has_hyst, has_inner, mtype in method_configs:
            if has_hyst:
                continue  # handled in Loop 2

            if mtype == "rectangular":
                instant_states = classify_rectangular_roi(detections, slot_rects, num_slots)
            elif mtype == "polygon":
                instant_states = classify_polygon_only(detections, areas, num_slots)
            elif mtype == "inner_core":
                instant_states = classify_polygon_inner_core(
                    detections, areas, inner_areas, num_slots)

            # Record GT comparison
            for s in range(num_slots):
                key = (frame_idx, s)
                if key not in slot_state_gt:
                    continue
                gt_state = slot_state_gt[key]
                pred_state = instant_states[s]
                method_y_true[mname].append(gt_state)
                method_y_pred[mname].append(pred_state)

                # Flickering: compare to previous predicted state for this slot
                prev = prev_pred_states[mname].get(s)
                if prev is not None and prev != pred_state:
                    method_flick_trans[mname] += 1
                prev_pred_states[mname][s] = pred_state

            # Outside detection
            gt_bboxes_outside = bbox_gt.get(frame_idx, [])
            pred_outside = detect_outside(detections, areas, num_slots)
            otp, ofp, ofn = match_outside(pred_outside, gt_bboxes_outside, args.iou_threshold)
            method_outside_tp[mname] += otp
            method_outside_fp[mname] += ofp
            method_outside_fn[mname] += ofn

        method_elapsed[mname] += time.perf_counter() - t_frame

    # ---- Loop 2: Proposed Framework (full hysteresis loop) ----
    mname = "Proposed Framework"
    t_hyst = time.perf_counter()
    flickers_local = defaultdict(lambda: None)

    for frame_idx in range(VIDEO_START_FRAME, VIDEO_END_FRAME + 1):
        detections = all_detections.get(frame_idx, [])
        prev_states = list(hyst_current)

        classify_proposed(
            detections, areas, inner_areas, num_slots,
            args.hyst_threshold, hyst_counters, hyst_current)

        for s in range(num_slots):
            if prev_states[s] != hyst_current[s]:
                method_trans_count[mname] += 1

        if frame_idx in gt_frame_set:
            for s in range(num_slots):
                key = (frame_idx, s)
                if key not in slot_state_gt:
                    continue
                gt_state = slot_state_gt[key]
                pred_state = hyst_current[s]
                method_y_true[mname].append(gt_state)
                method_y_pred[mname].append(pred_state)

                prev = flickers_local[s]
                if prev is not None and prev != pred_state:
                    method_flick_trans[mname] += 1
                flickers_local[s] = pred_state

            gt_bboxes_outside = bbox_gt.get(frame_idx, [])
            pred_outside = detect_outside(detections, areas, num_slots)
            otp, ofp, ofn = match_outside(pred_outside, gt_bboxes_outside, args.iou_threshold)
            method_outside_tp[mname] += otp
            method_outside_fp[mname] += ofp
            method_outside_fn[mname] += ofn

    method_elapsed[mname] = time.perf_counter() - t_hyst

    # -------------------------------------------------------------------------
    # Compute metrics per method
    # -------------------------------------------------------------------------
    for mname, has_hyst, has_inner, mtype in method_configs:
        y_true = method_y_true[mname]
        y_pred = method_y_pred[mname]
        total = len(y_true)

        flicker_trans = method_flick_trans[mname]
        flick_rate = flicker_trans / total * 100 if total > 0 else 0.0
        trans_count = method_trans_count[mname]

        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / total * 100 if total > 0 else 0.0
        slot_res, cm = compute_metrics(y_true, y_pred, PRED_STATES)

        outside_tp = method_outside_tp[mname]
        outside_fp = method_outside_fp[mname]
        outside_fn = method_outside_fn[mname]
        total_outside = outside_tp + outside_fn
        out_p = outside_tp / (outside_tp + outside_fp) if (outside_tp + outside_fp) > 0 else 0.0
        out_r = outside_tp / (outside_tp + outside_fn) if (outside_tp + outside_fn) > 0 else 0.0
        out_f1 = 2 * out_p * out_r / (out_p + out_r) if (out_p + out_r) > 0 else 0.0

        results[mname] = {
            "method": mname,
            "hysteresis": has_hyst,
            "inner_core": has_inner,
            "accuracy": round(accuracy, 2),
            "macro_f1_3class": slot_res["_overall"]["macro_f1_3class"],
            "weighted_f1_3class": slot_res["_overall"]["weighted_f1_3class"],
            "flickering_rate": round(flick_rate, 4),
            "flickering_transitions": flicker_trans,
            "total_transitions": trans_count,
            "stability_score": round(accuracy, 4),
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
            "elapsed_seconds": round(method_elapsed[mname], 2),
        }

    return results


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def fmt_summary_table(results):
    W = 90
    out = []
    out.append("")
    out.append("=" * W)
    out.append("SECTION 5.4: SO SANH 4 PHUONG PHAP SUY LUAN CHIEU CHO O DO XE")
    out.append("=" * W)
    out.append("")

    out.append("  [BANG 5.4] Tong quan hieu suat")
    out.append(f"  {'Phuong phap':<25} {'Accuracy':>10} {'Macro-F1':>10} "
              "{'Weighted-F1':>12} {'Flick Rate':>12} {'Outside F1':>12}")
    out.append("  " + "-" * W)
    for name, res in results.items():
        out_f1 = res.get("outside_metrics", {}).get("f1", 0.0)
        out.append(f"  {name:<25} {res['accuracy']:>9.2f}% "
                  f"{res['macro_f1_3class']:>9.2f}% {res['weighted_f1_3class']:>11.2f}% "
                  f"{res['flickering_rate']:>11.4f}% {out_f1:>11.2f}%")

    out.append("")
    out.append("  [BANG 5.5] Chi so theo lop (Precision / Recall / F1)")
    out.append(f"  {'Phuong phap':<25} {'Avail P/R/F1':>18} {'Occu P/R/F1':>18} "
              "{'Over P/R/F1':>16}")
    out.append("  " + "-" * W)
    for name, res in results.items():
        pc = res.get("per_class", {})
        av = pc.get("available", {})
        oc = pc.get("occupied", {})
        ov = pc.get("overlapping", {})
        out.append(f"  {name:<25} "
                   f"{av['precision']:.1f}/{av['recall']:.1f}/{av['f1']:.1f}   "
                   f"{oc['precision']:.1f}/{oc['recall']:.1f}/{oc['f1']:.1f}   "
                   f"{ov['precision']:.1f}/{ov['recall']:.1f}/{ov['f1']:.1f}")

    out.append("")
    out.append("  [BANG 5.6] Confusion Matrix (rows=GT, cols=Pred)")
    out.append("  " + "-" * W)
    for name, res in results.items():
        cm = res.get("confusion_matrix", {})
        mat = cm.get("matrix", [])
        out.append(f"  {name}:")
        out.append("  " + "".join(f"{'':>8}{c:>12}" for c in PRED_STATES))
        for i, row_c in enumerate(PRED_STATES):
            row_vals = mat[i] if i < len(mat) else [0]*len(PRED_STATES)
            out.append("  " + f"{row_c:<{8+12}}"
                       + "".join(f"{v:>12}" for v in row_vals))

    out.append("")
    out.append("=" * W)
    return "\n".join(out)


def fmt_markdown(results):
    lines = []
    lines.append("## 5.4 Hiệu suất suy luận chiếm chỗ tổng thể")
    lines.append("")
    lines.append("### Bảng 5.4. Tổng hợp các chỉ số hiệu suất chính")
    lines.append("")
    lines.append("| Phương pháp | Accuracy | Macro-F1 (3-class) | Weighted-F1 (3-class) | Flickering Rate | Outside F1 |")
    lines.append("|---|---|---|---|---|---|")
    for name, res in results.items():
        out_f1 = res.get("outside_metrics", {}).get("f1", 0.0)
        lines.append(f"| {name} | {res['accuracy']:.2f}% | {res['macro_f1_3class']:.2f}% | "
                     f"{res['weighted_f1_3class']:.2f}% | {res['flickering_rate']:.4f} | {out_f1:.2f}% |")
    lines.append("")
    lines.append("### Bảng 5.5. Chi tiết theo lớp (Precision / Recall / F1)")
    lines.append("")
    lines.append("| Phương pháp | Available P/R/F1 | Occupied P/R/F1 | Overlapping P/R/F1 |")
    lines.append("|---|---|---|---|")
    for name, res in results.items():
        pc = res.get("per_class", {})
        av = pc.get("available", {})
        oc = pc.get("occupied", {})
        ov = pc.get("overlapping", {})
        lines.append(f"| {name} | "
                     f"{av['precision']:.1f}/{av['recall']:.1f}/{av['f1']:.1f} | "
                     f"{oc['precision']:.1f}/{oc['recall']:.1f}/{oc['f1']:.1f} | "
                     f"{ov['precision']:.1f}/{ov['recall']:.1f}/{ov['f1']:.1f} |")
    lines.append("")
    lines.append("### Bảng 5.6. Ma trận nhầm lẫn (hàng=GT, cột=Pred)")
    lines.append("")
    lines.append(f"| | " + " | ".join(f"**{c}**" for c in PRED_STATES) + " |")
    lines.append("|" + "|".join("---" for _ in range(len(PRED_STATES) + 1)) + "|")
    for name, res in results.items():
        cm = res.get("confusion_matrix", {})
        mat = cm.get("matrix", [])
        lines.append(f"**{name}**")
        for i, row_c in enumerate(PRED_STATES):
            row_vals = mat[i] if i < len(mat) else [0]*len(PRED_STATES)
            lines.append(f"| {row_c} | " + " | ".join(str(v) for v in row_vals) + " |")
    return "\n".join(lines)


def save_outputs(results, output_dir, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_json = {
        "config": {
            "video": resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/")),
            "slot_gt": resolve_path("annotations", args.slot_gt, DEFAULT_SLOT_GT),
            "frame_list": resolve_path("annotations", args.frame_list, DEFAULT_FRAME_LIST),
            "yolo_confidence": args.conf,
            "model_size": args.model_size,
            "hyst_threshold": args.hyst_threshold,
            "iou_threshold": args.iou_threshold,
            "fps": DEFAULT_VIDEO_FPS,
            "frames_evaluated": 1000,
        },
        "methods": results,
    }
    json_path = output_dir / "method_comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {json_path}")

    csv_path = output_dir / "method_comparison_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        cols = ["method", "accuracy", "macro_f1_3class", "weighted_f1_3class",
                "flickering_rate", "total_transitions", "frames_evaluated",
                "available_f1", "occupied_f1", "overlapping_f1",
                "outside_f1", "outside_precision", "outside_recall"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name, res in results.items():
            om = res.get("outside_metrics", {})
            pc = res.get("per_class", {})
            w.writerow({
                "method": name,
                "accuracy": res["accuracy"],
                "macro_f1_3class": res["macro_f1_3class"],
                "weighted_f1_3class": res["weighted_f1_3class"],
                "flickering_rate": res["flickering_rate"],
                "total_transitions": res["total_transitions"],
                "frames_evaluated": res["frames_evaluated"],
                "available_f1": pc.get("available", {}).get("f1", 0),
                "occupied_f1": pc.get("occupied", {}).get("f1", 0),
                "overlapping_f1": pc.get("overlapping", {}).get("f1", 0),
                "outside_f1": om.get("f1", 0),
                "outside_precision": om.get("precision", 0),
                "outside_recall": om.get("recall", 0),
            })
    print(f"[Saved] {csv_path}")

    md_path = output_dir / "method_comparison_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(fmt_markdown(results))
    print(f"[Saved] {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args is None:
        sys.exit(1)

    print("=" * 70)
    print("  eval_method_comparison.py — Compare 4 Parking Detection Methods")
    print("=" * 70)
    print(f"  Video:     {args.video}")
    print(f"  Device:    {args.device}")
    print(f"  Conf:      {args.conf}")
    print(f"  Hyst thr:  {args.hyst_threshold} frames")
    print(f"  Output:    {args.output}")

    slot_gt_path = resolve_path("annotations", args.slot_gt, DEFAULT_SLOT_GT)
    bbox_gt_path = resolve_path("annotations", args.bbox_gt, DEFAULT_BBOX_GT)
    frame_list_path = resolve_path("annotations", args.frame_list, DEFAULT_FRAME_LIST)

    print(f"\n[Loading data]")
    slot_state_gt = load_slot_state_gt(slot_gt_path)
    bbox_gt = load_bbox_gt(bbox_gt_path)
    frame_numbers = load_frame_numbers(frame_list_path)
    areas, inner_areas, slot_rects = load_parking_zones(args.video)
    num_slots = len(areas)

    print(f"  GT slot states: {len(slot_state_gt)} entries")
    print(f"  GT bbox outside: {sum(len(v) for v in bbox_gt.values())} entries")
    print(f"  Frame numbers: {len(frame_numbers)} frames")
    print(f"  Parking slots: {num_slots}")

    if args.use_cache:
        print(f"\n[Cache mode] Skipping YOLO inference.")
        all_detections = None
    else:
        print(f"\n[Init YOLO]")
        yolo_model, device_norm = init_yolo(args.device, args.model_size)
        if yolo_model is None:
            print("[Error] Cannot init YOLO. Exiting.")
            sys.exit(1)

        t_total = time.perf_counter()
        print(f"\n[Step 1/2] Running full video YOLO inference...")
        all_detections = run_full_video_inference(
            args, yolo_model, areas, inner_areas, slot_rects, num_slots)
        if all_detections is None:
            print("[Error] No detections available. Check video path.")
            sys.exit(1)

    print(f"\n[Step 2/2] Evaluating 4 methods...")
    results = run_evaluation(
        args, all_detections, areas, inner_areas, slot_rects, num_slots,
        frame_numbers, slot_state_gt, bbox_gt)

    if results:
        # Print per-method results with timing
        print(f"\n  {'Phuong phap':<25} {'Time':>8} {'Acc':>8} {'WF1':>8} {'FlickRate':>10}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for name in ["Rectangular ROI", "Polygon Only", "Polygon + Inner-Core", "Proposed Framework"]:
            r = results[name]
            elapsed = r.get("elapsed_seconds", 0)
            print(f"  {name:<25} {elapsed:>7.1f}s "
                  f"{r['accuracy']:>7.2f}% {r['weighted_f1_3class']:>7.2f}% "
                  f"{r['flickering_rate']:>9.4f}%")
        print("")
        print(fmt_summary_table(results))
        save_outputs(results, args.output, args)
    else:
        print("[Error] Evaluation produced no results.")

    print(f"\n{'='*70}")
    print(f"  DONE.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
