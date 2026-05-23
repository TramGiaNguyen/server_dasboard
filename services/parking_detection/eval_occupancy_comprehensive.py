#!/usr/bin/env python3
"""
eval_occupancy_comprehensive.py

Danh gia hien tuong chi em cho bai do xe - metrics dong lat.

Noi dung:
  A. Cong thuc co ban         - P/R/F1/CM cho 4 trang thai
  B. Metrics per-slot         - P/R/F1 cho tung o (slot 0-18)
  C. 4 che do thoi gian      - hysteresis on/off x inner-core on/off
  D. Ty le chuyen trang thai gia (false transition rate)
  E. Do tre cap nhat trang thai (state update latency)

Usage:
    python eval_occupancy_comprehensive.py

    # Chi chay phan A+B (nhanh, dung 100-1000 frame co GT)
    python eval_occupancy_comprehensive.py --sections ab

    # Chay tat ca 5 phan (can toan bo video)
    python eval_occupancy_comprehensive.py --sections abcde --video static/video/CAM_PARKING.mp4

    # Chi phan A voi tham so tuy chinh
    python eval_occupancy_comprehensive.py --conf 0.30 --iou 0.30
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Inline functions from detection.py (avoid full import chain with dotenv)
# ---------------------------------------------------------------------------

PARKING_ZONES_CONFIG_PATH = PROJECT_ROOT / "services" / "parking_detection" / "parking_zones_config.txt"


def _load_zones_from_config():
    cfg = {"PARKING_SLOTS": [], "ENTRY_ZONES": [], "ENTRY_LINE_ZONES": []}
    if not PARKING_ZONES_CONFIG_PATH.exists():
        return cfg
    try:
        namespace = {}
        with open(PARKING_ZONES_CONFIG_PATH, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, str(PARKING_ZONES_CONFIG_PATH), "exec"), namespace, namespace)
        if isinstance(namespace.get("PARKING_SLOTS"), list):
            cfg["PARKING_SLOTS"] = namespace["PARKING_SLOTS"]
    except Exception:
        pass
    return cfg


def define_parking_areas():
    cfg = _load_zones_from_config()
    slots = cfg.get("PARKING_SLOTS") or []
    if slots:
        return slots
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


def get_detection_params(slot_index):
    if slot_index == 18:
        return {'min_confidence': 0.35, 'min_area_size': 1500,
                'min_dimension': 30, 'allow_partial': False}
    return {'min_confidence': 0.50, 'min_area_size': 1500,
            'min_dimension': 30, 'allow_partial': False}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_VIDEO       = "static/video/CAM_PARKING.mp4"
DEFAULT_SLOT_GT     = "annotations/CAM_PARKING_slot_state_gt.csv"
DEFAULT_BBOX_GT     = "annotations/CAM_PARKING_bbox_outside_gt.csv"
DEFAULT_FRAME_LIST  = "annotations/frame_numbers.txt"
DEFAULT_OUTPUT      = "eval_results/comprehensive"

STATE_CATEGORIES    = ["available", "occupied", "overlapping", "outside"]
ALL_STATES          = ["available", "occupied", "overlapping", "outside"]

# YOLO vehicle classes (COCO)
VEHICLE_CLASS_IDS  = [2, 5, 7]

# Hysteresis config
DEFAULT_HYST_THRESHOLD = 45   # frames
DEFAULT_VIDEO_FPS       = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Comprehensive Occupancy Metrics: P/R/F1/CM per state, "
                    "per-slot, temporal modes, false transition, latency.")
    p.add_argument("--video",        default=DEFAULT_VIDEO,
                   help="Path to video file")
    p.add_argument("--slot-gt",     default=DEFAULT_SLOT_GT,
                   help="Path to slot state GT CSV")
    p.add_argument("--bbox-gt",     default=DEFAULT_BBOX_GT,
                   help="Path to outside bbox GT CSV")
    p.add_argument("--frame-list",  default=DEFAULT_FRAME_LIST,
                   help="Path to frame numbers list")
    p.add_argument("--output",       default=DEFAULT_OUTPUT,
                   help="Output directory")
    p.add_argument("--sections",     default="abcde",
                   help="Sections to run: a=basic, b=perslot, c=temporal, "
                        "d=falsetrans, e=latency. Default: abcde")
    p.add_argument("--conf",         type=float, default=0.30,
                   help="YOLO confidence threshold (default: 0.30)")
    p.add_argument("--iou",          type=float, default=0.30,
                   help="IoU threshold for bbox matching (default: 0.30)")
    p.add_argument("--hyst-threshold", type=int, default=DEFAULT_HYST_THRESHOLD,
                   help="Hysteresis counter threshold in frames (default: 45)")
    p.add_argument("--device", default="0",
                   help="Device for YOLO inference: 'cpu' or GPU index e.g. '0' (default: 0)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip frames already in GT (annotation mode)")
    return p.parse_args()


def resolve_path(base, path_arg, default):
    p = Path(path_arg)
    if p.is_absolute():
        return str(p)
    candidates = [path_arg, PROJECT_ROOT / path_arg, PROJECT_ROOT / default]
    for c in candidates:
        if Path(c).exists():
            return str(Path(c).resolve())
    return str(PROJECT_ROOT / default)


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

def init_yolo():
    model_path = PROJECT_ROOT / "static" / "models" / "yolov8l.pt"
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print(f"[YOLO] Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"[YOLO] ERROR: cannot load model ({e})")
        return None


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
# Slot state inference (2 variants)
# ---------------------------------------------------------------------------

def infer_slot_state_instant(detections, areas, inner_areas, num_slots,
                              use_inner_core=True):
    """
    Inference KHONG hysteresis - chuyen trang thai ngay lap tuc.
    Tra ve: list of predicted states per slot.
    """
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


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Outside vehicle inference
# ---------------------------------------------------------------------------

def detect_outside(detections, areas, num_slots):
    """Tra ve detections nam ngoai tat ca cac slot polygons."""
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
# IoU & matching
# ---------------------------------------------------------------------------

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
        fn = sum(cm[i][c] for c in range(n)) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        res[c] = {
            "precision": round(p * 100, 2),
            "recall":    round(r * 100, 2),
            "f1":        round(f * 100, 2),
            "support":   sum(cm[i]),
            "tp": tp, "fp": fp, "fn": fn,
        }
    total_tp = sum(cm[i][i] for i in range(n))
    acc = total_tp / total if total > 0 else 0.0
    res["_overall"] = {
        "accuracy":        round(acc * 100, 2),
        "macro_precision": round(np.mean([res[c]["precision"] for c in classes]), 2),
        "macro_recall":    round(np.mean([res[c]["recall"]    for c in classes]), 2),
        "macro_f1":        round(np.mean([res[c]["f1"]        for c in classes]), 2),
        "total": total,
    }
    return res, cm


def binary_metrics(labels, preds, pos_val):
    t = [l == pos_val for l in labels]
    p = [l == pos_val for l in preds]
    tp = sum(1 for a, b in zip(t, p) if a and b)
    fp = sum(1 for a, b in zip(t, p) if not a and b)
    fn = sum(1 for a, b in zip(t, p) if a and not b)
    sup = sum(t)
    pp = tp / (tp + fp) if (tp + fp) > 0 else 0
    rr = tp / (tp + fn) if (tp + fn) > 0 else 0
    ff = 2 * pp * rr / (pp + rr) if (pp + rr) > 0 else 0
    acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return {
        "accuracy":  round(acc * 100, 2),
        "precision": round(pp * 100, 2),
        "recall":    round(rr * 100, 2),
        "f1":        round(ff * 100, 2),
        "support":   sup,
        "tp": tp, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Section A: Basic metrics (P/R/F1/CM for 4 states)
# ---------------------------------------------------------------------------

def section_a_basic_metrics(args, yolo_model, areas, inner_areas,
                             num_slots, frame_numbers, slot_state_gt, bbox_gt):
    """Tinh P/R/F1/CM cho 4 trang thai tren cac frame co GT."""
    eval_frames = []
    if args.skip_existing:
        gt_frames = set(f for f, s in slot_state_gt)
        eval_frames = sorted(set(frame_numbers) & gt_frames)
    else:
        gt_frames = set(f for f, s in slot_state_gt)
        eval_frames = sorted(set(frame_numbers) & gt_frames)
        if not eval_frames:
            eval_frames = sorted(gt_frames)
            print(f"[Warn] Khong co overlap frame_numbers vs GT. "
                  f"Dung {len(eval_frames)} frame co GT.")

    if not eval_frames:
        print("[Error] Khong co frame nao de danh gia.")
        return None

    print(f"\n[Section A] Danh gia {len(eval_frames)} frames (co GT)...")

    state_true, state_pred = [], []
    outside_tp, outside_fp, outside_fn = 0, 0, 0
    wrong_pk = {"gt_total": 0, "detected": 0, "missed": 0, "fp": 0, "recall": 0.0}
    flickers = defaultdict(lambda: None)
    flicker_transitions = 0
    per_frame = []

    video_path = resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")
    fps_vid = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    cap = cv2.VideoCapture(str(video_path))

    # Per-slot prediction storage (for section B)
    per_slot_preds = {}  # {frame: {slot: pred_state}}

    for fi, frame_num in enumerate(eval_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((500, 1020, 3), dtype=np.uint8)

        detections = detect_yolo(yolo_model, frame, conf=args.conf, device=args.device)

        # Infer slot states (instant, no hysteresis for basic eval)
        slot_states_pred = infer_slot_state_instant(
            detections, areas, inner_areas, num_slots, use_inner_core=True)

        per_slot_preds[frame_num] = {s: slot_states_pred[s] for s in range(num_slots)}  # dict slot->state

        frame_gt_states, frame_pred_states = [], []
        for s in range(num_slots):
            key = (frame_num, s)
            if key not in slot_state_gt:
                continue
            gt_state = slot_state_gt[key]
            pred_state = slot_states_pred[s]
            frame_gt_states.append(gt_state)
            frame_pred_states.append(pred_state)
            state_true.append(gt_state)
            state_pred.append(pred_state)

            if gt_state == "overlapping":
                wrong_pk["gt_total"] += 1
                if pred_state == "overlapping":
                    wrong_pk["detected"] += 1
                else:
                    wrong_pk["missed"] += 1
            elif pred_state == "overlapping" and gt_state != "overlapping":
                wrong_pk["fp"] += 1

            prev = flickers[s]
            if prev is not None and prev != pred_state:
                flicker_transitions += 1
            flickers[s] = pred_state

        gt_outside = bbox_gt.get(frame_num, [])
        pred_outside = detect_outside(detections, areas, num_slots)
        tp, fp, fn = match_detections_to_gt(pred_outside, gt_outside, args.iou)
        outside_tp += tp
        outside_fp += fp
        outside_fn += fn

        correct = sum(t == p for t, p in zip(frame_gt_states, frame_pred_states))
        per_frame.append({
            "frame": frame_num,
            "correct": correct,
            "total": len(frame_gt_states),
            "accuracy": round(correct / len(frame_gt_states) * 100, 2)
                        if len(frame_gt_states) > 0 else 0,
            "gt_free": frame_gt_states.count("available"),
            "gt_occupied": frame_gt_states.count("occupied"),
            "gt_overlap": frame_gt_states.count("overlapping"),
            "pred_free": frame_pred_states.count("available"),
            "pred_occupied": frame_pred_states.count("occupied"),
            "pred_overlap": frame_pred_states.count("overlapping"),
            "outside_tp": tp, "outside_fp": fp, "outside_fn": fn,
            "num_detections": len(detections),
            "flickering_transitions": flicker_transitions,
        })

        if (fi + 1) % 20 == 0:
            print(f"  [{fi+1}/{len(eval_frames)}] frames processed...")

    cap.release()

    # Aggregate
    if wrong_pk["gt_total"] > 0:
        wrong_pk["recall"] = round(wrong_pk["detected"] / wrong_pk["gt_total"] * 100, 2)
    wrong_pk["accuracy"] = round(
        wrong_pk["detected"] / (wrong_pk["detected"] + wrong_pk["fp"] + wrong_pk["missed"])
        * 100, 2) if (wrong_pk["detected"] + wrong_pk["fp"] + wrong_pk["missed"]) > 0 else 0.0

    total_slot_checks = len(state_true)
    flick_rate = flicker_transitions / total_slot_checks * 100 if total_slot_checks > 0 else 0.0
    correct_total = sum(1 for t, p in zip(state_true, state_pred) if t == p)
    stability = correct_total / total_slot_checks * 100 if total_slot_checks > 0 else 0.0

    slot_results, slot_cm = compute_confusion_matrix(
        state_true, state_pred, STATE_CATEGORIES)
    slot_results["_cm"] = slot_cm

    free_res = binary_metrics(state_true, state_pred, "available")
    occ_res  = binary_metrics(state_true, state_pred, "occupied")

    outside_prec = outside_tp / (outside_tp + outside_fp) * 100 \
                   if (outside_tp + outside_fp) > 0 else 0.0
    outside_rec  = outside_tp / (outside_tp + outside_fn) * 100 \
                   if (outside_tp + outside_fn) > 0 else 0.0
    outside_f1   = 2 * outside_prec * outside_rec / (outside_prec + outside_rec) \
                   if (outside_prec + outside_rec) > 0 else 0.0
    outside_acc  = outside_tp / (outside_tp + outside_fp + outside_fn) * 100 \
                   if (outside_tp + outside_fp + outside_fn) > 0 else 0.0

    result = {
        "parking_state": {k: v for k, v in slot_results.items() if k != "_cm"},
        "confusion_matrix": {
            "classes": STATE_CATEGORIES,
            "matrix": slot_cm,
        },
        "free_slots":   free_res,
        "occupied_slots": occ_res,
        "wrong_parking": wrong_pk,
        "outside_vehicles": {
            "tp": outside_tp, "fp": outside_fp, "fn": outside_fn,
            "accuracy": round(outside_acc, 2),
            "precision": round(outside_prec, 2),
            "recall":    round(outside_rec, 2),
            "f1":        round(outside_f1, 2),
        },
        "flickering_rate": round(flick_rate, 4),
        "flickering_transitions": flicker_transitions,
        "stability_score": round(stability, 4),
        "yolo_confidence": args.conf,
        "iou_threshold": args.iou,
        "fps": round(fps_vid, 1),
        "frames_evaluated": len(eval_frames),
        "per_frame": per_frame,
        "_per_slot_preds": per_slot_preds,  # internal: slot preds per frame
    }
    return result


# ---------------------------------------------------------------------------
# Section B: Per-slot metrics
# ---------------------------------------------------------------------------

def section_b_per_slot(args, result_a):
    """Tinh P/R/F1 cho tung slot 0-18."""
    per_slot_preds = result_a.get("_per_slot_preds", {})
    if not per_slot_preds:
        return None

    slot_state_gt_path = resolve_path("annotations", args.slot_gt, DEFAULT_SLOT_GT)
    gt = load_slot_state_gt(slot_state_gt_path)
    num_slots = 19

    from collections import defaultdict
    slot_tp = defaultdict(int)
    slot_fp = defaultdict(int)
    slot_fn = defaultdict(int)
    slot_support = defaultdict(int)

    for frame_num, preds in per_slot_preds.items():
        for s in range(num_slots):
            key = (frame_num, s)
            if key not in gt:
                continue
            gt_state = gt[key]
            pred_state = preds.get(s, "available")
            slot_support[s] += 1

            if gt_state == pred_state:
                slot_tp[s] += 1
            else:
                slot_fn[s] += 1
                if pred_state != "available":
                    slot_fp[s] += 1

    slot_metrics = {}
    for s in range(num_slots):
        tp, fp, fn = slot_tp[s], slot_fp[s], slot_fn[s]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        acc = tp / slot_support[s] if slot_support[s] > 0 else 0.0
        slot_metrics[s] = {
            "accuracy":  round(acc * 100, 2),
            "precision": round(p * 100, 2),
            "recall":    round(r * 100, 2),
            "f1":        round(f * 100, 2),
            "support":   slot_support[s],
            "tp": tp, "fp": fp, "fn": fn,
        }

    print(f"\n[Section B] Per-slot metrics computed for {num_slots} slots.")
    return slot_metrics


# ---------------------------------------------------------------------------
# Section C: 4 temporal modes (hysteresis x inner-core)
# ---------------------------------------------------------------------------

def section_c_temporal_modes(args, yolo_model, areas, inner_areas,
                              num_slots, frame_numbers, slot_state_gt):
    """
    So sanh 4 che do:
      Hyst OFF / Inner ON  (instant, with inner-core)
      Hyst OFF / Inner OFF (instant, no inner-core)
      Hyst ON  / Inner ON  (hysteresis, with inner-core)
      Hyst ON  / Inner OFF (hysteresis, no inner-core)

    Hyst_ON modes run on CONTINUOUS video frames (not just GT frames) so that
    the counter can accumulate properly frame-to-frame. Predictions are compared
    with GT only at frames where GT exists.
    """
    gt_frames = set(f for f, s in slot_state_gt)
    eval_frames = sorted(set(frame_numbers) & gt_frames)
    if len(eval_frames) < 2:
        print("[Warn] Can it nhat 2 frame co GT de danh gia temporal. "
              "Bo qua Section C.")
        return None

    video_path = resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/"))
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    warm_start = max(0, eval_frames[0] - 50)
    eval_end = eval_frames[-1]

    print(f"\n[Section C] Temporal comparison on continuous frames "
          f"{warm_start}–{eval_end} ({eval_end - warm_start} frames, "
          f"{len(eval_frames)} have GT)...")

    modes = [
        ("Hyst_OFF_Inner_ON",  False, True),
        ("Hyst_OFF_Inner_OFF", False, False),
        ("Hyst_ON_Inner_ON",   True,  True),
        ("Hyst_ON_Inner_OFF",  True,  False),
    ]

    gt_frame_set = set(eval_frames)

    results = {}
    for mode_name, use_hyst, use_inner in modes:
        print(f"  Running mode: {mode_name}...")

        state_true, state_pred = [], []
        flickers = defaultdict(lambda: None)
        flicker_trans = 0
        hyst_counters = [0] * num_slots

        hyst_current = [
            slot_state_gt.get((eval_frames[0], s), "available")
            for s in range(num_slots)
        ]

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm_start)
        frame_idx = warm_start

        while frame_idx <= eval_end:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            detections = detect_yolo(yolo_model, frame, conf=args.conf, device=args.device)

            if use_hyst:
                slot_states_pred = _hysteresis_step(
                    detections, areas, inner_areas, num_slots,
                    args.hyst_threshold, use_inner,
                    hyst_counters, hyst_current)
            else:
                slot_states_pred = infer_slot_state_instant(
                    detections, areas, inner_areas, num_slots, use_inner_core=use_inner)

            if frame_idx in gt_frame_set:
                for s in range(num_slots):
                    key = (frame_idx, s)
                    if key not in slot_state_gt:
                        continue
                    gt_state = slot_state_gt[key]
                    pred_state = slot_states_pred[s]
                    state_true.append(gt_state)
                    state_pred.append(pred_state)

                    prev = flickers[s]
                    if prev is not None and prev != pred_state:
                        flicker_trans += 1
                    flickers[s] = pred_state

        cap.release()

        total = len(state_true)
        flick_rate = flicker_trans / total * 100 if total > 0 else 0.0
        correct = sum(1 for t, p in zip(state_true, state_pred) if t == p)
        accuracy = correct / total * 100 if total > 0 else 0.0

        slot_res, cm = compute_confusion_matrix(
            state_true, state_pred, STATE_CATEGORIES)

        results[mode_name] = {
            "mode": mode_name,
            "hysteresis": use_hyst,
            "inner_core": use_inner,
            "accuracy": round(accuracy, 2),
            "macro_precision": slot_res["_overall"]["macro_precision"],
            "macro_recall":    slot_res["_overall"]["macro_recall"],
            "macro_f1":        slot_res["_overall"]["macro_f1"],
            "flickering_rate": round(flick_rate, 4),
            "flickering_transitions": flicker_trans,
            "stability_score": round(
                correct / total * 100, 4) if total > 0 else 0.0,
            "per_class": {c: slot_res[c] for c in STATE_CATEGORIES if c in slot_res},
            "confusion_matrix": {"classes": STATE_CATEGORIES, "matrix": cm},
            "frames_evaluated": len(eval_frames),
        }

    # Comparison summary
    summary = []
    for name, res in results.items():
        summary.append({
            "mode": name,
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"],
            "flickering_rate": res["flickering_rate"],
        })

    print(f"\n  [Temporal Comparison Summary]")
    print(f"  {'Mode':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Flicker%':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for s in summary:
        print(f"  {s['mode']:<25} {s['accuracy']:>9.2f}% {s['macro_f1']:>9.2f}% "
              f"{s['flickering_rate']:>9.4f}%")

    return results


def _hysteresis_step(detections, areas, inner_areas, num_slots,
                      threshold, use_inner_core, counters, current_states):
    """
    Apply ONE step of per-class hysteresis counter logic for all slots.

    - Threshold for available <-> occupied transitions: full threshold (default 45 frames)
    - Threshold for overlapping transitions (entry/exit): overlap_threshold (default 8 frames)
    - Counter increments when instant != current (wants to change).
    - Counter resets when instant == current (stable).
    """
    overlap_threshold = 8  # frames required to confirm/clear overlapping state
    instant_states = infer_slot_state_instant(
        detections, areas, inner_areas, num_slots, use_inner_core=use_inner_core)

    new_states = list(current_states)

    for s in range(num_slots):
        inst = instant_states[s]
        curr = current_states[s]

        if inst == curr:
            counters[s] = 0
            continue

        # Determine effective threshold for this transition
        is_overlapping_transition = (inst == "overlapping" or curr == "overlapping")
        eff_thresh = overlap_threshold if is_overlapping_transition else threshold

        counters[s] = min(counters[s] + 1, eff_thresh + 10)

        if counters[s] >= eff_thresh:
            new_states[s] = inst
            counters[s] = 0

    for s in range(num_slots):
        current_states[s] = new_states[s]

    return new_states


# ---------------------------------------------------------------------------
# Section D: False transition rate
# ---------------------------------------------------------------------------

def section_d_false_transition(args, yolo_model, areas, inner_areas,
                                num_slots, frame_numbers):
    """
    Chay tren toan bo video (khong chi frame co GT) de tinh
    false transition rate.

    False transition = chuyen trang thai nhung gia tri counter nho
    (cho thay su dao dong/nhieu).

    Tra ve:
      - false_transitions: tong so transition gia
      - false_trans_per_hour
      - false_trans_per_minute
      - total_transitions: tong so transition
      - false_trans_rate: ty le (%)
    """
    video_path = resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps_vid if fps_vid > 0 else 1.0
    cap.release()

    print(f"\n[Section D] False transition rate on FULL video "
          f"({total_frames} frames, {duration_sec:.1f}s)...")

    # Chi doc mot so frame跳 qua video (moi N frame)
    SAMPLE_STEP = 1  # doc moi frame (co the tang len neu can)
    total_sampled = min(total_frames, 41363)

    cap = cv2.VideoCapture(str(video_path))

    # Hysteresis state per slot
    counters = [0] * num_slots
    current_states = ["available"] * num_slots

    # Transition log
    # transition_log: list of (frame, slot, from_state, to_state, counter_at_transition, is_false)
    transition_log = []

    false_trans = 0
    total_trans = 0
    min_count_for_true = args.hyst_threshold // 3  # transition la true neu counter >= 15

    frame_idx = 0
    frames_read = 0

    while frames_read < total_sampled:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
        frame_idx += 1

        if (frames_read - 1) % SAMPLE_STEP != 0:
            continue

        detections = detect_yolo(yolo_model, frame, conf=args.conf, device=args.device)
        instant_states = infer_slot_state_instant(
            detections, areas, inner_areas, num_slots, use_inner_core=True)

        for s in range(num_slots):
            inst = instant_states[s]
            prev_state = current_states[s]

            if inst == prev_state:
                counters[s] = min(counters[s] + 1, args.hyst_threshold)
            else:
                counters[s] = max(counters[s] - 1, 0)

            # Check if state change should happen
            would_change = False
            if prev_state == "available" and inst != "available":
                if counters[s] >= args.hyst_threshold:
                    would_change = True
            elif prev_state != "available" and inst == "available":
                if counters[s] >= args.hyst_threshold:
                    would_change = True

            # We only log when state changes OR counter is low (potential false)
            counter_before = counters[s]

            if would_change:
                new_state = inst if inst != "available" else "available"
                is_false = counter_before < min_count_for_true
                transition_log.append({
                    "frame": frame_idx,
                    "slot": s,
                    "from_state": prev_state,
                    "to_state": new_state,
                    "counter": counter_before,
                    "is_false": is_false,
                })
                if is_false:
                    false_trans += 1
                total_trans += 1
                current_states[s] = new_state
                counters[s] = 0

    cap.release()

    false_rate = false_trans / total_trans * 100 if total_trans > 0 else 0.0
    false_per_sec = false_trans / duration_sec
    false_per_min = false_per_sec * 60
    false_per_hour = false_per_sec * 3600

    print(f"  Total transitions: {total_trans}")
    print(f"  False transitions: {false_trans}")
    print(f"  False rate: {false_rate:.2f}%")
    print(f"  False / hour: {false_per_hour:.2f}")
    print(f"  False / minute: {false_per_min:.4f}")

    return {
        "total_transitions": total_trans,
        "false_transitions": false_trans,
        "false_transition_rate_pct": round(false_rate, 4),
        "false_trans_per_second": round(false_per_sec, 4),
        "false_trans_per_minute": round(false_per_min, 4),
        "false_trans_per_hour": round(false_per_hour, 4),
        "video_duration_sec": round(duration_sec, 2),
        "video_total_frames": total_frames,
        "frames_sampled": frames_read,
        "hysteresis_threshold": args.hyst_threshold,
        "transition_log": transition_log[:5000],
    }


# ---------------------------------------------------------------------------
# Section E: State update latency
# ---------------------------------------------------------------------------

def section_e_latency(args, yolo_model, areas, inner_areas,
                       num_slots, frame_numbers, slot_state_gt):
    """
    Do do tre cap nhat trang thai.

    Latency = so frame tu luc trang thai GT thay doi den luc
    trang thai prediction thay doi (voi hysteresis).

    Chay hysteresis tren VIDEO LIEN TUCI (khong chi GT frames) de counter
    tich luy dung. Chi so sanh voi GT tai cac frame co GT.
    """
    gt_frames = set(f for f, s in slot_state_gt)
    eval_frames = sorted(set(frame_numbers) & gt_frames)
    if len(eval_frames) < 2:
        print("[Warn] Can it nhat 2 frame co GT de tinh latency. "
              "Bo qua Section E.")
        return None

    video_path = resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/"))
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    warm_start = max(0, eval_frames[0] - 50)
    eval_end = eval_frames[-1]

    print(f"\n[Section E] State update latency on continuous frames "
          f"{warm_start}–{eval_end} ({eval_end - warm_start} frames, "
          f"{len(eval_frames)} have GT)...")

    # Xac dinh cac transition trong GT
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

    if not gt_transitions:
        print("[Warn] Khong co transition nao trong GT. Bo qua Section E.")
        return None

    # Chay hysteresis inference tren VIDEO LIEN TUC
    counters = [0] * num_slots
    current_states = [
        slot_state_gt.get((eval_frames[0], s), "available")
        for s in range(num_slots)
    ]

    pred_transitions = []
    pred_states_by_frame = {}
    gt_frame_set = set(eval_frames)

    cap = cv2.VideoCapture(str(video_path))

    # Warm-up: chay hysteresis tu warm_start den eval_frames[0]-1
    # de tich luy counter. Khong can cap.set frame 0 vi
    # warm_start = eval_frames[0] - 50 >= 0 (frame_numbers min = 56)
    if warm_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm_start)
        fw_idx = warm_start
        while fw_idx < eval_frames[0]:
            ret, frame = cap.read()
            if not ret:
                break
            fw_idx += 1
            detections = detect_yolo(yolo_model, frame, conf=args.conf, device=args.device)
            _hysteresis_step(
                detections, areas, inner_areas, num_slots,
                args.hyst_threshold, True, counters, current_states)

    # Reset counters khi bat dau GT period de tranh false transition
    # tu warm-up. Prediction se tich luy lai tu dau.
    counters = [0] * num_slots
    # Re-init states from GT tai frame dau tien
    current_states = [
        slot_state_gt.get((eval_frames[0], s), "available")
        for s in range(num_slots)
    ]

    cap.set(cv2.CAP_PROP_POS_FRAMES, eval_frames[0])
    frame_idx = eval_frames[0]

    while frame_idx <= eval_end:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = detect_yolo(yolo_model, frame, conf=args.conf, device=args.device)

        prev_states = list(current_states)
        slot_states_pred = _hysteresis_step(
            detections, areas, inner_areas, num_slots,
            args.hyst_threshold, True,  # always use inner_core for E
            counters, current_states)

        # Record transitions that happened this frame
        for s in range(num_slots):
            if prev_states[s] != current_states[s]:
                pred_transitions.append({
                    "frame": frame_idx,
                    "slot": s,
                    "from_state": prev_states[s],
                    "to_state": current_states[s],
                    "counter_at_fire": 0,
                })

        # Only store pred states at GT frames
        if frame_idx in gt_frame_set:
            pred_states_by_frame[frame_idx] = dict(enumerate(current_states))

    cap.release()

    # Debug
    print(f"  [Debug] GT transitions: {len(gt_transitions)}, Pred transitions: {len(pred_transitions)}")

    # Tinh latency = frame_pred_transition - frame_gt_transition
    from collections import defaultdict as dd2
    gt_by_slot2 = dd2(list)
    for t in gt_transitions:
        gt_by_slot2[t["slot"]].append(t)

    pred_by_slot2 = dd2(list)
    for t in pred_transitions:
        pred_by_slot2[t["slot"]].append(t)

    latencies = []
    for s in range(num_slots):
        gt_list = sorted(gt_by_slot2[s], key=lambda x: x["frame"])
        pred_list = sorted(pred_by_slot2[s], key=lambda x: x["frame"])

        for gt_t in gt_list:
            for pred_t in pred_list:
                if (gt_t["to_state"] == pred_t["to_state"] and
                        pred_t["frame"] >= gt_t["frame"]):
                    latency_frames = pred_t["frame"] - gt_t["frame"]
                    latencies.append({
                        "slot": s,
                        "gt_frame": gt_t["frame"],
                        "pred_frame": pred_t["frame"],
                        "latency_frames": latency_frames,
                        "latency_sec": round(latency_frames / 30.0, 3),
                        "to_state": gt_t["to_state"],
                    })
                    break

    if not latencies:
        print("[Warn] Khong tim thay cap transition GT/pred nao. "
              "Co the do hysteresis tre.")
        return None

    latency_vals = [l["latency_frames"] for l in latencies]
    lat_arr = np.array(latency_vals)

    result = {
        "count": len(latency_vals),
        "min_frames": int(np.min(lat_arr)),
        "max_frames": int(np.max(lat_arr)),
        "mean_frames": round(float(np.mean(lat_arr)), 2),
        "median_frames": round(float(np.median(lat_arr)), 2),
        "std_frames": round(float(np.std(lat_arr)), 2),
        "p95_frames": int(np.percentile(lat_arr, 95)),
        "p99_frames": int(np.percentile(lat_arr, 99)),
        "min_sec": round(float(np.min(lat_arr)) / 30.0, 3),
        "max_sec": round(float(np.max(lat_arr)) / 30.0, 3),
        "mean_sec": round(float(np.mean(lat_arr)) / 30.0, 3),
        "median_sec": round(float(np.median(lat_arr)) / 30.0, 3),
        "std_sec": round(float(np.std(lat_arr)) / 30.0, 3),
        "p95_sec": round(float(np.percentile(lat_arr, 95)) / 30.0, 3),
        "p99_sec": round(float(np.percentile(lat_arr, 99)) / 30.0, 3),
        "fps": 30.0,
        "hysteresis_threshold": args.hyst_threshold,
        "transitions": latencies[:500],
    }

    print(f"  Latency frames: min={result['min_frames']}, "
          f"max={result['max_frames']}, "
          f"mean={result['mean_frames']}, "
          f"median={result['median_frames']}, "
          f"P95={result['p95_frames']}")
    print(f"  Latency sec:    min={result['min_sec']}s, "
          f"max={result['max_sec']}s, "
          f"mean={result['mean_sec']}s, "
          f"median={result['median_sec']}s, "
          f"P95={result['p95_sec']}s")

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt_section_a(result):
    W = 70
    out = []
    out.append("")
    out.append("=" * W)
    out.append("SECTION A: Cong thuc Co Ban (P/R/F1/CM)")
    out.append("=" * W)

    sr = result["parking_state"]
    ov = sr["_overall"]
    out.append(f"  Frames evaluated: {result['frames_evaluated']}   "
               f"YOLO conf: {result['yolo_confidence']}   "
               f"IoU: {result['iou_threshold']}   FPS: {result['fps']}")
    out.append("")
    out.append(f"  {'Trang thai':<20} {'Precision':>10} {'Recall':>10} "
               f"{'F1':>10} {'Support':>10}")
    out.append("  " + "-" * W)
    for c in STATE_CATEGORIES:
        if c in sr:
            r = sr[c]
            out.append(f"  {c:<20} {r['precision']:>9.2f}% {r['recall']:>9.2f}% "
                       f"{r['f1']:>9.2f}% {r['support']:>10}")
    out.append("")
    out.append(f"  Accuracy:     {ov['accuracy']:.2f}%")
    out.append(f"  Macro-Prec:   {ov['macro_precision']:.2f}%")
    out.append(f"  Macro-Recall: {ov['macro_recall']:.2f}%")
    out.append(f"  Macro-F1:     {ov['macro_f1']:.2f}%")

    cm = result["confusion_matrix"]
    out.append("")
    out.append("  Confusion Matrix (rows=true, cols=pred):")
    out.append("  " + "".join(f"{'':>5}{c:>{14}}" for c in STATE_CATEGORIES))
    for i, rc in enumerate(STATE_CATEGORIES):
        out.append("  " + f"{rc:<{5+14}}"
                   + "".join(f"{cm['matrix'][i][j]:>14}" for j in range(len(STATE_CATEGORIES))))

    out.append("")
    out.append("  [Free Slots (available)]")
    fr = result["free_slots"]
    out.append(f"    P={fr['precision']:.2f}%  R={fr['recall']:.2f}%  "
               f"F1={fr['f1']:.2f}%  TP={fr['tp']} FP={fr['fp']} FN={fr['fn']}")
    out.append("  [Occupied Slots]")
    or_ = result["occupied_slots"]
    out.append(f"    P={or_['precision']:.2f}%  R={or_['recall']:.2f}%  "
               f"F1={or_['f1']:.2f}%  TP={or_['tp']} FP={or_['fp']} FN={or_['fn']}")
    out.append("  [Wrong Parking Detection]")
    wp = result["wrong_parking"]
    out.append(f"    Recall={wp['recall']:.2f}%  "
               f"GT={wp['gt_total']}  Detected={wp['detected']}  "
               f"Missed={wp['missed']}  FP={wp['fp']}")
    out.append("  [Outside Vehicle Detection]")
    ov_ = result["outside_vehicles"]
    out.append(f"    P={ov_['precision']:.2f}%  R={ov_['recall']:.2f}%  "
               f"F1={ov_['f1']:.2f}%  TP={ov_['tp']} FP={ov_['fp']} FN={ov_['fn']}")

    out.append("")
    out.append(f"  Flickering Rate:   {result['flickering_rate']:.4f}%")
    out.append(f"  Flicker Transitions: {result['flickering_transitions']}")
    out.append(f"  Stability Score:  {result['stability_score']:.4f}%")
    out.append("=" * W)
    return "\n".join(out)


def fmt_section_b(slot_metrics):
    W = 70
    out = []
    out.append("")
    out.append("=" * W)
    out.append("SECTION B: Metrics Theo Tung O (Per-Slot)")
    out.append("=" * W)
    out.append(f"  {'Slot':>6} {'Accuracy':>10} {'Precision':>10} "
               f"{'Recall':>10} {'F1':>10} {'Support':>8}")
    out.append("  " + "-" * W)
    for s in range(19):
        m = slot_metrics.get(s, {})
        out.append(f"  {s:>6} {m.get('accuracy',0):>9.2f}% "
                   f"{m.get('precision',0):>9.2f}% {m.get('recall',0):>9.2f}% "
                   f"{m.get('f1',0):>9.2f}% {m.get('support',0):>8}")
    out.append("=" * W)
    return "\n".join(out)


def fmt_section_c(results):
    W = 70
    out = []
    out.append("")
    out.append("=" * W)
    out.append("SECTION C: 4 Che Do Thoi Gian (Hysteresis x Inner-Core)")
    out.append("=" * W)
    out.append(f"  {'Mode':<25} {'Accuracy':>10} {'Macro-F1':>10} "
               f"{'FlickerRate':>12} {'Transitions':>12}")
    out.append("  " + "-" * W)
    for name, res in results.items():
        out.append(f"  {name:<25} {res['accuracy']:>9.2f}% "
                   f"{res['macro_f1']:>9.2f}% {res['flickering_rate']:>11.4f}% "
                   f"{res['flickering_transitions']:>12}")
    out.append("")
    out.append("  Per-Class F1-Score:")
    out.append(f"  {'Mode':<25} " + "".join(f"{c:>12}" for c in STATE_CATEGORIES))
    out.append("  " + "-" * W)
    for name, res in results.items():
        pc = res.get("per_class", {})
        vals = "".join(f"{pc.get(c,{}).get('f1',0):>11.1f}%" for c in STATE_CATEGORIES)
        out.append(f"  {name:<25}{vals}")
    out.append("=" * W)
    return "\n".join(out)


def fmt_section_d(result):
    W = 70
    out = []
    out.append("")
    out.append("=" * W)
    out.append("SECTION D: Ty Le Chuyen Trang Thai Gia (False Transition Rate)")
    out.append("=" * W)
    out.append(f"  Tong transition:          {result['total_transitions']}")
    out.append(f"  Transition gia:           {result['false_transitions']}")
    out.append(f"  Ty le transition gia:     {result['false_transition_rate_pct']:.4f}%")
    out.append(f"  Thoi gian video:          {result['video_duration_sec']:.1f}s")
    out.append(f"  Frames sampled:           {result['frames_sampled']}")
    out.append(f"  Hysteresis threshold:     {result['hysteresis_threshold']} frames")
    out.append("")
    out.append(f"  False transitions / hour: {result['false_trans_per_hour']:.4f}")
    out.append(f"  False transitions / min:  {result['false_trans_per_minute']:.6f}")
    out.append(f"  False transitions / sec:  {result['false_trans_per_second']:.6f}")
    out.append("=" * W)
    return "\n".join(out)


def fmt_section_e(result):
    W = 70
    out = []
    out.append("")
    out.append("=" * W)
    out.append("SECTION E: Do Tre Cap Nhat Trang Thai (State Update Latency)")
    out.append("=" * W)
    out.append(f"  So cap transition: {result['count']}")
    out.append(f"  FPS:              {result['fps']}")
    out.append(f"  Hysteresis threshold: {result['hysteresis_threshold']} frames")
    out.append("")
    out.append(f"  {'Metric':<20} {'(frames)':>12} {'(giay)':>12}")
    out.append("  " + "-" * W)
    for lbl, key in [("Min", "min"), ("Max", "max"), ("Mean", "mean"),
                      ("Median", "median"), ("Std", "std"),
                      ("P95", "p95"), ("P99", "p99")]:
        out.append(f"  {lbl:<20} {result[f'{key}_frames']:>12} "
                   f"{result[f'{key}_sec']:>12.3f}s")
    out.append("=" * W)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    sections = set(args.sections.lower())

    # Resolve paths
    video_path    = resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/"))
    slot_gt_path  = resolve_path("annotations", args.slot_gt, DEFAULT_SLOT_GT)
    bbox_gt_path  = resolve_path("annotations", args.bbox_gt, DEFAULT_BBOX_GT)
    frame_list_path = resolve_path("annotations", args.frame_list, DEFAULT_FRAME_LIST)
    output_dir    = PROJECT_ROOT / args.output

    for lbl, p in [("Video", video_path), ("Slot GT", slot_gt_path),
                   ("BBox GT", bbox_gt_path), ("Frame list", frame_list_path)]:
        if not Path(p).exists():
            print(f"[Warn] {lbl} not found: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Config]")
    print(f"  Video:       {video_path}")
    print(f"  Slot GT:     {slot_gt_path}")
    print(f"  BBox GT:     {bbox_gt_path}")
    print(f"  Frame list:  {frame_list_path}")
    print(f"  Output:      {output_dir}")
    print(f"  Sections:    {args.sections}")

    # Load data
    frame_numbers = []
    if Path(frame_list_path).exists():
        frame_numbers = load_frame_numbers(frame_list_path)
        print(f"\n[Data] {len(frame_numbers)} frames in frame_numbers.txt")

    slot_state_gt = {}
    if Path(slot_gt_path).exists():
        slot_state_gt = load_slot_state_gt(slot_gt_path)
        print(f"[Data] {len(slot_state_gt)} slot-gt entries "
              f"({len(set(f for f,s in slot_state_gt))} frames)")

    bbox_gt = {}
    if Path(bbox_gt_path).exists():
        bbox_gt = load_bbox_gt(bbox_gt_path)
        print(f"[Data] {sum(len(v) for v in bbox_gt.values())} bbox-gt entries "
              f"({len(bbox_gt)} frames)")

    # Setup areas
    ret, sample_frame = None, None
    if Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            ret, sample_frame = cap.read()
            cap.release()
    if sample_frame is None:
        sample_frame = np.zeros((1080, 1910, 3), dtype=np.uint8)
    h, w = sample_frame.shape[:2]
    sx, sy = w / 1020.0, h / 500.0
    areas     = scale_areas(define_parking_areas(), sx, sy)
    inner_areas = scale_areas(create_inner_zones(define_parking_areas()), sx, sy)
    num_slots = len(areas)
    print(f"[Data] {num_slots} parking slots, resolution: {w}x{h}")

    # Init YOLO
    yolo_model = None
    if sections & {"a", "c", "d", "e"}:
        yolo_model = init_yolo()
        if yolo_model is None:
            print("[Error] YOLO model not available.")
            sys.exit(1)

    results = {}

    # --- Section A ---
    if "a" in sections:
        t0 = time.perf_counter()
        res_a = section_a_basic_metrics(
            args, yolo_model, areas, inner_areas, num_slots,
            frame_numbers, slot_state_gt, bbox_gt)
        results["section_a"] = res_a
        print(f"[Section A] Done in {time.perf_counter()-t0:.1f}s")
        if res_a:
            print(fmt_section_a(res_a))

    # --- Section B ---
    if "b" in sections and results.get("section_a"):
        res_b = section_b_per_slot(args, results["section_a"])
        results["section_b"] = res_b
        if res_b:
            print(fmt_section_b(res_b))

    # --- Section C ---
    if "c" in sections:
        t0 = time.perf_counter()
        res_c = section_c_temporal_modes(
            args, yolo_model, areas, inner_areas, num_slots,
            frame_numbers, slot_state_gt)
        results["section_c"] = res_c
        print(f"[Section C] Done in {time.perf_counter()-t0:.1f}s")
        if res_c:
            print(fmt_section_c(res_c))

    # --- Section D ---
    if "d" in sections:
        t0 = time.perf_counter()
        res_d = section_d_false_transition(
            args, yolo_model, areas, inner_areas, num_slots, frame_numbers)
        results["section_d"] = res_d
        print(f"[Section D] Done in {time.perf_counter()-t0:.1f}s")
        if res_d:
            print(fmt_section_d(res_d))

    # --- Section E ---
    if "e" in sections:
        t0 = time.perf_counter()
        res_e = section_e_latency(
            args, yolo_model, areas, inner_areas, num_slots,
            frame_numbers, slot_state_gt)
        results["section_e"] = res_e
        print(f"[Section E] Done in {time.perf_counter()-t0:.1f}s")
        if res_e:
            print(fmt_section_e(res_e))

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    # Main JSON (no per_frame/transitions large arrays)
    def _strip_big(obj):
        if isinstance(obj, dict):
            return {k: _strip_big(v) for k, v in obj.items()
                    if k not in ("per_frame", "transitions", "transition_log",
                                 "_per_slot_preds")}
        if isinstance(obj, list):
            return [_strip_big(x) for x in obj[:20]]
        return obj

    main_json = {
        "config": {
            "video": video_path,
            "slot_gt": slot_gt_path,
            "bbox_gt": bbox_gt_path,
            "frame_list": frame_list_path,
            "yolo_confidence": args.conf,
            "iou_threshold": args.iou,
            "hysteresis_threshold": args.hyst_threshold,
            "sections": args.sections,
            "fps_video": 30.0,
        },
        **{k: v for k, v in results.items() if v is not None},
    }
    with open(output_dir / "occupancy_comprehensive_metrics.json", "w",
              encoding="utf-8") as f:
        json.dump(main_json, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {output_dir / 'occupancy_comprehensive_metrics.json'}")

    # Section A per-frame CSV
    if "a" in sections and results.get("section_a"):
        pf = results["section_a"].get("per_frame", [])
        if pf:
            csv_path = output_dir / "occupancy_per_frame.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                cols = ["frame", "correct", "total", "accuracy",
                        "gt_free", "gt_occupied", "gt_overlap",
                        "pred_free", "pred_occupied", "pred_overlap",
                        "outside_tp", "outside_fp", "outside_fn",
                        "num_detections", "flickering_transitions"]
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(pf)
            print(f"[Saved] {csv_path}")

    # Confusion matrix CSV
    if "a" in sections and results.get("section_a"):
        cm = results["section_a"].get("confusion_matrix", {})
        if cm:
            cm_csv = output_dir / "occupancy_confusion_matrix.csv"
            with open(cm_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([""] + cm["classes"])
                for i, row in enumerate(cm["matrix"]):
                    w.writerow([cm["classes"][i]] + row)
            print(f"[Saved] {cm_csv}")

    # Section B per-slot CSV
    if "b" in sections and results.get("section_b"):
        sb = results["section_b"]
        if sb:
            csv_path = output_dir / "occupancy_per_slot.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "slot", "accuracy", "precision", "recall", "f1",
                    "support", "tp", "fp", "fn"])
                w.writeheader()
                for s in range(num_slots):
                    m = sb.get(s, {})
                    w.writerow({
                        "slot": s,
                        "accuracy": m.get("accuracy", 0),
                        "precision": m.get("precision", 0),
                        "recall": m.get("recall", 0),
                        "f1": m.get("f1", 0),
                        "support": m.get("support", 0),
                        "tp": m.get("tp", 0),
                        "fp": m.get("fp", 0),
                        "fn": m.get("fn", 0),
                    })
            print(f"[Saved] {csv_path}")

    # Section C temporal comparison JSON
    if "c" in sections and results.get("section_c"):
        with open(output_dir / "temporal_comparison.json", "w",
                  encoding="utf-8") as f:
            json.dump(results["section_c"], f, indent=2, ensure_ascii=False)
        print(f"[Saved] {output_dir / 'temporal_comparison.json'}")

    # Section D false transition JSON
    if "d" in sections and results.get("section_d"):
        sd = dict(results["section_d"])
        sd["transition_log"] = sd.get("transition_log", [])[:5000]
        with open(output_dir / "false_transition_report.json", "w",
                  encoding="utf-8") as f:
            json.dump(sd, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {output_dir / 'false_transition_report.json'}")

    # Section E latency JSON
    if "e" in sections and results.get("section_e"):
        se = dict(results["section_e"])
        se["transitions"] = se.get("transitions", [])[:500]
        with open(output_dir / "state_latency_report.json", "w",
                  encoding="utf-8") as f:
            json.dump(se, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {output_dir / 'state_latency_report.json'}")

    # Section D transition log CSV
    if "d" in sections and results.get("section_d"):
        tl = results["section_d"].get("transition_log", [])
        if tl:
            csv_path = output_dir / "slot_transition_log.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "frame", "slot", "from_state", "to_state",
                    "counter", "is_false"])
                w.writeheader()
                w.writerows(tl)
            print(f"[Saved] {csv_path}")

    print("\n[Done] Tat ca ket qua da luu.")


if __name__ == "__main__":
    main()
