#!/usr/bin/env python3
"""
eval_parking_occupancy.py - Section 5.4 Occupancy Inference Performance

So sanh YOLO detections vs ground truth tu annotation tool:
  - annotations/CAM_PARKING_slot_state_gt.csv   -> trang thai slot (available/occupied/overlapping)
  - annotations/CAM_PARKING_bbox_outside_gt.csv  -> bbox xe ngoai

Danh gia:
  1. Parking State Accuracy (available vs occupied vs overlapping)
  2. Outside Vehicle Detection (TP/FP/FN, Precision, Recall, F1)

Usage:
    python eval_parking_occupancy.py
    python eval_parking_occupancy.py --slot-gt annotations/CAM_PARKING_slot_state_gt.csv
                                     --bbox-gt annotations/CAM_PARKING_bbox_outside_gt.csv
                                     --frame-list annotations/frame_numbers.txt
                                     --output eval_results/my_test
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from services.parking_detection.detection import (
    define_parking_areas,
    create_inner_zones,
    scale_areas,
)


DEFAULT_VIDEO       = "static/video/CAM_PARKING.mp4"
DEFAULT_SLOT_GT     = "annotations/CAM_PARKING_slot_state_gt.csv"
DEFAULT_BBOX_GT     = "annotations/CAM_PARKING_bbox_outside_gt.csv"
DEFAULT_FRAME_LIST  = "annotations/frame_numbers.txt"
DEFAULT_OUTPUT      = "eval_results/parking_occupancy"

STATE_CATEGORIES    = ["available", "occupied", "overlapping"]
PROJECT_ROOT        = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Section 5.4 Occupancy Inference Evaluation")
    p.add_argument("--video",      default=DEFAULT_VIDEO,      help="Path to video file")
    p.add_argument("--slot-gt",    default=DEFAULT_SLOT_GT,    help="Path to slot state GT CSV")
    p.add_argument("--bbox-gt",    default=DEFAULT_BBOX_GT,    help="Path to outside bbox GT CSV")
    p.add_argument("--frame-list", default=DEFAULT_FRAME_LIST, help="Path to frame numbers list")
    p.add_argument("--output",     default=DEFAULT_OUTPUT,    help="Output directory")
    p.add_argument("--conf",        type=float, default=0.30,
                   help="YOLO confidence threshold (default: 0.30)")
    p.add_argument("--iou-threshold", type=float, default=0.30,
                   help="IoU threshold for bbox matching (default: 0.30)")
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
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
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


def detect_yolo(model, frame, conf=0.30):
    if model is None:
        return []
    try:
        results = model(frame, conf=conf, iou=0.45,
                        classes=[2, 5, 7], verbose=False, device="cpu")
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
                    "class_id": cls_id,
                    "conf": conf_s,
                })
        return detections
    except Exception as e:
        print(f"[YOLO] Inference error: {e}")
        return []


# ---------------------------------------------------------------------------
# Parking slot inference from detections
# ---------------------------------------------------------------------------

def infer_slot_state(cx, cy, areas, inner_areas, slot_idx):
    """
    Dua tren tam (cx, cy) cua detection, xac dinh trang thai slot:
      - center nam trong inner zone -> occupied (dung vi tri)
      - center nam trong slot polygon nhung ngoai inner -> overlapping
      - khong nam trong polygon -> khong lien quan slot nay
    Tra ve: "occupied" | "overlapping" | "outside" (khong co trong slot)
    """
    area = areas[slot_idx]
    inner = inner_areas[slot_idx]
    if cv2.pointPolygonTest(np.array(inner, np.int32), (cx, cy), False) >= 0:
        return "occupied"
    if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
        return "overlapping"
    return "outside"


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
    """
    Greedy IoU matching giua danh sach prediction vs ground truth.
    Tra ve: (tp, fp, fn)
    """
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
# Metrics
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
            "precision": round(p * 100, 1),
            "recall":    round(r * 100, 1),
            "f1":        round(f * 100, 1),
            "support":   sum(cm[i]),
            "tp": tp, "fp": fp, "fn": fn,
        }

    total_tp = sum(cm[i][i] for i in range(n))
    acc = total_tp / total if total > 0 else 0.0
    res["_overall"] = {
        "accuracy":        round(acc * 100, 1),
        "macro_precision": round(np.mean([res[c]["precision"] for c in classes]), 1),
        "macro_recall":    round(np.mean([res[c]["recall"]    for c in classes]), 1),
        "macro_f1":        round(np.mean([res[c]["f1"]        for c in classes]), 1),
        "total": total,
    }
    return res, cm


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def fmt_table(slot_res, slot_cm, free_res, occ_res, wrong_pk,
              outside_tp, outside_fp, outside_fn, fps, conf, iou_th, num_frames,
              flick_rate_pct, stability_score, flicker_transitions):
    W = 68

    def row(lbl, val):
        return f"  {lbl:<38} {val:>15}"

    out = []
    out.append("")
    out.append("=" * W)
    out.append("5.4  Occupancy Inference Performance")
    out.append("=" * W)
    out.append(f"  Frames evaluated: {num_frames}   YOLO conf: {conf}   IoU thresh: {iou_th}   FPS: {fps:.1f}")

    # Parking State
    ov = slot_res["_overall"]
    out.append("")
    out.append("  [1] Parking State (available / occupied / overlapping)")
    out.append(row("Accuracy (%)",          f"{ov['accuracy']:.1f}%"))
    out.append(row("Macro Precision (%)",   f"{ov['macro_precision']:.1f}%"))
    out.append(row("Macro Recall (%)",       f"{ov['macro_recall']:.1f}%"))
    out.append(row("Macro F1-Score (%)",     f"{ov['macro_f1']:.1f}%"))
    out.append("")
    out.append(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    out.append("  " + "-" * W)
    for c in STATE_CATEGORIES:
        if c in slot_res:
            r = slot_res[c]
            out.append(f"  {c:<20} {r['precision']:>9.1f}% {r['recall']:>9.1f}% "
                       f"{r['f1']:>9.1f}% {r['support']:>10}")

    # Confusion Matrix
    out.append("")
    out.append("  Confusion Matrix (rows=true, cols=pred):")
    out.append("  " + "".join(f"{'':>5}{c:>{12}}" for c in STATE_CATEGORIES))
    for i, rc in enumerate(STATE_CATEGORIES):
        out.append("  " + f"{rc:<{5+12}}"
                   + "".join(f"{slot_cm[i][j]:>12}" for j in range(len(STATE_CATEGORIES))))

    # Free / Occupied binary
    for lbl, res in [("Free Slots (available)", free_res), ("Occupied Slots", occ_res)]:
        out.append("")
        out.append(f"  [{lbl}]")
        out.append(row("  Accuracy (%)",  f"{res['accuracy']:.1f}%"))
        out.append(row("  Precision (%)", f"{res['precision']:.1f}%"))
        out.append(row("  Recall (%)",    f"{res['recall']:.1f}%"))
        out.append(row("  F1-Score (%)",  f"{res['f1']:.1f}%"))
        out.append(row("  Support",       f"{res['support']}"))
        out.append(row("  TP / FP / FN",  f"{res['tp']} / {res['fp']} / {res['fn']}"))

    # Wrong Parking
    out.append("")
    out.append("  [Wrong Parking Detection]")
    out.append(row("  Accuracy (%)",                   f"{wrong_pk['accuracy']:.1f}%"))
    out.append(row("  GT Overlapping Total",            f"{wrong_pk['gt_total']}"))
    out.append(row("  Correctly Detected",              f"{wrong_pk['detected']}"))
    out.append(row("  Missed",                         f"{wrong_pk['missed']}"))
    out.append(row("  Recall (%)",                      f"{wrong_pk['recall']:.1f}%"))
    out.append(row("  False Positives (over-flag)",     f"{wrong_pk['fp']}"))

    # Outside Vehicle Detection
    outside_prec = outside_tp / (outside_tp + outside_fp) if (outside_tp + outside_fp) > 0 else 0
    outside_rec  = outside_tp / (outside_tp + outside_fn) if (outside_tp + outside_fn) > 0 else 0
    outside_f1   = 2 * outside_prec * outside_rec / (outside_prec + outside_rec) if (outside_prec + outside_rec) > 0 else 0
    outside_acc  = outside_tp / (outside_tp + outside_fp + outside_fn) if (outside_tp + outside_fp + outside_fn) > 0 else 0

    out.append("")
    out.append("  [2] Outside Vehicle Detection (vs bbox GT)")
    out.append(row("  GT Outside Vehicles",    f"{outside_tp + outside_fn}"))
    out.append(row("  YOLO Detected",          f"{outside_tp + outside_fp}"))
    out.append(row("  TP / FP / FN",            f"{outside_tp} / {outside_fp} / {outside_fn}"))
    out.append(row("  Accuracy (%)",            f"{outside_acc * 100:.1f}%"))
    out.append(row("  Precision (%)",           f"{outside_prec * 100:.1f}%"))
    out.append(row("  Recall (%)",              f"{outside_rec * 100:.1f}%"))
    out.append(row("  F1-Score (%)",           f"{outside_f1 * 100:.1f}%"))

    # Flickering & Stability
    out.append("")
    out.append("  [Temporal Metrics]")
    out.append(row("  Flickering Rate (%)",     f"{flick_rate_pct:.2f}%"))
    out.append(row("  Flickering Transitions",  f"{flicker_transitions}"))
    out.append(row("  Stability Score (%)",    f"{stability_score:.2f}%"))

    out.append("")
    out.append("=" * W)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    video_path      = resolve_path("static/video", args.video,     DEFAULT_VIDEO.lstrip("static/"))
    slot_gt_path    = resolve_path("annotations",  args.slot_gt,   DEFAULT_SLOT_GT)
    bbox_gt_path    = resolve_path("annotations",  args.bbox_gt,   DEFAULT_BBOX_GT)
    frame_list_path = resolve_path("annotations",  args.frame_list, DEFAULT_FRAME_LIST)
    output_dir      = PROJECT_ROOT / args.output

    print("[Config]")
    for lbl, val in [("Video", video_path), ("Slot GT", slot_gt_path),
                      ("BBox GT", bbox_gt_path), ("Frames", frame_list_path)]:
        print(f"  {lbl:<12} {val}")
    print(f"  YOLO conf   {args.conf}")
    print(f"  IoU thresh  {args.iou_threshold}")

    for lbl, p in [("Video", video_path), ("Slot GT", slot_gt_path),
                    ("BBox GT", bbox_gt_path), ("Frame list", frame_list_path)]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{lbl} not found: {p}")

    # Load data
    frame_numbers  = load_frame_numbers(frame_list_path)
    slot_state_gt  = load_slot_state_gt(slot_gt_path)
    bbox_gt        = load_bbox_gt(bbox_gt_path)

    print(f"\n[Data] Frame list: {len(frame_numbers)} frames")
    print(f"[Data] Slot state GT: {len(slot_state_gt)} entries")
    print(f"[Data] BBox outside GT: {sum(len(v) for v in bbox_gt.values())} entries "
          f"across {len(bbox_gt)} frames")

    # Init YOLO
    yolo_model = init_yolo()
    if yolo_model is None:
        print("[Error] YOLO model not available. Exiting.")
        sys.exit(1)

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps_vid = cap.get(cv2.CAP_PROP_FPS)

    # Read one frame to get resolution
    ret, sample = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")
    h, w = sample.shape[:2]
    cap.release()
    fps_estimate = round(fps_vid, 1)
    sx, sy = w / 1020.0, h / 500.0
    areas    = scale_areas(define_parking_areas(), sx, sy)
    inner_areas = scale_areas(create_inner_zones(define_parking_areas()), sx, sy)
    num_slots = len(areas)

    # Determine eval frames (intersection of frame_numbers and frames with slot GT)
    slot_gt_frames   = set(f for (f, s) in slot_state_gt)
    eval_frames      = sorted(set(frame_numbers) & slot_gt_frames)
    if not eval_frames:
        eval_frames = sorted(slot_gt_frames)
        print(f"\n[Warn] No overlap between frame_numbers and slot GT. "
              f"Using {len(eval_frames)} GT frames.")

    print(f"[Eval] Frames to evaluate: {len(eval_frames)} "
          f"(range: {min(eval_frames)}-{max(eval_frames)})")

    # Per-frame accumulators
    state_true, state_pred = [], []

    # Outside detection accumulators
    total_outside_tp, total_outside_fp, total_outside_fn = 0, 0, 0

    # Wrong parking accumulator
    wrong_pk = {"gt_total": 0, "detected": 0, "missed": 0, "fp": 0, "recall": 0.0}

    # Flickering: track prev predicted state per slot to count flips
    # flickers[slot] = predicted state in previous frame (None if first frame)
    flickers = defaultdict(lambda: None)
    flicker_transitions = 0
    flicker_cumulative = 0  # cumulative count per frame for CSV

    # Per-frame detail
    per_frame = []

    # Open video for inference
    cap = cv2.VideoCapture(str(video_path))
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for fi, frame_num in enumerate(eval_frames):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((int(h), int(w), 3), dtype=np.uint8)

        # Run YOLO
        detections = detect_yolo(yolo_model, frame, conf=args.conf)

        # Infer per-slot state from detections
        # Start with all slots as "available"
        slot_states_pred = {s: "available" for s in range(num_slots)}
        for det in detections:
            cx, cy = det["cx"], det["cy"]
            for s in range(num_slots):
                state = infer_slot_state(cx, cy, areas, inner_areas, s)
                if state in ("occupied", "overlapping"):
                    # if not already set to occupied/overlapping, set it
                    if slot_states_pred[s] == "available":
                        slot_states_pred[s] = state
                    elif slot_states_pred[s] == "occupied" and state == "overlapping":
                        slot_states_pred[s] = "overlapping"

        # Accumulate slot state comparisons
        frame_gt_states   = []
        frame_pred_states = []
        for s in range(num_slots):
            key = (frame_num, s)
            if key not in slot_state_gt:
                continue
            gt_state = slot_state_gt[key]
            pred_state = slot_states_pred.get(s, "available")
            frame_gt_states.append(gt_state)
            frame_pred_states.append(pred_state)
            state_true.append(gt_state)
            state_pred.append(pred_state)

            # Wrong parking accumulation
            if gt_state == "overlapping":
                wrong_pk["gt_total"] += 1
                if pred_state == "overlapping":
                    wrong_pk["detected"] += 1
                else:
                    wrong_pk["missed"] += 1
            elif pred_state == "overlapping" and gt_state != "overlapping":
                wrong_pk["fp"] += 1

            # Flickering: count slot-state transitions (occupied <-> available)
            prev = flickers[s]
            if prev is not None and prev != pred_state:
                flicker_transitions += 1
            flickers[s] = pred_state
        flicker_cumulative = flicker_transitions

        # Outside vehicle detection: match YOLO detections NOT in any slot
        # vs GT outside bboxes for this frame
        gt_outside = bbox_gt.get(frame_num, [])

        # Predicted outside = detections whose center is NOT in any slot polygon
        pred_outside = []
        for det in detections:
            cx, cy = det["cx"], det["cy"]
            in_any_slot = False
            for s in range(num_slots):
                if cv2.pointPolygonTest(np.array(areas[s], np.int32), (cx, cy), False) >= 0:
                    in_any_slot = True
                    break
            if not in_any_slot:
                pred_outside.append(det)

        tp, fp, fn = match_detections_to_gt(pred_outside, gt_outside, args.iou_threshold)
        total_outside_tp += tp
        total_outside_fp += fp
        total_outside_fn += fn

        # Per-frame summary
        correct = sum(t == p for t, p in zip(frame_gt_states, frame_pred_states))
        total_s = len(frame_gt_states)
        per_frame.append({
            "frame":       frame_num,
            "correct":     correct,
            "total":       total_s,
            "accuracy":    round(correct / total_s * 100, 1) if total_s > 0 else 0,
            "gt_free":     frame_gt_states.count("available"),
            "gt_occupied": frame_gt_states.count("occupied"),
            "gt_overlap":  frame_gt_states.count("overlapping"),
            "pred_free":   frame_pred_states.count("available"),
            "pred_occupied":  frame_pred_states.count("occupied"),
            "pred_overlap":    frame_pred_states.count("overlapping"),
            "outside_tp":   tp, "outside_fp": fp, "outside_fn": fn,
            "num_detections": len(detections),
            "flickering_transitions_so_far": flicker_cumulative,
        })

        if (fi + 1) % 10 == 0:
            print(f"[Progress] {fi+1}/{len(eval_frames)} frames processed...")

    cap.release()

    # Wrong parking recall
    if wrong_pk["gt_total"] > 0:
        wrong_pk["recall"] = round(wrong_pk["detected"] / wrong_pk["gt_total"] * 100, 1)
    else:
        wrong_pk["recall"] = 0.0
    total_wrong_pk = wrong_pk["detected"] + wrong_pk["fp"] + wrong_pk["missed"]
    wrong_pk["accuracy"] = round(wrong_pk["detected"] / total_wrong_pk * 100, 1) if total_wrong_pk > 0 else 0.0

    # Flickering & Stability
    # Flickering Rate = transitions / total_slot_checks (excluding first frame per slot)
    total_slot_checks = len(state_true)  # total (slot, frame) pairs evaluated
    flick_rate = flicker_transitions / total_slot_checks if total_slot_checks > 0 else 0.0
    flick_rate_pct = round(flick_rate * 100, 2)

    # Stability Score = % of (slot, frame) predictions matching GT
    correct_total = sum(1 for t, p in zip(state_true, state_pred) if t == p)
    stability_score = round(correct_total / total_slot_checks * 100, 2) if total_slot_checks > 0 else 0.0

    # Compute slot state confusion matrix
    slot_results, slot_cm = compute_confusion_matrix(
        state_true, state_pred, STATE_CATEGORIES)
    slot_results["_cm"] = slot_cm

    # Free / Occupied binary
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
            "accuracy":  round(acc * 100, 1),
            "precision": round(pp * 100, 1),
            "recall":    round(rr * 100, 1),
            "f1":        round(ff * 100, 1),
            "support":   sup,
            "tp": tp, "fp": fp, "fn": fn,
        }

    free_results = binary_metrics(state_true, state_pred, "available")
    occ_results  = binary_metrics(state_true, state_pred, "occupied")

    # Print
    print(fmt_table(
        slot_results, slot_cm, free_results, occ_results, wrong_pk,
        total_outside_tp, total_outside_fp, total_outside_fn,
        fps_estimate, args.conf, args.iou_threshold, len(eval_frames),
        flick_rate_pct, stability_score, flicker_transitions))

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    json_out = {
        "parking_state":    {k: v for k, v in slot_results.items() if k != "_cm"},
        "confusion_matrix": {"classes": STATE_CATEGORIES, "matrix": slot_cm},
        "free_slots":       free_results,
        "occupied_slots":   occ_results,
        "wrong_parking":    wrong_pk,
        "outside_vehicles": {
            "tp": total_outside_tp, "fp": total_outside_fp, "fn": total_outside_fn,
            "precision": round(total_outside_tp / (total_outside_tp + total_outside_fp) * 100, 1)
                        if (total_outside_tp + total_outside_fp) > 0 else 0,
            "recall": round(total_outside_tp / (total_outside_tp + total_outside_fn) * 100, 1)
                        if (total_outside_tp + total_outside_fn) > 0 else 0,
            "f1": round(2 * (total_outside_tp / (total_outside_tp + total_outside_fp)) *
                        (total_outside_tp / (total_outside_tp + total_outside_fn)) /
                        ((total_outside_tp / (total_outside_tp + total_outside_fp)) +
                         (total_outside_tp / (total_outside_tp + total_outside_fn))) * 100, 1)
                        if (total_outside_tp + total_outside_fp) > 0 and
                           (total_outside_tp + total_outside_fn) > 0 else 0,
        },
        "flickering_rate":    flick_rate_pct,
        "flickering_transitions": flicker_transitions,
        "stability_score":   stability_score,
        "yolo_confidence":  args.conf,
        "iou_threshold":    args.iou_threshold,
        "fps":             fps_estimate,
        "frames_evaluated": len(eval_frames),
    }
    json_path = output_dir / "parking_occupancy_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {json_path}")

    csv_path = output_dir / "parking_occupancy_per_frame.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame", "correct", "total", "accuracy",
            "gt_free", "gt_occupied", "gt_overlap",
            "pred_free", "pred_occupied", "pred_overlap",
            "outside_tp", "outside_fp", "outside_fn",
            "num_detections", "flickering_transitions_so_far",
        ])
        w.writeheader()
        w.writerows(per_frame)
    print(f"[Saved] {csv_path}")
    print("\n[Done]")


if __name__ == "__main__":
    main()
