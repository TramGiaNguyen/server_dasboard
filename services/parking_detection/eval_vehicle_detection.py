#!/usr/bin/env python3
"""
eval_vehicle_detection.py — Section 5.3 Vehicle Detection Performance Evaluation

Measures per-frame bounding-box detection quality against ground truth annotations.
Outputs: Precision, Recall, F1-score, Inference Speed (FPS), IoU Threshold.

Usage:
    python eval_vehicle_detection.py
    python eval_vehicle_detection.py --video static/video/CAM_PARKING.mp4
                                     --gt annotations/CAM_PARKING_gt.csv
                                     --model yolov8l.pt
                                     --output eval_results/my_test
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_VIDEO = "static/video/CAM_PARKING.mp4"
DEFAULT_GT = "annotations/CAM_PARKING_gt.csv"
DEFAULT_MODEL = "yolov8l.pt"
DEFAULT_OUTPUT = "eval_results/my_test"
DEFAULT_FRAME_LIST = "annotations/frame_numbers.txt"

IOU_THRESHOLD = 0.50
CONF_THRESHOLD = 0.20

# COCO vehicle classes: 2=car, 5=bus, 7=truck
VEHICLE_CLASS_IDS = [2, 5, 7]
VEHICLE_CLASS_NAMES = {2: "car", 5: "bus", 7: "truck"}


# ── Inline bbox_iou (avoids circular import) ──────────────────────────────────
def bbox_iou(a, b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    return float(inter / union) if union > 0.0 else 0.0


# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Section 5.3 Vehicle Detection Evaluation")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to video file")
    parser.add_argument("--gt", default=DEFAULT_GT, help="Path to ground truth CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLO model")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--frame-list", default=DEFAULT_FRAME_LIST,
                        help="Path to txt file containing frame numbers (one per line)")
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--fps-frames", type=int, default=200, help="Frames for FPS measurement")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Warmup frames before FPS")
    parser.add_argument("--device", default=None,
                        help="Device for inference, e.g. cpu, cuda:0. Default: auto-detect CUDA")
    return parser.parse_args()


def resolve_path(base, path_arg, default):
    """Resolve path relative to project root if not absolute."""
    p = Path(path_arg)
    if p.is_absolute():
        return str(p)
    resolved = PROJECT_ROOT / base / default if not p.exists() else str(p)
    candidates = [
        path_arg,
        PROJECT_ROOT / path_arg,
        PROJECT_ROOT / default,
    ]
    for c in candidates:
        if Path(c).exists():
            return str(Path(c).resolve())
    return str(PROJECT_ROOT / default)


def load_ground_truth(gt_path: str) -> dict:
    """
    Load ground truth from CSV.
    Returns: {frame_number: [(x1, y1, x2, y2, class_name), ...]}
    """
    gt = defaultdict(list)
    with open(gt_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            cls = row["class_name"].strip()
            gt[frame].append((x1, y1, x2, y2, cls))
    return dict(gt)


def compute_frame_metrics(
    gt_boxes: list,
    pred_boxes: list,
    iou_threshold: float
) -> dict:
    """
    Compute TP, FP, FN for a single frame.
    Matching: greedy — each GT box matched to the highest-IoU prediction above threshold.
    Each box matched at most once.
    Class is NOT checked — only bbox IoU matters.
    """
    tp = 0
    fp = 0
    fn = 0

    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)

    # Match GT to predictions by IoU only (no class check)
    for i, gt in enumerate(gt_boxes):
        best_iou = 0.0
        best_j = -1
        for j, pred in enumerate(pred_boxes):
            if pred_matched[j]:
                continue
            iou = bbox_iou(gt[:4], pred[:4])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            tp += 1
            gt_matched[i] = True
            pred_matched[best_j] = True

    # Count unmatched
    fn = sum(1 for m in gt_matched if not m)
    fp = sum(1 for m in pred_matched if not m)

    return {"tp": tp, "fp": fp, "fn": fn}


def format_table(metrics: dict, fps_metrics: dict, iou_threshold: float) -> str:
    """Build a formatted table for Section 5.3 output."""
    p = metrics["precision"]
    r = metrics["recall"]
    f1 = metrics["f1_score"]
    avg_fps = fps_metrics["avg_fps"]

    lines = [
        "=" * 60,
        "5.3 Vehicle Detection Performance",
        "=" * 60,
        f"{'Metric':<20} {'Result':>15}",
        "-" * 60,
        f"{'Precision':<20} {p:>14.1f}%",
        f"{'Recall':<20} {r:>14.1f}%",
        f"{'F1-score':<20} {f1:>14.1f}%",
        f"{'Inference Speed':<20} {avg_fps:>14.1f} FPS",
        f"{'IoU Threshold':<20} {iou_threshold:>14.2f}",
        "-" * 60,
        f"Frames evaluated:  {metrics['frames_evaluated']}",
        f"Total TP / FP / FN: {metrics['total_tp']} / {metrics['total_fp']} / {metrics['total_fn']}",
        "=" * 60,
    ]
    return "\n".join(lines)


def main():
    args = parse_args()

    # Resolve paths
    video_path = resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/"))
    gt_path = resolve_path("annotations", args.gt, DEFAULT_GT)
    frame_list_path = resolve_path("annotations", args.frame_list, DEFAULT_FRAME_LIST)
    model_path = resolve_path("", args.model, DEFAULT_MODEL)
    output_dir = PROJECT_ROOT / args.output

    print(f"[Config]")
    print(f"  Video:         {video_path}")
    print(f"  Ground truth:  {gt_path}")
    print(f"  Frame list:    {frame_list_path}")
    print(f"  Model:         {model_path}")
    print(f"  Output:        {output_dir}")
    print(f"  IoU thresh:    {args.iou_threshold}")
    print(f"  Conf thresh:   {args.conf_threshold}")

    # Validate inputs
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(gt_path).exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    if not Path(frame_list_path).exists():
        raise FileNotFoundError(f"Frame list not found: {frame_list_path}  "
                                "(run extract_random_frames.py first)")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load frame list
    with open(frame_list_path, "r") as f:
        frame_numbers = sorted(int(line.strip()) for line in f if line.strip())
    print(f"\n[Data] Loaded {len(frame_numbers)} frame numbers from list")

    # Load ground truth
    gt_data = load_ground_truth(gt_path)
    print(f"[Data] Loaded {len(gt_data)} annotated frames from GT")

    # Open video to get resolution
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Data] Video: {video_width}x{video_height}, {total_frames} frames, {fps_vid:.1f} FPS")

    # Load YOLO model
    print(f"\n[Model] Loading YOLO from {model_path} ...")
    model = YOLO(model_path)
    if args.device:
        device = args.device
    elif hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"[Model] Device: {device}")

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"\n[FPS] Warming up ({args.warmup_frames} frames)...")
    warmup_frames = 0
    while warmup_frames < args.warmup_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            break
        model.predict(frame, conf=args.conf_threshold, verbose=False, device=device)
        warmup_frames += 1

    # ── FPS Benchmark ─────────────────────────────────────────────────────────
    print(f"[FPS] Measuring on {args.fps_frames} frames (after warmup)...")
    fps_times = []
    fps_start = time.perf_counter()

    for i in range(args.fps_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        model.predict(frame, conf=args.conf_threshold, verbose=False, device=device)
        t1 = time.perf_counter()
        fps_times.append(t1 - t0)

    fps_total_elapsed = time.perf_counter() - fps_start

    fps_metrics = {
        "avg_fps": len(fps_times) / sum(fps_times) if fps_times else 0,
        "min_fps": 1 / max(fps_times) if fps_times else 0,
        "max_fps": 1 / min(fps_times) if fps_times else 0,
        "median_fps": 1 / np.median(fps_times) if fps_times else 0,
        "p95_fps": 1 / np.percentile(fps_times, 95) if fps_times else 0,
        "std_fps": np.std([1/t for t in fps_times]) if fps_times else 0,
        "frames_measured": len(fps_times),
        "warmup_frames": args.warmup_frames,
    }

    print(f"[FPS] Average: {fps_metrics['avg_fps']:.2f} FPS  (min: {fps_metrics['min_fps']:.2f}, max: {fps_metrics['max_fps']:.2f})")

    # ── Per-frame Evaluation ─────────────────────────────────────────────────
    print(f"\n[Eval] Evaluating {len(frame_numbers)} frames...")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_frame_results = []

    for idx, frame_num in enumerate(frame_numbers):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"  [WARN] Cannot read frame {frame_num}, skipping")
            continue

        # GT boxes at native resolution (no scaling needed)
        gt_boxes = gt_data.get(frame_num, [])

        # YOLO inference
        results = model.predict(frame, conf=args.conf_threshold, verbose=False, device=device)
        result = results[0]

        # Extract vehicle detections (bbox only, no class check)
        pred_boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_boxes.append((x1, y1, x2, y2))

        # Compute frame metrics
        frame_metrics = compute_frame_metrics(gt_boxes, pred_boxes, args.iou_threshold)
        total_tp += frame_metrics["tp"]
        total_fp += frame_metrics["fp"]
        total_fn += frame_metrics["fn"]

        p = frame_metrics["tp"] / (frame_metrics["tp"] + frame_metrics["fp"]) if (frame_metrics["tp"] + frame_metrics["fp"]) > 0 else 0.0
        r = frame_metrics["tp"] / (frame_metrics["tp"] + frame_metrics["fn"]) if (frame_metrics["tp"] + frame_metrics["fn"]) > 0 else 0.0

        per_frame_results.append({
            "frame": frame_num,
            "tp": frame_metrics["tp"],
            "fp": frame_metrics["fp"],
            "fn": frame_metrics["fn"],
            "num_gt": len(gt_boxes),
            "num_pred": len(pred_boxes),
            "precision": round(p, 4),
            "recall": round(r, 4),
        })

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(frame_numbers)}] frame={frame_num}: TP={frame_metrics['tp']}, FP={frame_metrics['fp']}, FN={frame_metrics['fn']}")

    cap.release()

    # ── Aggregate metrics ────────────────────────────────────────────────────
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics_summary = {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1_score": round(f1 * 100, 1),
        "precision_raw": round(precision, 4),
        "recall_raw": round(recall, 4),
        "f1_score_raw": round(f1, 4),
        "frames_evaluated": len(per_frame_results),
    }

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + format_table(metrics_summary, fps_metrics, args.iou_threshold))

    # ── Save results ──────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_out = {
        "fps": {k: float(v) for k, v in fps_metrics.items()},
        "metrics": {**metrics_summary, "per_frame": per_frame_results},
    }
    json_path = output_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {json_path}")

    # Per-frame CSV
    csv_path = output_dir / "per_frame_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "tp", "fp", "fn", "num_gt", "num_pred", "precision", "recall"])
        writer.writeheader()
        writer.writerows(per_frame_results)
    print(f"[Saved] {csv_path}")

    # FPS CSV
    fps_csv_path = output_dir / "fps_log.csv"
    with open(fps_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stat", "value"])
        writer.writeheader()
        for k, v in fps_metrics.items():
            writer.writerow({"stat": k, "value": f"{v:.4f}"})
    print(f"[Saved] {fps_csv_path}")

    print("\n[Done] Evaluation complete.")


if __name__ == "__main__":
    main()
