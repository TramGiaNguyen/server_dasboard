#!/usr/bin/env python3
"""
eval_pipeline_benchmark.py

Benchmark each stage of the parking detection pipeline:

  Stage 1 - YOLO + ByteTrack:  model.track() with bytetrack config
  Stage 2 - Geometric Reasoning:  cv2.pointPolygonTest for outer polygon
  Stage 3 - Inner-Core Verification:  cv2.pointPolygonTest for inner polygon
  Stage 4 - Hysteresis State Machine:  counter + transition logic
  Stage 5 - Total Pipeline:  sum of all stages

Output:
  - ASCII table: Stage | Frames | Avg FPS | Min FPS | Max FPS | Avg ms/frame | p95 ms
  - CSV:  pipeline_benchmark_results.csv
  - JSON: pipeline_benchmark_results.json

Usage:
  python eval_pipeline_benchmark.py --device 0 --warmup 50 --benchmark-frames 500
  python eval_pipeline_benchmark.py --device cpu --warmup 10 --benchmark-frames 100
"""

import argparse
import csv
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

DEFAULT_VIDEO      = "static/video/CAM_PARKING.mp4"
DEFAULT_OUTPUT     = "eval_results/pipeline_benchmark"
DEFAULT_FRAMES     = 500
DEFAULT_WARMUP     = 50
DEFAULT_CONF       = 0.20

VEHICLE_CLASS_IDS = [2, 5, 7]
DEFAULT_VIDEO_FPS  = 30.0
DEFAULT_HYST_THRESH = 45

import torch  # imported early so device check works everywhere


def _detect_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        return gpu_count, gpu_name
    return 0, None


# ---------------------------------------------------------------------------
# Slot definitions (same as eval_method_comparison.py)
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
    _FAR_SLOTS       = set(range(0, 14)) | {17, 18}
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
    candidates = [
        video_path,
        PROJECT_ROOT / video_path,
        PROJECT_ROOT / DEFAULT_VIDEO.lstrip("/"),
    ]
    for vp in candidates:
        if Path(vp).exists():
            break

    sample_frame = None
    if Path(vp).exists():
        cap = cv2.VideoCapture(str(vp))
        if cap.isOpened():
            ret, sample_frame = cap.read()
            cap.release()

    if sample_frame is None:
        sample_frame = np.zeros((500, 1020, 3), dtype=np.uint8)

    h, w = sample_frame.shape[:2]
    sx, sy = w / 1020.0, h / 500.0

    raw_areas  = define_parking_areas()
    raw_inner  = create_inner_zones(raw_areas)
    raw_rects  = make_slot_rects(raw_areas)
    areas      = scale_areas(raw_areas, sx, sy)
    inner_areas = scale_areas(raw_inner, sx, sy)
    slot_rects = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                  for (x1,y1,x2,y2) in raw_rects]

    print(f"[Parking] {len(areas)} slots, baseline 1020x500 -> scaled to {w}x{h} "
          f"(sx={sx:.3f}, sy={sy:.3f})")
    return areas, inner_areas, slot_rects


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def classify_polygon_only(detections, areas, num_slots):
    """Geometric reasoning: centroid inside outer polygon."""
    states = ["available"] * num_slots
    for det in detections:
        cx, cy = det["cx"], det["cy"]
        for s in range(num_slots):
            if cv2.pointPolygonTest(
                    np.array(areas[s], np.int32), (cx, cy), False) >= 0:
                states[s] = "occupied"
    return states


def classify_polygon_inner_core(detections, areas, inner_areas, num_slots):
    """Geometric reasoning + Inner-Core verification."""
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


def update_hysteresis(detections, areas, inner_areas, num_slots,
                      hyst_thresh, counters, current):
    """Hysteresis state machine update. Returns new current states."""
    instant_states = classify_polygon_inner_core(detections, areas, inner_areas, num_slots)
    for s in range(num_slots):
        inst = instant_states[s]
        if inst == current[s]:
            counters[s] = 0
        else:
            counters[s] += 1
            if counters[s] >= hyst_thresh:
                current[s] = inst
                counters[s] = 0
    return current


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark each stage of the parking detection pipeline.")
    p.add_argument("--video",    default=DEFAULT_VIDEO,  help="Path to video")
    p.add_argument("--output",   default=DEFAULT_OUTPUT, help="Output directory")
    p.add_argument("--benchmark-frames", type=int, default=DEFAULT_FRAMES,
                   help="Number of frames to benchmark (default: 500)")
    p.add_argument("--warmup",   type=int, default=DEFAULT_WARMUP,
                   help="Warm-up frames before benchmark (default: 50)")
    p.add_argument("--conf",     type=float, default=DEFAULT_CONF,
                   help="YOLO confidence threshold (default: 0.30)")
    p.add_argument("--model-size", default="l",
                   choices=["n", "s", "m", "l"],
                   help="YOLO model size (default: l)")
    p.add_argument("--device",   default=None,
                   help="'cuda' (or '0') for GPU, 'cpu' for CPU. "
                        "Default: auto-detect — uses GPU if available, falls back to CPU")
    p.add_argument("--half",    action="store_true",
                   help="Enable FP16 half precision on GPU (20-40%% speedup, default: auto)")
    p.add_argument("--start-frame", type=int, default=100,
                   help="Frame to start benchmark from (default: 100)")
    p.add_argument("--hyst-thresh", type=int, default=DEFAULT_HYST_THRESH,
                   help="Hysteresis threshold in frames (default: 45)")
    p.add_argument("--use-cache",  action="store_true",
                   help="Only read from cache, skip YOLO inference")
    p.add_argument("--no-cache",  action="store_true",
                   help="Force re-run YOLO inference (ignore cache)")
    return p.parse_args()


def _normalize_device(d, use_half):
    gpu_count, gpu_name = _detect_gpu_info()
    if d is None:
        if gpu_count > 0:
            d = "cuda"
            print(f"[GPU] Found {gpu_count} GPU(s): {gpu_name}")
            print(f"[GPU] Half precision: {'enabled' if use_half else 'disabled'}")
        else:
            d = "cpu"
            print("[GPU] No GPU found — using CPU")
        return d, gpu_count > 0
    if d in ("cpu", "mps"):
        return "cpu", False
    if d.isdigit():
        return d, gpu_count > 0
    if d.startswith("cuda"):
        return d, gpu_count > 0
    return "cpu", gpu_count > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device, has_gpu = _normalize_device(args.device, args.half)
    model_name = f"yolov8{args.model_size}.pt"
    model_path = PROJECT_ROOT / "static" / "models" / model_name

    # Resolve video path
    candidates = [
        args.video,
        PROJECT_ROOT / args.video,
        PROJECT_ROOT / DEFAULT_VIDEO.lstrip("/"),
    ]
    video_path = None
    for vp in candidates:
        if Path(vp).exists():
            video_path = vp
            break
    if video_path is None:
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "yolo_detections_cache.pkl"
    if args.use_cache and cache_file.exists():
        print(f"[Cache] Loading from {cache_file}")
        with open(cache_file, "rb") as f:
            all_detections = pickle.load(f)
    else:
        # Load YOLO
        sys.path.insert(0, str(PROJECT_ROOT))
        from ultralytics import YOLO
        tracker_cfg = PROJECT_ROOT / "cfg" / "trackers" / "bytetrack_parking.yaml"

        print(f"[YOLO] Loading {model_path} on device={device}")
        model = YOLO(str(model_path))
        model.to(device)
        model.fuse()
        if args.half and device.startswith("cuda"):
            model.model.half()
            print("[YOLO] Half precision (FP16) enabled — expect 20-40% speedup")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid = cap.get(cv2.CAP_PROP_FPS)
        print(f"[Video] {total_frames} frames @ {fps_vid:.1f} FPS")

        # Warm-up
        print(f"[Warmup] {args.warmup} frames...")
        for _ in range(args.warmup):
            ret, frame = cap.read()
            if not ret:
                break
            _ = model.track(frame, persist=True, tracker=str(tracker_cfg),
                           conf=args.conf, iou=0.5, verbose=False)
        print(f"[Warmup] Done.")

        # Run inference on benchmark range
        start_f = args.start_frame
        end_f   = start_f + args.benchmark_frames
        print(f"\n[Inference] Benchmarking frames {start_f} -> {end_f} ({args.benchmark_frames} frames)")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frame_idx = 0
        all_detections = {}

        t0 = time.perf_counter()
        last_print = t0

        while frame_idx < args.benchmark_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = model.track(
                frame, persist=True, tracker=str(tracker_cfg),
                conf=args.conf, iou=0.5, verbose=False)

            detections = []
            boxes = results[0].boxes if (results and results[0].boxes is not None) else None
            if boxes is not None and boxes.id is not None:
                xyxy   = boxes.xyxy.cpu().numpy()
                cls_np = boxes.cls.cpu().numpy().astype(int)
                confs  = boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    ci = int(cls_np[i])
                    if ci not in VEHICLE_CLASS_IDS:
                        continue
                    x1, y1, x2, y2 = xyxy[i]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": cx, "cy": cy,
                        "class_id": ci, "conf": float(confs[i]),
                        "track_id": int(boxes.id[i].cpu().numpy()),
                    })

            all_detections[start_f + frame_idx] = detections

            now = time.perf_counter()
            if now - last_print >= 5.0:
                elapsed = now - t0
                done = frame_idx
                eta = elapsed / done * (args.benchmark_frames - done)
                print(f"  [{done}/{args.benchmark_frames} frames] "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                last_print = now

        cap.release()
        elapsed_total = time.perf_counter() - t0
        print(f"[Inference] Done in {elapsed_total:.1f}s | "
              f"{len(all_detections)} frames cached")

        with open(cache_file, "wb") as f:
            pickle.dump(all_detections, f)
        print(f"[Cache] Saved to {cache_file}")

    # Load parking zones
    areas, inner_areas, slot_rects = load_parking_zones(str(video_path))
    num_slots = len(areas)

    # Collect per-frame timings for each stage
    timings = {
        "yolo_track": [],
        "geometric":  [],
        "inner_core": [],
        "hysteresis": [],
        "total":      [],
    }

    print(f"\n[Benchmark] Processing {len(all_detections)} frames "
          f"through all pipeline stages...")

    # Hysteresis state (persistent across frames)
    hyst_counters = [0] * num_slots
    hyst_current  = ["available"] * num_slots

    for frame_idx in sorted(all_detections.keys()):
        detections = all_detections[frame_idx]

        # ---- Stage 1: YOLO + ByteTrack (already done, simulate zero cost
        #                in geometric reasoning phase since detections already
        #                include track IDs)
        t0_total = time.perf_counter()

        # ---- Stage 2: Geometric Reasoning (outer polygon centroid test)
        t2 = time.perf_counter()
        _ = classify_polygon_only(detections, areas, num_slots)
        t_geometric = (time.perf_counter() - t2) * 1000  # ms

        # ---- Stage 3: Inner-Core Verification (inner polygon centroid test)
        t3 = time.perf_counter()
        _ = classify_polygon_inner_core(detections, areas, inner_areas, num_slots)
        t_inner_core = (time.perf_counter() - t3) * 1000  # ms

        # ---- Stage 4: Hysteresis State Machine
        t4 = time.perf_counter()
        _ = update_hysteresis(detections, areas, inner_areas, num_slots,
                               args.hyst_thresh, hyst_counters, hyst_current)
        t_hysteresis = (time.perf_counter() - t4) * 1000  # ms

        t_total = (time.perf_counter() - t0_total) * 1000  # ms

        timings["geometric"].append(t_geometric)
        timings["inner_core"].append(t_inner_core)
        timings["hysteresis"].append(t_hysteresis)
        timings["total"].append(t_total)

    n = len(timings["geometric"])

    # ---- Compute statistics
    def stats(arr):
        arr = np.array(arr)
        return {
            "avg_ms":  float(np.mean(arr)),
            "min_ms":  float(np.min(arr)),
            "max_ms":  float(np.max(arr)),
            "p50_ms":  float(np.median(arr)),
            "p95_ms":  float(np.percentile(arr, 95)),
            "p99_ms":  float(np.percentile(arr, 99)),
            "avg_fps": float(1000.0 / np.mean(arr)) if np.mean(arr) > 0 else 0,
            "min_fps": float(1000.0 / np.max(arr)) if np.max(arr) > 0 else 0,
            "max_fps": float(1000.0 / np.min(arr)) if np.min(arr) > 0 else 0,
        }

    results = {
        "config": {
            "video":            str(video_path),
            "num_slots":        num_slots,
            "num_frames":       n,
            "warmup_frames":    args.warmup,
            "hyst_threshold":   args.hyst_thresh,
            "conf_threshold":    args.conf,
            "model":            model_name,
            "device":           device,
        },
        "stages": {
            "YOLO + ByteTrack": {
                "description":  "model.track() with bytetrack — YOLO inference + vehicle tracking",
                "avg_ms":       None,  # Already run; tracked separately below
                "avg_fps":      None,
            },
            "Geometric Reasoning": stats(timings["geometric"]),
            "Inner-Core Verification": stats(timings["inner_core"]),
            "Hysteresis State Machine": stats(timings["hysteresis"]),
            "Pipeline (Geo+IC+Hyst)": stats(timings["total"]),
        },
        "per_frame_ms": {
            "geometric": timings["geometric"],
            "inner_core": timings["inner_core"],
            "hysteresis": timings["hysteresis"],
            "total": timings["total"],
        },
    }

    # YOLO + ByteTrack timing from inference phase (re-extract from cache)
    # Re-run YOLO+BT on the same frames to get accurate timing.
    # Open video ONCE and read sequentially — no reopening/seeking per frame.
    print("\n[Benchmark] Re-running YOLO + ByteTrack for accurate timing...")
    sys.path.insert(0, str(PROJECT_ROOT))
    from ultralytics import YOLO
    tracker_cfg = PROJECT_ROOT / "cfg" / "trackers" / "bytetrack_parking.yaml"
    model = YOLO(str(model_path))
    model.to(device)
    model.fuse()
    if args.half and device.startswith("cuda"):
        model.model.half()

    frame_keys = sorted(all_detections.keys())
    sample_video_path = video_path

    # Open video once; read sequentially from start
    cap_timing = cv2.VideoCapture(str(sample_video_path))
    cap_timing.set(cv2.CAP_PROP_POS_FRAMES, frame_keys[0])
    prev_frame = None
    ret_prev, prev_frame = cap_timing.read()

    yolo_track_times = []
    for i, fidx in enumerate(frame_keys):
        if ret_prev and prev_frame is not None:
            t_yolo_start = time.perf_counter()
            _ = model.track(prev_frame, persist=(i > 0),
                           tracker=str(tracker_cfg),
                           conf=args.conf, iou=0.5, verbose=False)
            t_yolo_total = time.perf_counter() - t_yolo_start
            yolo_track_times.append(t_yolo_total * 1000)
        else:
            yolo_track_times.append(0.0)

        # Read next frame sequentially (no seek, no reopen)
        ret_prev, prev_frame = cap_timing.read()
        if not ret_prev:
            break

    cap_timing.release()

    if yolo_track_times:
        results["stages"]["YOLO + ByteTrack"] = stats(yolo_track_times)
    else:
        results["stages"]["YOLO + ByteTrack"] = {
            "avg_ms": None, "avg_fps": None,
            "note": "Could not measure YOLO+BT timing"
        }

    # Total pipeline including YOLO (yolo_track_times now has same length as timings["total"])
    total_pipeline_times = []
    n_yolo = len(yolo_track_times)
    n_geo  = len(timings["total"])
    for i in range(min(n_yolo, n_geo)):
        total_pipeline_times.append(timings["total"][i] + yolo_track_times[i])
    results["stages"]["Full Pipeline (YOLO+BT+Geo+IC+Hyst)"] = stats(total_pipeline_times)

    # ---- Print ASCII table
    print("\n" + "=" * 100)
    print("  PIPELINE BENCHMARK RESULTS")
    print("=" * 100)
    half_str = "FP16" if (args.half and device.startswith("cuda")) else "FP32"
    print(f"  Config: {model_name} | {device} ({half_str}) | {n} frames | {num_slots} slots")
    print(f"  Hysteresis threshold: {args.hyst_thresh} frames")
    print("-" * 100)

    header = f"  {'Stage':<40} {'Avg FPS':>8} {'Min FPS':>8} {'Max FPS':>8} {'Avg ms':>8} {'p95 ms':>8}"
    print(header)
    print("-" * 100)

    stage_order = [
        "YOLO + ByteTrack",
        "Geometric Reasoning",
        "Inner-Core Verification",
        "Hysteresis State Machine",
        "Pipeline (Geo+IC+Hyst)",
        "Full Pipeline (YOLO+BT+Geo+IC+Hyst)",
    ]

    for sname in stage_order:
        s = results["stages"].get(sname, {})
        if s.get("avg_fps") is not None:
            print(f"  {sname:<40} {s['avg_fps']:>8.1f} {s['min_fps']:>8.1f} "
                  f"{s['max_fps']:>8.1f} {s['avg_ms']:>8.2f} {s['p95_ms']:>8.2f}")
        else:
            print(f"  {sname:<40} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

    print("-" * 100)

    # ---- Save CSV
    csv_path = output_dir / "pipeline_benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Stage", "Avg FPS", "Min FPS", "Max FPS",
                         "Avg ms/frame", "p50 ms", "p95 ms", "p99 ms"])
        for sname in stage_order:
            s = results["stages"].get(sname, {})
            if s.get("avg_fps") is not None:
                writer.writerow([
                    sname,
                    f"{s['avg_fps']:.2f}",
                    f"{s['min_fps']:.2f}",
                    f"{s['max_fps']:.2f}",
                    f"{s['avg_ms']:.2f}",
                    f"{s['p50_ms']:.2f}",
                    f"{s['p95_ms']:.2f}",
                    f"{s['p99_ms']:.2f}",
                ])
    print(f"\n[Output] CSV saved: {csv_path}")

    # ---- Save JSON
    json_path = output_dir / "pipeline_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Output] JSON saved: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
