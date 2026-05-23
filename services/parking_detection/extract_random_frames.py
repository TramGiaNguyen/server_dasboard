#!/usr/bin/env python3
"""
extract_random_frames.py

Truoc khi annotation, chay script nay de lay 1000 so frame ngau nhien tu video.
So frame duoc save vao `annotations/frame_numbers.txt` de co dinh tap hop danh gia.

Usage:
    python extract_random_frames.py
    python extract_random_frames.py --num 1000 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import cv2


DEFAULT_VIDEO = "static/video/CAM_PARKING.mp4"
DEFAULT_NUM = 1000
DEFAULT_SEED = 42
DEFAULT_OUTPUT = "annotations/frame_numbers.txt"
DEFAULT_FRAMES_JSON = "annotations/frame_numbers.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract random frame numbers from video for annotation")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to video")
    parser.add_argument("--num", type=int, default=DEFAULT_NUM, help="Number of frames to extract")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output-txt", default=DEFAULT_OUTPUT, help="Output txt file (frame numbers, one per line)")
    parser.add_argument("--output-json", default=DEFAULT_FRAMES_JSON, help="Output JSON file")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if frame_numbers.txt already exists (reuse existing set)")
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    video_path = project_root / args.video

    # Check if already extracted
    existing_path = project_root / args.output_txt
    if args.skip_existing and existing_path.exists():
        with open(existing_path, "r") as f:
            frames = [int(line.strip()) for line in f if line.strip()]
        print(f"[Info] Reusing existing {len(frames)} frame numbers from {existing_path}")
        return frames

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"[Video] {width}x{height}, {total_frames} frames, {fps:.1f} FPS, {duration_sec:.1f}s duration")

    # Sample random frames (avoid first 30 and last 30 frames)
    rng = random.Random(args.seed)
    valid_range_start = 30
    valid_range_end = total_frames - 30

    if args.num > (valid_range_end - valid_range_start):
        raise ValueError(f"Cannot extract {args.num} frames from range {valid_range_end - valid_range_start}")

    frames = sorted(rng.sample(range(valid_range_start, valid_range_end), args.num))

    # Save as txt
    out_txt = project_root / args.output_txt
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        for fr in frames:
            f.write(f"{fr}\n")

    # Save as JSON (with metadata)
    out_json = project_root / args.output_json
    metadata = {
        "video": str(video_path),
        "video_width": width,
        "video_height": height,
        "total_frames": total_frames,
        "fps": fps,
        "num_frames": args.num,
        "seed": args.seed,
        "frame_numbers": frames,
    }
    with open(out_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Output] {len(frames)} frame numbers saved to:")
    print(f"  TXT:  {out_txt}")
    print(f"  JSON: {out_json}")

    print(f"\nFrame numbers ({len(frames)} frames):")
    print(json.dumps(frames, indent=2))


if __name__ == "__main__":
    main()
