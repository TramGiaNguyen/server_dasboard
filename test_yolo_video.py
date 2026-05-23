"""
test_yolo_video.py
==================
Standalone test script: loads YOLOv8-Large from config, reads the RTSP stream
configured in .env (PARKING_RTSP_URL), and renders a live window with:

  - Bounding box around each detected vehicle
  - Confidence score
  - Class label
  - Centre-point (cx, cy) coordinates

Press 'q' to quit.
"""

import os
import sys
import cv2
import torch
import numpy as np

# ── Project paths ────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

# ── Load environment ─────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(_BASE_DIR, ".env"))

# ── Config ───────────────────────────────────────────────────────────────────
RTSP_URL     = os.getenv("PARKING_RTSP_URL", "rtsp://localhost:8554/cam_parking")
MODEL_PATH   = os.path.join(_BASE_DIR, "static", "models", "yolov8l.pt")
COCO_PATH    = os.path.join(_BASE_DIR, "coco.txt")
CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", "0.35"))

# Display window size (keep aspect ratio)
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 640

# COCO vehicle classes (COCO class indices: 2=car, 5=bus, 7=truck)
# Index matches the line number in coco.txt (0-based: person=0, bicycle=1, car=2, ...)
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

# BGR palette – one colour per class so vehicles are easy to distinguish
CLASS_PALETTE = {
    "car":   (0, 255, 255),   # Yellow
    "bus":   (255, 165, 0),   # Orange
    "truck": (128, 0, 255),   # Purple
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_coco_classes(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def initialize_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] Device: {device}")

    if os.path.exists(model_path):
        print(f"[MODEL] Loading: {model_path}")
    else:
        print(f"[MODEL] Model not found at {model_path} – Ultralytics will download yolov8l.pt on first run.")
        model_path = "yolov8l"   # let ultralytics handle download

    from ultralytics import YOLO
    model = YOLO(model_path)
    model.to(device)
    model.fuse()   # merge Conv+BN for ~5-10 % faster inference
    return model


def draw_detection(frame: np.ndarray, box: np.ndarray, conf: float,
                  cls_idx: int, cls_name: str, track_id=None) -> None:
    """Draw one detection on the frame: bbox + label + centre dot."""
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_PALETTE.get(cls_name, (0, 255, 0))

    # Bounding box
    thickness = max(1, int(2.0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Centre point
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    dot_r = max(3, int(5.0))
    cv2.circle(frame, (cx, cy), dot_r, (0, 0, 255), -1)   # red dot
    cv2.circle(frame, (cx, cy), dot_r + 2, (255, 255, 255), 1)   # white ring

    # Label text
    label = f"{cls_name.upper()} {conf:.2f}"
    if track_id is not None:
        label = f"ID:{track_id} {cls_name.upper()} {conf:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = max(1, int(1.5))
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    # Background pill for label
    pad_x, pad_y = 4, 4
    lx1 = x1
    ly1 = max(y1 - th - pad_y * 2, 0)
    lx2 = lx1 + tw + pad_x * 2
    ly2 = ly1 + th + pad_y * 2
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.putText(frame, label, (lx1 + pad_x, ly2 - pad_y - 2),
                font, font_scale, (0, 0, 0), text_thickness)

    # Centre-point coordinate text (bottom-right of box)
    coord_text = f"({cx}, {cy})"
    (tw_c, th_c), _ = cv2.getTextSize(coord_text, font, 0.4, 1)
    ct_x = min(x2 - tw_c - 4, frame.shape[1] - tw_c - 4)
    ct_y = y2 + th_c + 4
    cv2.putText(frame, coord_text, (ct_x, ct_y),
                font, 0.4, (255, 255, 255), 1)


def annotate_frame(frame: np.ndarray, results, class_list: list) -> np.ndarray:
    """Draw all vehicle detections from ultralytics results."""
    if results is None or not results[0]:
        return frame

    res = results[0]
    if res.boxes is None:
        return frame

    boxes   = res.boxes
    xyxy    = boxes.xyxy.cpu().numpy()
    confs   = boxes.conf.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None

    for i in range(len(xyxy)):
        ci = cls_arr[i]
        if ci not in VEHICLE_CLASSES:
            continue

        conf = float(confs[i])
        if conf < CONFIDENCE:
            continue

        cls_name = VEHICLE_CLASSES[ci]
        tid = int(track_ids[i]) if track_ids is not None else None
        draw_detection(frame, xyxy[i], conf, ci, cls_name, track_id=tid)

    return frame


def resize_keep_aspect(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def main():
    print("=" * 60)
    print("  YOLOv8-Large – Parking Camera Test Window")
    print("=" * 60)
    print(f"  RTSP URL : {RTSP_URL}")
    print(f"  Model    : {MODEL_PATH}")
    print(f"  Confidence threshold: {CONFIDENCE}")
    print("  Press 'q' to quit")
    print("=" * 60)

    # Load model
    model = initialize_model(MODEL_PATH)

    # Load COCO class names
    coco_candidates = [
        COCO_PATH,
        os.path.join(_BASE_DIR, "coco.txt"),
    ]
    class_list = []
    for cp in coco_candidates:
        if os.path.exists(cp):
            class_list = load_coco_classes(cp)
            print(f"[INFO] Loaded {len(class_list)} COCO classes from: {cp}")
            break
    if not class_list:
        print("[WARN] coco.txt not found – using Ultralytics default class list.")
        class_list = None

    # Open RTSP stream
    print("[STREAM] Opening RTSP capture …")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open RTSP stream: {RTSP_URL}")
        print("       Make sure the simulator is running:")
        print("         cd simulator && docker-compose up -d && python stream.py")
        sys.exit(1)

    print("[STREAM] Connected.")

    # Optional FPS tracking
    fps_time = cv2.getTickCount()
    fps_display = 0.0
    frame_count = 0

    window_name = "YOLOv8-Large – Parking Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Frame grab failed – retrying …")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print("[ERROR] Cannot reconnect to stream.")
                break
            continue

        # ── Inference ──────────────────────────────────────────────────────────
        results = model(frame, conf=CONFIDENCE, iou=0.5, verbose=False)

        # ── Annotate ───────────────────────────────────────────────────────────
        annotated = annotate_frame(frame.copy(), results, class_list)

        # ── FPS overlay ────────────────────────────────────────────────────────
        frame_count += 1
        if frame_count % 10 == 0:
            fps_now = cv2.getTickFrequency() / (cv2.getTickCount() - fps_time) * 10
            fps_display = fps_now if fps_display == 0 else (fps_display * 0.8 + fps_now * 0.2)
            fps_time = cv2.getTickCount()

        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Legend
        legend_items = [
            ("CAR",   CLASS_PALETTE["car"]),
            ("BUS",   CLASS_PALETTE["bus"]),
            ("TRUCK", CLASS_PALETTE["truck"]),
        ]
        lx = 10
        ly = annotated.shape[0] - 15
        cv2.putText(annotated, "Legend:", (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        lx += 75
        for name, color in legend_items:
            cv2.rectangle(annotated, (lx, ly - 14), (lx + 14, ly), color, -1)
            cv2.putText(annotated, name, (lx + 18, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            lx += 90

        # Resize to display window
        display = resize_keep_aspect(annotated, WINDOW_WIDTH, WINDOW_HEIGHT)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("\n[QUIT] User requested exit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE]")


if __name__ == "__main__":
    main()
