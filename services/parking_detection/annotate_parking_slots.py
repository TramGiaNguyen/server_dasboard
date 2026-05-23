#!/usr/bin/env python3
"""
annotate_parking_slots.py

Cong cu annotation trang thai parking slot + bbox xe ngoai.

Tinh nang:
- Hien thi video frame + slot polygons
- Click vao slot de thay doi trang thai: available -> occupied -> overlapping
- Ve bbox nhu annotate_vehicles.py cho cac xe nam ngoai vung cho do xe
- Luu ra CSV: slot_state + outside bboxes

Su dung:
    python annotate_parking_slots.py
    python annotate_parking_slots.py --continue
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from services.parking_detection.detection import define_parking_areas, create_inner_zones


DEFAULT_VIDEO      = "static/video/CAM_PARKING.mp4"
DEFAULT_FRAME_LIST = "annotations/frame_numbers.txt"
DEFAULT_SLOT_OUT   = "annotations/CAM_PARKING_slot_state_gt.csv"
DEFAULT_BBOX_OUT   = "annotations/CAM_PARKING_bbox_outside_gt.csv"
DEFAULT_WINDOW     = "Annotate Slots - ESC:quit  TAB:mode  Space:next  Click:annotate"


# ── YOLO vehicle detection ────────────────────────────────────────────────────
_YOLO_CLASSES  = {2: "car", 5: "bus", 7: "truck"}
_YOLO_COLORS   = {2: (0, 255, 0), 5: (0, 165, 255), 7: (0, 0, 255)}
_YOLO_CONFIDENCE = 0.30
_YOLO_IOU       = 0.45


# ── Slot state ──────────────────────────────────────────────────────────────

STATE_CYCLE   = ["available", "occupied", "overlapping"]
STATE_COLORS  = {
    "available":  (0, 255, 0),
    "occupied":   (0, 0, 255),
    "overlapping": (0, 165, 255),
}
STATE_DISPLAY = {
    "available":  "FREE",
    "occupied":  "OCCUPIED",
    "overlapping": "WRONG",
}

# ── BBox class ─────────────────────────────────────────────────────────────

BBOX_CLASS_NAMES  = ["outside"]       # only "outside" class needed here
BBOX_CLASS_DISP   = {0: "OUTSIDE"}
BBOX_COLORS = {
    0: (255, 0, 255),    # tim - xe ngoai
}


def parse_args():
    p = argparse.ArgumentParser(description="Annotate parking slots + outside bboxes")
    p.add_argument("--video",      default=DEFAULT_VIDEO,      help="Path to video")
    p.add_argument("--frame-list",  default=DEFAULT_FRAME_LIST, help="Frame numbers file")
    p.add_argument("--slot-out",    default=DEFAULT_SLOT_OUT,    help="Output slot state CSV")
    p.add_argument("--bbox-out",    default=DEFAULT_BBOX_OUT,    help="Output outside bbox CSV")
    p.add_argument("--continue",    dest="continue_mode", action="store_true")
    return p.parse_args()


def resolve_path(base, path_arg, default):
    project_root = Path(__file__).resolve().parent.parent.parent
    for c in [path_arg, project_root / path_arg]:
        if Path(c).exists():
            return str(Path(c).resolve())
    return str(project_root / default)


class SlotBBoxAnnotator:
    def __init__(self, video_path, frame_numbers,
                 slot_out, bbox_out, continue_mode=False):

        self.video_path = video_path
        self.frame_numbers = frame_numbers
        self.slot_out = slot_out
        self.bbox_out = bbox_out
        self.continue_mode = continue_mode

        # Slot polygons + inner zones
        self.slot_polygons = define_parking_areas()
        self.slot_inner_polygons = create_inner_zones(self.slot_polygons)
        self.num_slots = len(self.slot_polygons)
        print(f"[Info] {self.num_slots} parking slots loaded")

        # Video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Khong mo duoc video: {video_path}")
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale_x = self.width / 1020.0
        self.scale_y = self.height / 500.0

        # YOLO model
        self._init_yolo()

        # State
        self.slot_annotations = {}   # {(frame, slot): state}
        self.bbox_annotations = {}   # {frame: [(x1,y1,x2,y2,cls),...]}
        self.change_log = []

        # Toggles
        self.show_inner = True
        self.show_yolo  = True   # overlay YOLO detections

        if self.continue_mode:
            self._load_existing()

        self.frame_idx = 0
        self._load_frame()

        # Annotation mode: "slot" or "bbox"
        self.mode = "slot"

        # BBox drawing state
        self.drawing = False
        self.start_pt = None
        self.temp_box = None
        self.selected_box_idx = -1

        cv2.namedWindow(DEFAULT_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEFAULT_WINDOW, self.width, self.height)
        cv2.setMouseCallback(DEFAULT_WINDOW, self._on_mouse)
        self._render()

    def _load_existing(self):
        if Path(self.slot_out).exists():
            with open(self.slot_out, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.slot_annotations[(int(row["frame"]), int(row["slot"]))] = row["state"].strip()
            print(f"[Info] Da load {len(set(f for f,s in self.slot_annotations))} frames slot")

        if Path(self.bbox_out).exists():
            with open(self.bbox_out, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    frame = int(row["frame"])
                    box = (float(row["x1"]), float(row["y1"]),
                           float(row["x2"]), float(row["y2"]), 0)
                    self.bbox_annotations.setdefault(frame, []).append(box)
            print(f"[Info] Da load {len(self.bbox_annotations)} frames bbox")

    def _scaled_poly(self, poly):
        return np.array([(int(x*self.scale_x), int(y*self.scale_y))
                        for x, y in poly], np.int32)

    def _load_frame(self):
        self.current_frame = self.frame_numbers[self.frame_idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, self.frame = self.cap.read()
        if not ret:
            self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # YOLO detection (cached)
        if self.yolo_model is not None:
            if self.current_frame not in self.yolo_detections:
                self.yolo_detections[self.current_frame] = self._run_yolo(self.frame)

        # Load slot states
        self.current_slot_states = {}
        for slot in range(self.num_slots):
            key = (self.current_frame, slot)
            self.current_slot_states[slot] = self.slot_annotations.get(key)

        # Load bboxes
        self.current_boxes = list(self.bbox_annotations.get(self.current_frame, []))
        self.drawing = False
        self.start_pt = None
        self.temp_box = None
        self.selected_box_idx = -1
        self.change_log = []

    # ── YOLO ──────────────────────────────────────────────────────────────────

    def _init_yolo(self):
        model_path = Path(__file__).resolve().parent.parent.parent / "static" / "models" / "yolov8l.pt"
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(str(model_path))
            self.yolo_detections = {}  # cache: frame -> list of detections
            print(f"[YOLO] Model loaded: {model_path}")
        except Exception as e:
            self.yolo_model = None
            print(f"[YOLO] Warning: could not load model ({e}). YOLO overlay disabled.")

    def _run_yolo(self, frame):
        if self.yolo_model is None:
            return []
        try:
            results = self.yolo_model(
                frame,
                conf=_YOLO_CONFIDENCE,
                iou=_YOLO_IOU,
                classes=[2, 5, 7],
                verbose=False,
                device="cpu",
            )
            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf   = float(box.conf[0].cpu().numpy())
                    detections.append((x1, y1, x2, y2, cls_id, conf))
            return detections
        except Exception as e:
            print(f"[YOLO] Inference error: {e}")
            return []

    def _draw_yolo_detections(self, canvas, detections):
        for det in detections:
            x1, y1, x2, y2, cls_id, conf = det
            color = _YOLO_COLORS.get(cls_id, (128, 128, 128))
            thick = max(1, int(1.5 * self.scale_x))

            # BBox
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)

            # Label
            name = _YOLO_CLASSES.get(cls_id, f"C{cls_id}")
            label = f"{name} {conf:.2f}"
            lx, ly = int(x1), max(0, int(y1) - 6)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(canvas, (lx, ly - lh - 2), (lx + lw + 4, ly + 2), color, -1)
            cv2.putText(canvas, label, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Center dot
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(canvas, (cx, cy), 4, color, -1)
            cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), 1)
            cv2.putText(canvas, f"({cx},{cy})", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    # ── Slot helpers ────────────────────────────────────────────────────────

    def _point_in_poly(self, px, py, poly):
        return cv2.pointPolygonTest(poly, (px, py), False) >= 0

    def _find_slot_at(self, x, y):
        for slot, poly in enumerate(self.slot_polygons):
            sp = self._scaled_poly(poly)
            if self._point_in_poly(x, y, sp):
                return slot
        return -1

    def _cycle_slot_state(self, current):
        if current is None:
            return "available"
        idx = STATE_CYCLE.index(current) if current in STATE_CYCLE else -1
        return STATE_CYCLE[(idx + 1) % len(STATE_CYCLE)]

    # ── BBox helpers ────────────────────────────────────────────────────────

    def _point_in_box(self, px, py, box):
        x1, y1, x2, y2, _ = box
        return x1 <= px <= x2 and y1 <= py <= y2

    def _draw_bbox(self, img, x1, y1, x2, y2, cls_idx, thickness=2):
        color = BBOX_COLORS.get(cls_idx, (255, 0, 255))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        label = BBOX_CLASS_DISP.get(cls_idx, "?")
        lx, ly = int(x1), max(0, int(y1) - 8)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (lx, ly - lh - 2), (lx + lw + 4, ly + 2), color, -1)
        cv2.putText(img, label, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Render ──────────────────────────────────────────────────────────────

    def _render(self):
        canvas = self.frame.copy()

        # YOLO detections overlay (semi-transparent so slots still visible)
        if self.show_yolo and self.yolo_model is not None:
            detections = self.yolo_detections.get(self.current_frame, [])
            if detections:
                # Draw on a copy for transparency
                yolo_canvas = canvas.copy()
                self._draw_yolo_detections(yolo_canvas, detections)
                cv2.addWeighted(yolo_canvas, 0.70, canvas, 0.30, 0, canvas)

        # Ve slot polygons
        for slot, poly in enumerate(self.slot_polygons):
            sp = self._scaled_poly(poly)
            state = self.current_slot_states.get(slot)
            color = STATE_COLORS.get(state, (80, 80, 80))

            if state:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [sp], color)
                cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

            cv2.polylines(canvas, [sp], True, color, 2)

            M = cv2.moments(sp)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                xs = [p[0] for p in sp]; ys = [p[1] for p in sp]
                cx = sum(xs) // len(xs); cy = sum(ys) // len(ys)

            lbl = f"{slot}"
            if state:
                lbl += f":{STATE_DISPLAY.get(state, state)}"
            lcolor = color if state else (150, 150, 150)
            cv2.putText(canvas, lbl, (cx - 15, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, lcolor, 1)

            # Ve inner zone (dashed line, white/gray)
            if self.show_inner:
                inner_sp = self._scaled_poly(self.slot_inner_polygons[slot])
                cv2.polylines(canvas, [inner_sp], True, (220, 220, 220), 1, lineType=cv2.LINE_4)

        # Ve bbox
        for i, box in enumerate(self.current_boxes):
            is_sel = (i == self.selected_box_idx)
            thickness = 3 if is_sel else 2
            self._draw_bbox(canvas, *box, thickness=thickness)
            if is_sel:
                x1, y1, x2, y2, ci = box
                cv2.circle(canvas, (int((x1+x2)/2), int((y1+y2)/2)), 5, (255, 255, 0), -1)

        if self.temp_box and self.start_pt:
            x1, y1, x2, y2 = self.temp_box
            self._draw_bbox(canvas, x1, y1, x2, y2, 0, 2)

        # HUD - top bar
        h = 25
        mode_color = (0, 255, 0) if self.mode == "slot" else (255, 0, 255)
        mode_name  = "SLOT" if self.mode == "slot" else "BBOX-OUT"
        cv2.putText(canvas,
                    f"Frame {self.current_frame}  [{self.frame_idx+1}/{len(self.frame_numbers)}]  "
                    f"Mode: [{mode_name}]",
                    (5, h), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2)
        h += 28

        if self.mode == "slot":
            annotated = sum(1 for v in self.current_slot_states.values() if v is not None)
            detections = self.yolo_detections.get(self.current_frame, [])
            det_count = len(detections)
            cv2.putText(canvas,
                        f"Slots: {annotated}/{self.num_slots}  YOLO: {det_count}  [Click] slot to toggle  [1] free  [2] occu  [3] wrong",
                        (5, h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            hint = "[TAB] Bbox mode  |  Click OUTSIDE slot to draw bbox"
        else:
            cv2.putText(canvas,
                        f"Outside bboxs: {len(self.current_boxes)}  [Click+drag] draw bbox  [Del] delete sel  [D] delete last",
                        (5, h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            hint = "[TAB] Slot mode"
        h += 28

        # Legend
        legend_y = h + 32
        if self.mode == "slot":
            for i, (st, lbl) in enumerate([
                ("available",    "FREE"),
                ("occupied",     "OCCUPIED"),
                ("overlapping",  "WRONG"),
            ]):
                lx = 5 + i * 140
                cv2.rectangle(canvas, (lx, legend_y), (lx + 120, legend_y + 18),
                              STATE_COLORS[st], -1)
                cv2.putText(canvas, f"[{i+1}] {lbl}", (lx+5, legend_y+13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        else:
            lx = 5
            cv2.rectangle(canvas, (lx, legend_y), (lx + 120, legend_y + 18), BBOX_COLORS[0], -1)
            cv2.putText(canvas, "[OUTSIDE]", (lx+5, legend_y+13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # Inner zone legend
        inner_legend_x = 480
        cv2.rectangle(canvas, (inner_legend_x, legend_y), (inner_legend_x + 170, legend_y + 18),
                      (80, 80, 80), 1)
        cv2.putText(canvas, f"INNER [{'ON' if self.show_inner else 'OFF'}]  [I]",
                    (inner_legend_x + 5, legend_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # YOLO legend
        yolo_legend_x = inner_legend_x + 180
        yolo_legend_color = (0, 255, 0) if self.show_yolo else (60, 60, 60)
        cv2.rectangle(canvas, (yolo_legend_x, legend_y), (yolo_legend_x + 170, legend_y + 18),
                      yolo_legend_color, -1)
        cv2.putText(canvas, f"YOLO [{'ON' if self.show_yolo else 'OFF'}]  [Y]",
                    (yolo_legend_x + 5, legend_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        h = legend_y + 28
        cv2.putText(canvas,
                    "[TAB] mode  [Space] next  [B] prev  [R] reset  [I] inner  [Y] yolo  [ESC] quit+save",
                    (5, h), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        h += 22
        cv2.putText(canvas, hint,
                    (5, h), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_color, 1)

        if self.selected_box_idx >= 0 and self.mode == "bbox":
            cv2.putText(canvas,
                        f"Selected bbox: {self.selected_box_idx}  [Del] to delete",
                        (5, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        if self.drawing and self.start_pt:
            cv2.putText(canvas,
                        "DRAWING... release to confirm bbox",
                        (5, h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        cv2.imshow(DEFAULT_WINDOW, canvas)

    # ── Mouse ───────────────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
        # Always handle mouse move for live preview
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_pt:
                sx, sy = self.start_pt
                self.temp_box = (min(sx, x), min(sy, y),
                                 max(sx, x), max(sy, y))
                self._render()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "slot":
                slot = self._find_slot_at(x, y)
                if slot >= 0:
                    old = self.current_slot_states.get(slot)
                    new = self._cycle_slot_state(old)
                    self.current_slot_states[slot] = new
                    self.change_log.append(("slot", slot, old, new))
                    self._render()
                else:
                    # Clicked outside any slot -> switch to bbox mode and start drawing
                    self.mode = "bbox"
                    self.drawing = True
                    self.start_pt = (x, y)
                    self.temp_box = (x, y, x, y)
                    self.selected_box_idx = -1
                    self._render()
            else:
                # BBox mode
                # Check if clicked on existing bbox
                hit_idx = -1
                for i, box in enumerate(self.current_boxes):
                    if self._point_in_box(x, y, box):
                        hit_idx = i
                        break
                if hit_idx >= 0:
                    self.selected_box_idx = hit_idx
                else:
                    # Start new bbox
                    self.drawing = True
                    self.start_pt = (x, y)
                    self.temp_box = (x, y, x, y)
                    self.selected_box_idx = -1
                self._render()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == "bbox" and self.drawing and self.start_pt:
                sx, sy = self.start_pt
                if abs(x - sx) > 5 and abs(y - sy) > 5:
                    box = (min(sx, x), min(sy, y),
                           max(sx, x), max(sy, y), 0)
                    self.current_boxes.append(box)
                self.drawing = False
                self.start_pt = None
                self.temp_box = None
                self._render()

    # ── Actions ────────────────────────────────────────────────────────────

    def _toggle_mode(self):
        self.mode = "bbox" if self.mode == "slot" else "slot"
        self.selected_box_idx = -1
        self.drawing = False
        self.start_pt = None
        self.temp_box = None
        self._render()

    def _delete_selected(self):
        if 0 <= self.selected_box_idx < len(self.current_boxes):
            self.current_boxes.pop(self.selected_box_idx)
            self.selected_box_idx = -1
            self._render()

    def _delete_last(self):
        if self.mode == "bbox" and self.current_boxes:
            self.current_boxes.pop()
            if self.selected_box_idx >= len(self.current_boxes):
                self.selected_box_idx = -1
            self._render()
        elif self.mode == "slot" and self.change_log:
            last = self.change_log.pop()
            if last[0] == "slot":
                _, slot, old, _ = last
                self.current_slot_states[slot] = old
                self._render()

    def _reset_frame(self):
        for slot in range(self.num_slots):
            self.current_slot_states[slot] = None
        self.current_boxes.clear()
        self.change_log.clear()
        self.selected_box_idx = -1
        self._render()

    def _set_all_slots(self, state):
        for slot in range(self.num_slots):
            self.current_slot_states[slot] = state
        self.change_log.clear()
        self._render()

    def _save(self):
        # Slot state
        for slot, state in self.current_slot_states.items():
            if state is not None:
                self.slot_annotations[(self.current_frame, slot)] = state

        Path(self.slot_out).parent.mkdir(parents=True, exist_ok=True)
        frames = sorted(set(f for f, s in self.slot_annotations))
        slot_rows = []
        for frame in frames:
            for slot in range(self.num_slots):
                if (frame, slot) in self.slot_annotations:
                    slot_rows.append({"frame": frame, "slot": slot,
                                      "state": self.slot_annotations[(frame, slot)]})
        with open(self.slot_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "slot", "state"])
            w.writeheader()
            w.writerows(slot_rows)

        # BBox outside
        self.bbox_annotations[self.current_frame] = list(self.current_boxes)
        Path(self.bbox_out).parent.mkdir(parents=True, exist_ok=True)
        bbox_rows = []
        for frame in sorted(self.bbox_annotations):
            for box in self.bbox_annotations[frame]:
                x1, y1, x2, y2, ci = box
                bbox_rows.append({
                    "frame": frame, "class_name": "outside",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        with open(self.bbox_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "class_name", "x1", "y1", "x2", "y2"])
            w.writeheader()
            w.writerows(bbox_rows)

    def _next_frame(self):
        if self.frame_idx < len(self.frame_numbers) - 1:
            self._save()
            prev_slot_states = dict(self.current_slot_states)
            prev_boxes = list(self.current_boxes)
            self.frame_idx += 1
            self.current_frame = self.frame_numbers[self.frame_idx]
            self._load_frame()
            if self.current_frame not in self.slot_annotations:
                self.current_slot_states = dict(prev_slot_states)
            if self.current_frame not in self.bbox_annotations:
                self.current_boxes = list(prev_boxes)
            self._render()

    def _prev_frame(self):
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self._load_frame()
            self._render()

    def run(self):
        print("\n=== Annotation Instructions ===")
        print("  [TAB]      Chuyen doi che do: SLOT <-> BBOX-OUT")
        print("")
        print("  CHE DO SLOT (man hinh xanh la):")
        print("    Click slot     Chuyen doi: free -> occupied -> wrong")
        print("    [1]            Tat ca = available (free)")
        print("    [2]            Tat ca = occupied")
        print("    [3]            Tat ca = overlapping (wrong parking)")
        print("    [I]            An/Hien inner zone (duong trong)")
        print("    [Y]            An/Hien YOLO detections (bbox + tam)")
        print("    [D]            Undo thay doi cuoi")
        print("")
        print("  CHE DO BBOX (man hinh tim):")
        print("    Click+drag     Ve bbox cho xe nam ngoai vung cho do")
        print("    Click bbox     Chon bbox (highlight vang)")
        print("    [Del]          Xoa bbox da chon")
        print("    [D]            Xoa bbox cuoi cung")
        print("")
        print("  CHUNG:")
        print("    [Space]  Frame tiep (+ tu dong copy tu frame truoc)")
        print("    [B]      Frame lui")
        print("    [R]      Reset frame nay")
        print("    [ESC]    Thoat + luu tat ca")
        print("================================\n")
        print(f"[Info] {self.width}x{self.height}")
        print(f"[Info] Slots: {self.num_slots}")
        print(f"[Info] Frames: {len(self.frame_numbers)}")
        print(f"[Info] Slot output:  {self.slot_out}")
        print(f"[Info] BBox output:  {self.bbox_out}")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                print("\n[Info] Thoat va luu...")
                self._save()
                self._print_summary()
                break
            elif key == 9:   # TAB
                self._toggle_mode()
            elif key == 32:  # Space
                self._next_frame()
            elif key in (ord('b'), ord('B')):
                self._prev_frame()
            elif key in (ord('d'), ord('D')):
                self._delete_last()
            elif key in (ord('r'), ord('R')):
                self._reset_frame()
            elif key == 46:  # Delete
                self._delete_selected()
            elif key in (ord('1'),):
                self._set_all_slots("available")
            elif key in (ord('2'),):
                self._set_all_slots("occupied")
            elif key in (ord('3'),):
                self._set_all_slots("overlapping")
            elif key in (ord('i'), ord('I')):
                self.show_inner = not self.show_inner
                self._render()
            elif key in (ord('y'), ord('Y')):
                self.show_yolo = not self.show_yolo
                self._render()

        cv2.destroyAllWindows()
        self.cap.release()

    def _print_summary(self):
        fslot = len(set(f for f, s in self.slot_annotations))
        counts = {"available": 0, "occupied": 0, "overlapping": 0}
        for st in self.slot_annotations.values():
            if st in counts:
                counts[st] += 1
        total_bbox = sum(len(v) for v in self.bbox_annotations.values())
        print(f"\n=== Summary ===")
        print(f"  Slot frames annotated: {fslot}/{len(self.frame_numbers)}")
        for st, cnt in counts.items():
            print(f"    {st}: {cnt}")
        print(f"  Outside bbox frames: {len(self.bbox_annotations)}")
        print(f"  Total outside bboxes: {total_bbox}")
        print(f"  Slot output:  {self.slot_out}")
        print(f"  BBox output:  {self.bbox_out}")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent

    video_path = resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/"))
    frame_list_path = resolve_path("annotations", args.frame_list, DEFAULT_FRAME_LIST)
    slot_out = project_root / args.slot_out
    bbox_out = project_root / args.bbox_out

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(frame_list_path).exists():
        raise FileNotFoundError(f"Frame list not found: {frame_list_path}")

    with open(frame_list_path) as f:
        frame_numbers = sorted(int(l.strip()) for l in f if l.strip())

    print(f"[Info] Video: {video_path}")
    print(f"[Info] Frames: {len(frame_numbers)}")

    annotator = SlotBBoxAnnotator(
        video_path=video_path,
        frame_numbers=frame_numbers,
        slot_out=str(slot_out),
        bbox_out=str(bbox_out),
        continue_mode=args.continue_mode,
    )
    annotator.run()


if __name__ == "__main__":
    main()
