#!/usr/bin/env python3
"""
annotate_vehicles.py

Cong cu annotation bbox phuong tien tren cac frame da chon.
Chi can chon 100 frame, ve bbox, luu ra CSV.

Luu y:
- Annotation o do phan giai goc 1910x1080 (khong scale)
- Chi annotation 100 frame tu file frame_numbers.txt

Usage:
    python annotate_vehicles.py                              # che do tua chua tham
    python annotate_vehicles.py --continue                  # tiep tuc tu lan truoc
    python annotate_vehicles.py --video static/video/CAM_PARKING.mp4
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO = "static/video/CAM_PARKING.mp4"
DEFAULT_FRAME_LIST = "annotations/frame_numbers.txt"
DEFAULT_OUTPUT = "annotations/CAM_PARKING_gt.csv"
DEFAULT_WINDOW = "Annotate Vehicles - ESC:quit  Space:next  B:back  D:delete last  S:skip"

# Coco classes
VEHICLE_CLASS_NAMES = ["car", "bus", "truck"]
VEHICLE_CLASS_DISPLAY = {0: "CAR", 1: "BUS", 2: "TRUCK"}

# Mau cho tung class
CLASS_COLORS = {
    0: (0, 255, 0),    # car - xanh la
    1: (0, 165, 255),  # bus - cam
    2: (0, 0, 255),    # truck - do
}


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate vehicle bounding boxes")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to video")
    parser.add_argument("--frame-list", default=DEFAULT_FRAME_LIST, help="Frame numbers file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV")
    parser.add_argument("--continue", dest="continue_mode", action="store_true",
                        help="Continue from existing CSV")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip frames that already have annotations in CSV")
    return parser.parse_args()


def resolve_path(base, path_arg, default):
    project_root = Path(__file__).resolve().parent.parent.parent
    candidates = [path_arg, project_root / path_arg]
    for c in candidates:
        if Path(c).exists():
            return str(Path(c).resolve())
    return str(project_root / default)


class BBoxAnnotator:
    def __init__(self, video_path, frame_numbers, output_path,
                 continue_mode=False, skip_existing=False):
        self.video_path = video_path
        self.frame_numbers = frame_numbers
        self.output_path = output_path
        self.continue_mode = continue_mode
        self.skip_existing = skip_existing

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Khong mo duoc video: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.existing = {}
        if self.continue_mode and Path(self.output_path).exists():
            self._load_existing()

        self.annotated = set(self.existing.keys())
        if skip_existing:
            self.frames_to_annotate = [f for f in self.frame_numbers if f not in self.annotated]
        else:
            self.frames_to_annotate = self.frame_numbers

        self.frame_idx = 0
        self.current_frame = self.frames_to_annotate[self.frame_idx] if self.frames_to_annotate else None

        self.drawing = False
        self.start_point = None
        self.temp_box = None
        self.selecting_class = 0

        self.selected_box_idx = -1

        cv2.namedWindow(DEFAULT_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEFAULT_WINDOW, self.width, self.height)
        cv2.setMouseCallback(DEFAULT_WINDOW, self._on_mouse)

        self._load_frame()
        self._render()

    def _load_existing(self):
        with open(self.output_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row["frame"])
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                cls = row["class_name"].strip()
                if cls == "car":
                    ci = 0
                elif cls == "bus":
                    ci = 1
                elif cls == "truck":
                    ci = 2
                else:
                    ci = 0
                self.existing.setdefault(frame, []).append((x1, y1, x2, y2, ci))
        print(f"[Info] Da load {len(self.existing)} frames da co annotation")

    def _load_frame(self):
        if self.current_frame is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, self.frame = self.cap.read()
        if not ret:
            print(f"[WARN] Khong doc duoc frame {self.current_frame}")
            self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.current_boxes = list(self.existing.get(self.current_frame, []))
        self.temp_box = None
        self.start_point = None
        self.drawing = False
        self.selected_box_idx = -1

    def _draw_box(self, img, x1, y1, x2, y2, cls_idx, thickness=2):
        color = CLASS_COLORS.get(cls_idx, (0, 255, 0))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        label = VEHICLE_CLASS_DISPLAY.get(cls_idx, "?")
        lx, ly = int(x1), max(0, int(y1) - 8)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (lx, ly - lh - 2), (lx + lw + 4, ly + 2), color, -1)
        cv2.putText(img, label, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _point_in_box(self, px, py, box):
        x1, y1, x2, y2, _ = box
        return x1 <= px <= x2 and y1 <= py <= y2

    def _render(self):
        canvas = self.frame.copy()

        for i, box in enumerate(self.current_boxes):
            is_selected = (i == self.selected_box_idx)
            thickness = 3 if is_selected else 2
            color_adj = (255, 255, 0) if is_selected else None
            self._draw_box(canvas, *box, thickness=thickness)
            if is_selected:
                x1, y1, x2, y2, cls_idx = box
                color = CLASS_COLORS.get(cls_idx, (0, 255, 0))
                cv2.circle(canvas, (int((x1+x2)/2), int((y1+y2)/2)), 5, (255, 255, 0), -1)

        if self.temp_box and self.start_point:
            x1, y1, x2, y2 = self.temp_box
            self._draw_box(canvas, x1, y1, x2, y2, self.selecting_class, 2)

        h = 25
        cv2.putText(canvas, f"Frame {self.current_frame}  [{self.frame_idx+1}/{len(self.frames_to_annotate)}]  BBoxs: {len(self.current_boxes)}",
                     (5, h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        h += 30
        cls_name = VEHICLE_CLASS_DISPLAY[self.selecting_class]
        cv2.putText(canvas, f"Class: {cls_name}  [1:car  2:bus  3:truck]",
                     (5, h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        h += 30
        help_lines = [
            "[Space] next(+copy bbox)  [B] prev  [S] skip  [ESC] quit",
            "[Click] select bbox  [Del] delete selected  [D] delete last",
        ]
        for i, line in enumerate(help_lines):
            cv2.putText(canvas, line, (5, h + i*25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if self.selected_box_idx >= 0:
            cv2.putText(canvas, f"Selected: {self.selected_box_idx}  [Del] to delete",
                         (5, h + len(help_lines)*25 + 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if self.current_frame in self.annotated:
            cv2.putText(canvas, "ALREADY ANNOTATED",
                         (5, h + (len(help_lines)+1)*25 + 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(DEFAULT_WINDOW, canvas)

    def _on_mouse(self, event, x, y, flags, param):
        if self.current_frame is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, box in enumerate(self.current_boxes):
                if self._point_in_box(x, y, box):
                    self.selected_box_idx = i
                    self._render()
                    return
            self.drawing = True
            self.start_point = (x, y)
            self.temp_box = (x, y, x, y)
            self.selected_box_idx = -1
            self._render()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                sx, sy = self.start_point
                self.temp_box = (sx, sy, x, y)
                self._render()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                sx, sy = self.start_point
                if abs(x - sx) > 5 and abs(y - sy) > 5:
                    box = (min(sx, x), min(sy, y), max(sx, x), max(sy, y), self.selecting_class)
                    self.current_boxes.append(box)
                self.drawing = False
                self.start_point = None
                self.temp_box = None
                self._render()

    def _save(self):
        self.existing[self.current_frame] = list(self.current_boxes)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        all_rows = []
        for frame in sorted(self.existing.keys()):
            for box in self.existing[frame]:
                x1, y1, x2, y2, cls_idx = box
                all_rows.append({
                    "frame": frame,
                    "class_name": VEHICLE_CLASS_NAMES[cls_idx],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "class_name", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(all_rows)
        self.annotated.add(self.current_frame)

    def _next_frame(self):
        if self.frame_idx < len(self.frames_to_annotate) - 1:
            self._save()
            prev_boxes = list(self.current_boxes)
            self.frame_idx += 1
            self.current_frame = self.frames_to_annotate[self.frame_idx]
            self._load_frame()
            if self.current_frame not in self.existing:
                self.current_boxes = list(prev_boxes)
            self._render()

    def _prev_frame(self):
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self.current_frame = self.frames_to_annotate[self.frame_idx]
            self._load_frame()
            self._render()

    def _delete_selected(self):
        if 0 <= self.selected_box_idx < len(self.current_boxes):
            self.current_boxes.pop(self.selected_box_idx)
            self.selected_box_idx = -1
            self._render()

    def _delete_last(self):
        if self.current_boxes:
            self.current_boxes.pop()
            if self.selected_box_idx >= len(self.current_boxes):
                self.selected_box_idx = -1
            self._render()

    def _set_class(self, cls_idx):
        self.selecting_class = cls_idx
        self._render()

    def _skip_frame(self):
        self._save()
        if self.frame_idx < len(self.frames_to_annotate) - 1:
            self.frame_idx += 1
            self.current_frame = self.frames_to_annotate[self.frame_idx]
            self._load_frame()
            self._render()

    def run(self):
        print("\n=== Annotation Instructions ===")
        print("  VE BBOX:    Click & drag chuot trai")
        print("  DOI CLASS:  Nhan 1=car  2=bus  3=truck")
        print("  CHON BBOX:  Click vao bbox da ve (highlight vang)")
        print("  XOA BBOX:   [Del] xoa bbox da chon  [D] xoa bbox cuoi")
        print("  FRAME TIEP: [Space] next (+ tu dong copy bbox tu frame truoc)")
        print("  FRAME LUI:  [B] prev")
        print("  BO QUA:     [S] skip")
        print("  THOAT:      [ESC] tu dong luu")
        print("================================\n")
        print(f"[Info] Annotation o {self.width}x{self.height} (khong scale)")
        print(f"[Info] {len(self.frames_to_annotate)} frame can annotation")
        print(f"[Info] Output: {self.output_path}")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                print("\n[Info] Thoat va luu...")
                self._save()
                self._print_summary()
                break
            elif key == 32:
                self._next_frame()
            elif key == ord('b') or key == ord('B'):
                self._prev_frame()
            elif key == ord('d') or key == ord('D'):
                self._delete_last()
            elif key == ord('s') or key == ord('S'):
                self._skip_frame()
            elif key == 46:  # Delete
                self._delete_selected()
            elif key == ord('1'):
                self._set_class(0)
            elif key == ord('2'):
                self._set_class(1)
            elif key == ord('3'):
                self._set_class(2)

        cv2.destroyAllWindows()
        self.cap.release()

    def _print_summary(self):
        total_frames = len(self.existing)
        total_boxes = sum(len(v) for v in self.existing.values())
        class_counts = {c: 0 for c in VEHICLE_CLASS_NAMES}
        for boxes in self.existing.values():
            for box in boxes:
                class_counts[VEHICLE_CLASS_NAMES[box[4]]] += 1
        print(f"\n=== Summary ===")
        print(f"  Frames annotated: {total_frames}/{len(self.frame_numbers)}")
        print(f"  Total bboxes:    {total_boxes}")
        for cls, cnt in class_counts.items():
            print(f"    {cls}: {cnt}")
        print(f"  Output: {self.output_path}")


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    video_path = resolve_path("static/video", args.video, DEFAULT_VIDEO.lstrip("static/"))
    frame_list_path = resolve_path("annotations", args.frame_list, DEFAULT_FRAME_LIST)
    output_path = project_root / args.output

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(frame_list_path).exists():
        raise FileNotFoundError(f"Frame list not found: {frame_list_path}")

    with open(frame_list_path, "r") as f:
        frame_numbers = sorted(int(line.strip()) for line in f if line.strip())

    print(f"[Info] Video:    {video_path}")
    print(f"[Info] Frames:    {len(frame_numbers)} frames")
    print(f"[Info] Output:    {output_path}")

    annotator = BBoxAnnotator(
        video_path=video_path,
        frame_numbers=frame_numbers,
        output_path=str(output_path),
        continue_mode=args.continue_mode,
        skip_existing=args.skip_existing,
    )
    annotator.run()


if __name__ == "__main__":
    main()
