"""
Script phan tich 40 frame overlapping trong GT:
- Xem pattern: slot nao, detection nhu the nao, centroid o dau
- Giai thich tai sao GT label "overlapping"
"""

import csv
import pickle
from pathlib import Path
from collections import defaultdict
import sys

# Config - use absolute paths
PROJECT_ROOT = Path(r"c:\Users\maous\Downloads\server_dasboard-master\server_dasboard-master")
GT_FILE = PROJECT_ROOT / "annotations" / "CAM_PARKING_slot_state_gt.csv"
SLOT_AREA_FILE = PROJECT_ROOT / "annotations" / "CAM_PARKING_slot_area.json"
CACHE_FILE = PROJECT_ROOT / "eval_results" / "detections_cache.pkl"

# Load GT
slot_state_gt = {}
with open(GT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (int(row["frame"]), int(row["slot"]))
        slot_state_gt[key] = row["state"]

# Find all overlapping frames
overlap_frames = defaultdict(list)
for (frame, slot), state in slot_state_gt.items():
    if state == "overlapping":
        overlap_frames[frame].append(slot)

print(f"=== Tong so frame co overlapping: {len(overlap_frames)} ===")
print(f"=== Tong so slot overlapping: {sum(len(v) for v in overlap_frames.values())} ===\n")

# Load detections cache
print("Loading detections cache...")
with open(CACHE_FILE, "rb") as f:
    all_detections = pickle.load(f)
print(f"Loaded {len(all_detections)} frames from cache\n")

# Load slot areas
import json
with open(SLOT_AREA_FILE, "r") as f:
    slot_areas_data = json.load(f)

areas = slot_areas_data["outer"]
inner_areas = slot_areas_data["inner"]
num_slots = len(areas)

import cv2
import numpy as np

def point_in_polygon(pts, poly):
    """Check if a list of (x,y) points are inside a polygon."""
    results = []
    for pt in pts:
        r = cv2.pointPolygonTest(np.array(poly, np.int32), pt, False)
        results.append(r >= 0)
    return results

def bbox_intersection_area(bbox, poly):
    """Tinh dien tich bbox giao voi polygon."""
    x1, y1, x2, y2 = bbox
    bbox_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    try:
        intersections = cv2.intersectConvexConvex(bbox_poly.astype(np.float32), np.array(poly, np.float32))
        if intersections is None:
            return 0.0
        _, polygon_intersection, area = intersections
        return area if area > 0 else 0.0
    except:
        return 0.0

# Phan tich chi tiet tung overlapping frame
print("=" * 80)
print("PHAN TICH CHI TIET 40 FRAME OVERLAPPING")
print("=" * 80)

# Group by slot
by_slot = defaultdict(list)
for frame, slots in overlap_frames.items():
    for s in slots:
        by_slot[s].append(frame)

print(f"\nOverlapping theo slot: {dict(by_slot)}\n")

for frame, slots in sorted(overlap_frames.items()):
    for s in slots:
        gt_state = slot_state_gt.get((frame, s), "?")
        dets = all_detections.get(frame, [])

        # Find all slots with detections
        slot_det_state = {}
        for det in dets:
            cx, cy = det["cx"], det["cy"]
            for slot_idx in range(num_slots):
                in_outer = cv2.pointPolygonTest(
                    np.array(areas[slot_idx], np.int32), (cx, cy), False) >= 0
                if not in_outer:
                    continue
                in_inner = cv2.pointPolygonTest(
                    np.array(inner_areas[slot_idx], np.int32), (cx, cy), False) >= 0
                state = "overlapping" if not in_inner else "occupied"
                if slot_idx not in slot_det_state or state == "occupied":
                    slot_det_state[slot_idx] = state

        # Lay bbox cua detection gan nhat voi slot s
        dets_in_slot = []
        for det in dets:
            in_outer = cv2.pointPolygonTest(
                np.array(areas[s], np.int32), (det["cx"], det["cy"]), False) >= 0
            if in_outer:
                dets_in_slot.append(det)

        # Trang thai slot lan can
        neighbors = []
        if s > 0:
            neighbors.append((s-1, slot_state_gt.get((frame, s-1), "?")))
        if s < num_slots - 1:
            neighbors.append((s+1, slot_state_gt.get((frame, s+1), "?")))

        # Tinh bbox area intersection voi neighbor slots
        neighbor_intersections = {}
        for det in dets_in_slot:
            bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
            for n_slot, n_state in neighbors:
                if n_state == "occupied":
                    ia = bbox_intersection_area(bbox, areas[n_slot])
                    if ia > 0:
                        neighbor_intersections[n_slot] = ia

        print(f"Frame {frame}: slot {s} (GT={gt_state})")
        print(f"  Neighbors: {neighbors}")
        print(f"  Detections in slot {s}: {len(dets_in_slot)}")
        for i, det in enumerate(dets_in_slot):
            bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            in_inner = cv2.pointPolygonTest(
                np.array(inner_areas[s], np.int32), (det["cx"], det["cy"]), False) >= 0

            # Intersection with neighbors
            neigh_info = ""
            for n_slot, n_state in neighbors:
                if n_state == "occupied":
                    ia = bbox_intersection_area(bbox, areas[n_slot])
                    if ia > 0:
                        neigh_info += f" | intersects slot {n_slot}: {ia:.0f}px²"
                    else:
                        neigh_info += f" | NO intersect slot {n_slot}"

            print(f"    Det {i}: cx={det['cx']:.1f}, cy={det['cy']:.1f}, "
                  f"bbox={w:.0f}x{h:.0f}, in_inner={in_inner}{neigh_info}")

        if not dets_in_slot:
            print(f"    NO detections in slot {s}")
        print()
