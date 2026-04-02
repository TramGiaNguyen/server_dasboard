"""
detection.py — Parking Detection Sub-module
Zone configuration, slot polygon definitions, and per-frame vehicle detection processing.
All coordinates are stored at the 1020×500 baseline and scaled to native resolution at runtime.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Zone / area definitions  (baseline: 1020 × 500 px)
# ---------------------------------------------------------------------------

PARKING_ZONES_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "parking_zones_config.txt",
)


def load_class_list(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        return f.read().split("\n")


def _load_zones_from_config() -> Dict[str, Any]:
    """
    Load PARKING_SLOTS / ENTRY_ZONES / ENTRY_LINE_ZONES from parking_zones_config.txt if present.
    The file is treated as a small Python config file.
    """
    cfg: Dict[str, Any] = {"PARKING_SLOTS": [], "ENTRY_ZONES": [], "ENTRY_LINE_ZONES": []}
    if not os.path.exists(PARKING_ZONES_CONFIG_PATH):
        return cfg

    try:
        # Execute in an isolated namespace
        namespace: Dict[str, Any] = {}
        with open(PARKING_ZONES_CONFIG_PATH, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, PARKING_ZONES_CONFIG_PATH, "exec"), namespace, namespace)
        slots = namespace.get("PARKING_SLOTS")
        zones = namespace.get("ENTRY_ZONES")
        line_zones = namespace.get("ENTRY_LINE_ZONES")
        if isinstance(slots, list):
            cfg["PARKING_SLOTS"] = slots
        if isinstance(zones, list):
            cfg["ENTRY_ZONES"] = zones
        if isinstance(line_zones, list):
            cfg["ENTRY_LINE_ZONES"] = line_zones
    except Exception as exc:  # pragma: no cover - defensive logging only
        print(f"[PARKING CONFIG] Failed to load parking_zones_config.txt: {exc}")
    return cfg


def define_parking_areas() -> List[List[Tuple[int, int]]]:
    """
    Return polygon vertices for each parking slot (baseline 1020×500).

    Priority:
    1. Use PARKING_SLOTS from parking_zones_config.txt if available.
    2. Fallback to built-in default layout below.
    """
    cfg = _load_zones_from_config()
    slots = cfg.get("PARKING_SLOTS") or []
    if slots:
        return slots

    # Built-in default layout (1020×500 baseline)
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
        [(839,134),(887,139),(954,205),(891,199)],  # Slot 19
    ]


def define_entry_zones() -> List[List[Tuple[int, int]]]:
    """
    Return polygon vertices for the entry zone / entry block (baseline 1020×500).

    Priority:
    1. Use ENTRY_ZONES from parking_zones_config.txt if available.
    2. Fallback to built-in default polygon below.
    """
    cfg = _load_zones_from_config()
    zones = cfg.get("ENTRY_ZONES") or []
    if zones:
        return zones

    # Built-in default polygon (legacy behaviour)
    return [
        [(139,47),(159,121),(204,127),(109,352),(173,429),(249,434),(251,338),(611,351),(627,254),(523,45)]
    ]


def define_entry_line_zones() -> List[List[Tuple[int, int]]]:
    """
    Return polygon vertices for the "red line" trigger zone (baseline 1020×500).

    Priority:
    1. Use ENTRY_LINE_ZONES from parking_zones_config.txt if available.
    2. Fallback to a small rectangle around the legacy red segment.
    """
    cfg = _load_zones_from_config()
    zones = cfg.get("ENTRY_LINE_ZONES") or []
    if zones:
        return zones

    # Legacy line segment was around: (204,44) → (198,71).
    # Provide a padded rectangle as a robust default trigger zone.
    x1, y1 = 204, 44
    x2, y2 = 198, 71
    pad = 18
    minx, maxx = min(x1, x2) - pad, max(x1, x2) + pad
    miny, maxy = min(y1, y2) - pad, max(y1, y2) + pad
    return [[(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]]


def scale_areas(raw_areas: List[List[Tuple[int, int]]], sx: float, sy: float) -> List[List[Tuple[int, int]]]:
    """Scale a list of polygon areas from baseline to native resolution."""
    return [[(int(x * sx), int(y * sy)) for x, y in area] for area in raw_areas]


# ---------------------------------------------------------------------------
# Slot configuration helpers
# ---------------------------------------------------------------------------

# Far slots that need smaller inner zones (0-indexed)
_FAR_SLOTS = set(range(0, 14)) | {17, 18}
_FAR_SLOT_SHRINK = 0.35
_DEFAULT_SHRINK = 0.20


def get_detection_params(slot_index: int) -> Dict[str, Any]:
    if slot_index == 18:  # Slot 19 — far, lower threshold
        return {'min_confidence': 0.35, 'min_area_size': 1500, 'min_dimension': 30, 'allow_partial': False}
    return {'min_confidence': 0.50, 'min_area_size': 1500, 'min_dimension': 30, 'allow_partial': False}


def create_inner_zones(areas: List[List[Tuple[int, int]]], shrink_percentage: float = 0.20) -> List[List[Tuple[int, int]]]:
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


def rect_poly_overlap_ratio(x1, y1, x2, y2, poly_pts, frame_shape):
    h, w = frame_shape[:2]
    box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(box_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.array(poly_pts, np.int32)], 255)
    inter = cv2.bitwise_and(box_mask, poly_mask)
    slot_area = max(1, cv2.contourArea(np.array(poly_pts, np.int32)))
    return float(np.count_nonzero(inter)) / float(slot_area)


def is_partial_vehicle_detection(class_name: str) -> bool:
    partial_indicators = ['car', 'bus', 'truck', 'wheel', 'tire', 'license plate', 'headlight', 'taillight']
    return any(ind in class_name.lower() for ind in partial_indicators)


# ---------------------------------------------------------------------------
# Per-frame detection processing
# ---------------------------------------------------------------------------

def process_detection(
    row, areas, inner_zones, frame, class_list, parking_space_status,
    outside_vehicles, overlapping_vehicles,
    default_min_confidence=0.5, default_min_area_size=1500, default_min_dimension=30,
    entry_line_center=None, entry_line_exclude_radius=0, scale_factor=1.0,
    draw_bbox=False  # NEW: disable bbox drawing (will use tracked vehicles instead)
):
    x1, y1, x2, y2, confidence, class_idx = map(float, row)
    x1, y1, x2, y2, class_idx = int(x1), int(y1), int(x2), int(y2), int(class_idx)
    class_name = class_list[class_idx]

    width = x2 - x1
    height = y2 - y1
    area_size = width * height
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Ghost bbox fix: skip detections centered on the entry line
    if entry_line_center is not None and entry_line_exclude_radius > 0:
        dx = cx - entry_line_center[0]
        dy = cy - entry_line_center[1]
        if dx * dx + dy * dy <= entry_line_exclude_radius * entry_line_exclude_radius:
            return

    if not (class_name in ['car', 'bus', 'truck'] or is_partial_vehicle_detection(class_name)):
        return

    detected_in_slot = False
    
    # Scale drawing parameters (only used if draw_bbox=True)
    bbox_thickness = max(1, int(2 * scale_factor))
    font_scale = 0.45 * scale_factor
    text_thickness = max(1, int(1 * scale_factor))
    text_offset_y = max(10, int(10 * scale_factor))

    for i, area in enumerate(areas):
        if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
            detected_in_slot = True
            params = get_detection_params(i)

            if confidence < params['min_confidence']:
                return

            if params['allow_partial']:
                if area_size < params['min_area_size'] or width < params['min_dimension'] or height < params['min_dimension']:
                    if area_size < 200 or width < 8 or height < 8:
                        print(f"[SLOT {i+1}] Rejected: too small ({area_size:.0f}px, {width}x{height})")
                        return
                    print(f"[SLOT {i+1}] Accepted partial: {class_name} conf={confidence:.2f} size={area_size:.0f}px")
            else:
                if area_size < params['min_area_size'] or width < params['min_dimension'] or height < params['min_dimension']:
                    return

            parking_space_status[i] += 1
            is_properly_parked = cv2.pointPolygonTest(np.array(inner_zones[i], np.int32), (cx, cy), False) >= 0
            detection_label = class_name
            if params['allow_partial'] and area_size < 1200:
                detection_label = f"{class_name}*"

            # Only draw bbox if explicitly requested (backward compatibility)
            if draw_bbox:
                if is_properly_parked:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bbox_thickness)
                    cv2.putText(frame, f"OK: {detection_label} {confidence:.2f}", (x1, y1 - text_offset_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), bbox_thickness)
                    cv2.putText(frame, f"Overlap: {detection_label} {confidence:.2f}", (x1, y1 - text_offset_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), max(1, int(2 * scale_factor)))
            
            if not is_properly_parked:
                overlapping_vehicles.append({
                    'class': class_name,
                    'area': i,
                    'confidence': float(confidence),
                    'partial': params['allow_partial'] and area_size < 1200,
                    'bbox': [x1, y1, x2, y2],
                })
            return

    if not detected_in_slot and class_name in ['car', 'bus', 'truck']:
        if confidence >= default_min_confidence and area_size >= default_min_area_size:
            # Only draw if requested
            if draw_bbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), bbox_thickness)
                cv2.putText(frame, f"Outside: {class_name} {confidence:.2f}", (x1, y1 - text_offset_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness)
            outside_vehicles.append(class_name)

    return {
        'detected': detected_in_slot,
        'slot': None,
        'bbox': [x1, y1, x2, y2],
        'class_name': class_name,
        'confidence': confidence,
    } if detected_in_slot and class_name in ['car', 'bus', 'truck'] else None
