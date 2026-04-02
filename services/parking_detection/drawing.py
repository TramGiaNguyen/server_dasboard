"""
drawing.py — Parking Detection Drawing Sub-module
Uses Supervision library for annotating vehicles + manual cv2 for zone/line/tracking drawings.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    import supervision as sv
    _SV_AVAILABLE = True
except ImportError:
    _SV_AVAILABLE = False
    print("[DRAWING] Warning: supervision not installed, falling back to manual cv2 drawing")


# ---------------------------------------------------------------------------
# Color palette — one color per line number (cycled if > 8 vehicles)
# Each color is BGR for direct cv2 usage and also wrapped as sv.Color when needed
# ---------------------------------------------------------------------------

LINE_PALETTE_BGR = [
    (0, 255, 255),    # Line 1 — Yellow
    (255, 255, 0),    # Line 2 — Cyan
    (255, 0, 255),    # Line 3 — Magenta
    (0, 255, 0),      # Line 4 — Green
    (0, 165, 255),    # Line 5 — Orange
    (128, 0, 255),    # Line 6 — Purple
    (255, 128, 0),    # Line 7 — Sky blue
    (0, 128, 255),    # Line 8 — Light orange
]


def line_color(line_num: int) -> Tuple[int, int, int]:
    """Return BGR color for a given line number (1-indexed, cycled)."""
    return LINE_PALETTE_BGR[(line_num - 1) % len(LINE_PALETTE_BGR)]


# ---------------------------------------------------------------------------
# Supervision annotators — lazily initialised to avoid venv dependency issues
# ---------------------------------------------------------------------------

_box_annotator: Optional[Any] = None
_label_annotator: Optional[Any] = None


def _get_box_annotator():
    global _box_annotator
    if _box_annotator is None and _SV_AVAILABLE:
        _box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    return _box_annotator


def _get_label_annotator():
    global _label_annotator
    if _label_annotator is None and _SV_AVAILABLE:
        _label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=3,
            color_lookup=sv.ColorLookup.TRACK,
        )
    return _label_annotator


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_vehicles_sv(frame: np.ndarray, detections_sv: Any, labels: List[str]) -> np.ndarray:
    """Draw vehicle bboxes + labels using Supervision annotators."""
    ba = _get_box_annotator()
    la = _get_label_annotator()
    if ba is not None and la is not None and len(detections_sv) > 0:
        frame = ba.annotate(frame, detections_sv)
        frame = la.annotate(frame, detections_sv, labels=labels)
    return frame


def draw_slot_overlay_cached(
    frame: np.ndarray,
    parking_space_status: List[int],
    area_arrays: List[np.ndarray],
    inner_zone_arrays: List[np.ndarray],
    area_anchor_pts: List[Tuple[int, int]],
    slot_layer: np.ndarray,
    slot_layer_mask: np.ndarray,
    last_slot_status: Optional[List[int]],
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Draw parking slot polygons with caching — only rebuilds when occupancy changes.
    Returns updated (slot_layer, slot_layer_mask, last_slot_status).
    """
    if parking_space_status != last_slot_status:
        slot_layer.fill(0)
        line_thickness = max(1, int(2 * scale_factor))
        inner_thickness = max(1, int(1 * scale_factor))
        font_scale = 0.5 * scale_factor
        text_thickness = max(1, int(1 * scale_factor))
        
        for i, status in enumerate(parking_space_status):
            color = (0, 0, 255) if status > 0 else (0, 255, 0)
            inner_color = (100, 100, 255) if status > 0 else (100, 255, 100)
            cv2.polylines(slot_layer, [area_arrays[i]], True, color, line_thickness)
            cv2.polylines(slot_layer, [inner_zone_arrays[i]], True, inner_color, inner_thickness)
            cv2.putText(slot_layer, str(i + 1), area_anchor_pts[i],
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)
        slot_layer_mask[:] = np.any(slot_layer > 0, axis=2)
        last_slot_status = list(parking_space_status)
    frame[slot_layer_mask] = slot_layer[slot_layer_mask]
    return slot_layer, slot_layer_mask, last_slot_status


def draw_entry_zone(frame: np.ndarray, entry_zones_scaled: List[List[Tuple[int, int]]], has_pending: bool, scale_factor: float = 1.0) -> None:
    """Draw entry zone polygon with optional semi-transparent fill."""
    line_thickness = max(1, int(2 * scale_factor))
    font_scale = 0.6 * scale_factor
    text_thickness = max(1, int(2 * scale_factor))
    offset_x = max(5, int(5 * scale_factor))
    offset_y = max(20, int(20 * scale_factor))
    
    for zone in entry_zones_scaled:
        pts = np.array(zone, np.int32).reshape((-1, 1, 2))
        if has_pending:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (255, 255, 0))
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        cv2.polylines(frame, [pts], True, (255, 255, 0), line_thickness)
        cv2.putText(frame, "ENTRY ZONE", (zone[0][0] + offset_x, zone[0][1] + offset_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), text_thickness)


def draw_entry_line(frame: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> None:
    """Draw the red detection entry line."""
    cv2.line(frame, start, end, (0, 0, 255), 3)


def draw_entry_line_zone(frame: np.ndarray, zone_scaled: List[Tuple[int, int]], scale_factor: float = 1.0) -> None:
    """Draw the red trigger zone polygon (replaces thin line) + ABCD bbox-corner labels."""
    if not zone_scaled:
        return
    
    line_thickness = max(1, int(2 * scale_factor))
    font_scale_trigger = 0.55 * scale_factor
    text_thickness_trigger = max(1, int(2 * scale_factor))
    offset_x = max(5, int(5 * scale_factor))
    offset_y = max(8, int(8 * scale_factor))
    
    pts = np.array(zone_scaled, np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 0, 255))
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.polylines(frame, [pts], True, (0, 0, 255), line_thickness)
    # Label near first vertex
    x0, y0 = zone_scaled[0]
    cv2.putText(frame, "ENTRY TRIGGER", (x0 + offset_x, y0 - offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_trigger, (0, 0, 255), text_thickness_trigger)

    # ABCD labels based on bbox corners of the polygon.
    xs = [p[0] for p in zone_scaled]
    ys = [p[1] for p in zone_scaled]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Pad labels inwards a bit so they stay readable.
    padX = max(6, int(0.03 * (x_max - x_min + 1)))
    padY = max(6, int(0.03 * (y_max - y_min + 1)))

    h, w = frame.shape[:2]

    def _clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    A = (_clamp(x_min + padX, 0, w - 1), _clamp(y_min + padY, 0, h - 1))  # top-left
    B = (_clamp(x_max - padX, 0, w - 1), _clamp(y_min + padY, 0, h - 1))  # top-right
    C = (_clamp(x_max - padX, 0, w - 1), _clamp(y_max - padY, 0, h - 1))  # bottom-right
    D = (_clamp(x_min + padX, 0, w - 1), _clamp(y_max - padY, 0, h - 1))  # bottom-left

    # Black outline then white fill for readability on red fill.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7 * scale_factor
    thickness = max(1, int(2 * scale_factor))
    outline_thickness = max(2, int(4 * scale_factor))

    def _put_label(pt: Tuple[int, int], ch: str) -> None:
        cv2.putText(frame, ch, pt, font, scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
        cv2.putText(frame, ch, pt, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    _put_label(A, "A")
    _put_label(B, "B")
    _put_label(C, "C")
    _put_label(D, "D")


def draw_pending_lines(
    frame: np.ndarray,
    pending_vehicles: Dict[int, Dict],
    track_id_to_vehicle: Dict[int, Dict],
    entry_line_start: Tuple[int, int],
) -> None:
    """
    Legacy: entry-zone pending lines removed (FIFO trigger matching). Kept as no-op for compatibility.
    """
    return


def draw_matched_plates(
    frame: np.ndarray,
    matched_vehicles: Dict[int, Dict],
    track_id_to_vehicle: Dict[int, Dict],
    scale_factor: float = 1.0,
) -> None:
    """Draw plate label above each matched vehicle's current bbox."""
    border_thickness = max(2, int(3 * scale_factor))
    
    for line_num, match_info in matched_vehicles.items():
        plate_text = match_info.get('plate') or ''
        frames_missing = match_info.get('frames_missing', 0)
        found_current_pos = match_info.get('last_seen_frame') == match_info.get('_current_frame')
        is_parked = match_info.get('parked', False)
        is_parked_outside = match_info.get('parked_outside', False)

        keep_recent = is_parked_outside or frames_missing <= 20
        if not (found_current_pos or keep_recent or is_parked or is_parked_outside):
            continue

        x1, y1, x2, y2 = match_info.get('bbox', (0, 0, 0, 0))

        # Draw cyan border for matched vehicles
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), border_thickness)

        if plate_text:
            font_scale = 0.7 * scale_factor
            text_thickness = max(1, int(2 * scale_factor))
            padding = max(5, int(5 * scale_factor))
            offset_y = max(10, int(10 * scale_factor))
            
            (tw, th), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            center_x = (x1 + x2) // 2
            label_x1 = center_x - tw // 2 - padding
            label_y1 = max(0, y1 - th - offset_y - padding)
            cv2.rectangle(frame, (label_x1, label_y1), (label_x1 + tw + padding * 2, y1 - padding), (0, 200, 255), -1)
            cv2.putText(frame, plate_text, (label_x1 + padding, y1 - offset_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)
        else:
            label = "..."
            font_scale = 1.0 * scale_factor
            text_thickness = max(2, int(3 * scale_factor))
            padding = max(5, int(5 * scale_factor))
            offset_y = max(10, int(10 * scale_factor))
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            center_x = (x1 + x2) // 2
            label_x1 = center_x - tw // 2 - padding
            label_y1 = max(0, y1 - th - offset_y - padding)
            cv2.rectangle(frame, (label_x1, label_y1), (label_x1 + tw + padding * 2, y1 - padding), (100, 100, 100), -1)
            cv2.putText(frame, label, (label_x1 + padding, y1 - offset_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)


def draw_parked_vehicles(
    frame: np.ndarray,
    tracked_vehicles: list,
    areas: list,
    inner_zones: list,
    class_list: list,
    scale_factor: float = 1.0,
    min_confidence: float = 0.20,
) -> dict:
    """
    Draw stable bboxes for parked vehicles using tracked positions.
    Returns dict with slot occupancy info.
    
    Args:
        frame: Frame to draw on
        tracked_vehicles: List of tracked vehicle dicts with bbox, track_id, class_name
        areas: Parking slot polygons
        inner_zones: Inner zone polygons for proper parking check
        class_list: COCO class names
        scale_factor: Scale factor for annotations
        min_confidence: Minimum confidence threshold (not used for tracked vehicles)
    
    Returns:
        dict: {'slot_occupancy': {slot_num: [vehicle_info]}, 'overlapping': [...]}
    """
    bbox_thickness = max(1, int(2 * scale_factor))
    font_scale = 0.45 * scale_factor
    text_thickness = max(1, int(1 * scale_factor))
    text_offset_y = max(10, int(10 * scale_factor))
    
    slot_occupancy = {}
    overlapping_vehicles = []
    
    for vehicle in tracked_vehicles:
        x1, y1, x2, y2 = vehicle['bbox']
        cx, cy = vehicle['cx'], vehicle['cy']
        class_name = vehicle.get('class_name', 'car')
        track_id = vehicle.get('track_id', 0)
        
        # Skip if not a motor vehicle
        if class_name not in ('car', 'bus', 'truck'):
            continue
        
        # Check which slot this vehicle is in
        slot_found = False
        for i, area in enumerate(areas):
            if cv2.pointPolygonTest(np.array(area, np.int32), (float(cx), float(cy)), False) >= 0:
                slot_num = i + 1
                slot_found = True
                
                # Check if properly parked (inside inner zone)
                is_properly_parked = cv2.pointPolygonTest(
                    np.array(inner_zones[i], np.int32), (float(cx), float(cy)), False
                ) >= 0
                
                # Draw stable bbox with track_id
                if is_properly_parked:
                    color = (0, 255, 0)  # Green for properly parked
                    label = f"Slot {slot_num} | ID:{track_id}"
                else:
                    color = (0, 165, 255)  # Orange for overlapping
                    label = f"Overlap {slot_num} | ID:{track_id}"
                    overlapping_vehicles.append({
                        'slot': slot_num,
                        'track_id': track_id,
                        'bbox': [x1, y1, x2, y2],
                    })
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thickness)
                cv2.putText(frame, label, (x1, y1 - text_offset_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)
                
                # Store occupancy info
                if slot_num not in slot_occupancy:
                    slot_occupancy[slot_num] = []
                slot_occupancy[slot_num].append({
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'class_name': class_name,
                    'properly_parked': is_properly_parked,
                })
                break
        
        # Draw outside vehicles (not in any slot)
        if not slot_found:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), bbox_thickness)
            cv2.putText(frame, f"Outside | ID:{track_id}", (x1, y1 - text_offset_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness)
    
    return {
        'slot_occupancy': slot_occupancy,
        'overlapping': overlapping_vehicles,
    }


def build_sv_detections(vehicles: List[Dict], id_offset: int = 0) -> Any:
    """
    Convert a list of vehicle dicts to a supervision Detections object.
    vehicles: list of {'bbox': [x1,y1,x2,y2], 'track_id': int, 'conf': float}
    Returns sv.Detections or None if supervision unavailable.
    """
    if not _SV_AVAILABLE or not vehicles:
        return None
    xyxy = np.array([v['bbox'] for v in vehicles], dtype=np.float32)
    track_ids = np.array([v.get('track_id', i) for i, v in enumerate(vehicles)], dtype=int)
    confidences = np.array([v.get('conf', 1.0) for v in vehicles], dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        tracker_id=track_ids,
        confidence=confidences,
    )
