"""
tracking.py — Parking Tracking helpers sub-module

Contains reusable tracking utilities that are shared by the parking
camera pipeline, such as IoU computation for bounding boxes.
"""

from typing import Sequence


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Compute IoU (Intersection over Union) between two [x1, y1, x2, y2] boxes.
    Returns 0.0 if there is no overlap or if areas are invalid.
    """
    if len(a) != 4 or len(b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

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
    if union <= 0.0:
        return 0.0

    return float(inter / union)


def rect_intersects_polygon(rect: Sequence[float], poly: Sequence[Sequence[int]]) -> bool:
    """
    Quick intersection check between a rectangle [x1,y1,x2,y2] and a polygon.
    Returns True if:
      - any rect corner is inside polygon, OR
      - any polygon vertex is inside rect.
    This is sufficient for our small trigger zones and avoids expensive masks.
    """
    if len(rect) != 4 or not poly:
        return False

    x1, y1, x2, y2 = map(float, rect)
    if x2 <= x1 or y2 <= y1:
        return False

    # Normalize
    rx1, ry1, rx2, ry2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    # Lazy import to avoid adding hard deps elsewhere
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    poly_np = np.array(poly, dtype=np.int32)

    # 1) rect corners inside polygon?
    corners = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]
    for cx, cy in corners:
        if cv2.pointPolygonTest(poly_np, (float(cx), float(cy)), False) >= 0:
            return True

    # 2) any polygon vertex inside rect?
    for px, py in poly:
        if rx1 <= px <= rx2 and ry1 <= py <= ry2:
            return True

    return False

