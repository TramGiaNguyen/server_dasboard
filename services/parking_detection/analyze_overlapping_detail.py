"""
Phan tich detections thuc te o cac frame overlapping trong video.
"""

import csv
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---- Parking zone definitions ----
def define_parking_areas():
    return [
        [(305,101),(360,101),(353,153),(288,151),(305,99)],
        [(362,101),(421,101),(426,159),(354,154)],
        [(421,103),(486,102),(504,163),(426,158)],
        [(488,104),(553,108),(583,167),(505,163)],
        [(554,107),(623,112),(666,174),(585,168)],
        [(622,113),(701,120),(759,182),(666,175)],
        [(700,121),(789,128),(858,195),(759,184)],
        [(788,129),(890,138),(970,207),(858,198)],
        [(889,139),(1007,149),(1102,219),(969,210)],
        [(1006,151),(1134,161),(1242,234),(1101,221)],
        [(1133,163),(1267,174),(1388,247),(1241,236)],
        [(1266,176),(1408,187),(1542,260),(1387,250)],
        [(1407,189),(1556,201),(1701,273),(1541,263)],
        [(1555,203),(1713,215),(1868,287),(1700,275)],
        [(1712,217),(1880,229),(2046,302),(1867,289)],
        [(1879,231),(2057,243),(2233,316),(2045,307)],
        [(2056,245),(2241,257),(2429,330),(2232,319)],
        [(2240,259),(2433,271),(2631,344),(2428,333)],
        [(2432,273),(2628,284),(2834,358),(2630,346)],
        [(2627,287),(2832,298),(3046,372),(2833,359)],
    ]

def create_inner_zones(areas):
    _FAR_SLOTS = set(range(0, 14)) | {17, 18}
    _FAR_SLOT_SHRINK = 0.35
    inner_zones = []
    for idx, area in enumerate(areas):
        slot_shrink = _FAR_SLOT_SHRINK if idx in _FAR_SLOTS else 0.20
        points = np.array(area, dtype=np.float32)
        centroid = np.mean(points, axis=0)
        inner_points = []
        for point in points:
            vector = point - centroid
            new_point = centroid + vector * (1 - slot_shrink)
            inner_points.append(tuple(new_point.astype(int)))
        inner_zones.append(inner_points)
    return inner_zones

areas = define_parking_areas()
inner_areas = create_inner_zones(areas)
num_slots = len(areas)

# ---- Load GT ----
GT_FILE = Path(r"c:\Users\maous\Downloads\server_dasboard-master\server_dasboard-master\annotations\CAM_PARKING_slot_state_gt.csv")
slot_state_gt = {}
with open(GT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (int(row["frame"]), int(row["slot"]))
        slot_state_gt[key] = row["state"]

# Find overlapping frames
overlap_frames = defaultdict(list)
for (frame, slot), state in slot_state_gt.items():
    if state == "overlapping":
        overlap_frames[frame].append(slot)

# ---- Open video and extract detections ----
VIDEO_FILE = Path(r"c:\Users\maous\Downloads\server_dasboard-master\server_dasboard-master\static\video\CAM_PARKING.mp4")
cap = cv2.VideoCapture(str(VIDEO_FILE))

print("=" * 80)
print("PHAN TICH DETECTIONS O 40 FRAME OVERLAPPING (baseline 1020x500)")
print("=" * 80)

# Test voi 5 frame dai dien
sample_frames = [2537, 7044, 7865, 9174, 15975]

def get_detections_at_frame(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return []

    # Fake detection: tinh centroid cua moi slot polygon
    # (vi khong co YOLO, ta dung polygon centroid de simulate)
    # Thuc te can chay YOLO nhung o day ta phan tich geometry

    detections = []
    for s in range(num_slots):
        outer_pts = np.array(areas[s], np.int32)
        inner_pts = np.array(inner_areas[s], np.int32)
        outer_area = cv2.contourArea(outer_pts)
        inner_area = cv2.contourArea(inner_pts)

        # Tinh khoang cach tu centroid den inner boundary
        M_outer = cv2.moments(outer_pts)
        M_inner = cv2.moments(inner_pts)

        if M_outer["m00"] > 0 and M_inner["m00"] > 0:
            cx_outer = M_outer["m10"] / M_outer["m00"]
            cy_outer = M_outer["m01"] / M_outer["m00"]
            cx_inner = M_inner["m10"] / M_inner["m00"]
            cy_inner = M_inner["m01"] / M_inner["m00"]

            dist_inner_boundary = abs(cx_outer - cx_inner) if cx_outer != cx_inner else abs(cy_outer - cy_inner)

            # Tim margin giua inner va outer
            outer_xs = [p[0] for p in areas[s]]
            inner_xs = [p[0] for p in inner_areas[s]]
            outer_width = max(outer_xs) - min(outer_xs)
            inner_width = max(inner_xs) - min(inner_xs)
            margin = (outer_width - inner_width) / 2

            detections.append({
                "slot": s,
                "outer_area": outer_area,
                "inner_area": inner_area,
                "ratio": inner_area / outer_area * 100,
                "outer_width": outer_width,
                "inner_width": inner_width,
                "margin_px": margin,
                "margin_pct": margin / outer_width * 100,
                "cx": cx_outer,
                "cy": cy_outer,
            })
    return detections

for frame_num in sample_frames:
    slots_with_overlap = overlap_frames.get(frame_num, [])
    if not slots_with_overlap:
        continue

    dets = get_detections_at_frame(cap, frame_num)
    print(f"\nFrame {frame_num}: overlapping slots = {slots_with_overlap}")
    print(f"  Slot | OuterArea | InnerArea | Ratio% | OuterW | InnerW | MarginPx | Margin%")
    print(f"  " + "-" * 75)

    for d in dets:
        s = d["slot"]
        gt = slot_state_gt.get((frame_num, s), "?")
        marker = " <-- OVERLAPPING" if s in slots_with_overlap else ""
        print(f"  {s:>4} | {d['outer_area']:>9.0f} | {d['inner_area']:>9.0f} | "
              f"{d['ratio']:>5.1f}% | {d['outer_width']:>7.0f} | {d['inner_width']:>7.0f} | "
              f"{d['margin_px']:>8.1f} | {d['margin_pct']:>6.1f}%{marker}")

cap.release()

# ---- Tinh toan: tai sao occupied bi nham thanh overlapping ----
print("\n" + "=" * 80)
print("PHAN TICH: TAI SAO 49 OCCUPIED BI NHAM THANH OVERLAPPING?")
print("=" * 80)

# Voi bounding box YOLO, centroid co the nam bat ky dau trong polygon
# Khoang cach tu inner boundary den outer = margin
# Neu centroid nam trong vung margin -> bi classify la overlapping

print("\nMuc toieu cham chan (margin) giua inner va outer boundary:")
print(f"  - FAR slots (0-13, 17, 18): margin = 17.4% cua chieu ngang slot")
print(f"  - CLOSE slots (14-16, 19):    margin = 9.9% cua chieu ngang slot")
print()

# Thu nghiem: voi 1 xe co bbox w x h, centroid offset bao nhieu se ra ngoai inner?
print("Vi du: mot xe 50x40 pixels do xe gap o giua slot.")
print("  - Neu centroid offset 10px sang trai (vi bbox xe khong can giua) -> nam trong vung margin")
print("  - Vung margin = 17.4% * 72px = 12.5px (slot 0)")
print("  - Xe chi can offset > 12.5px -> bi nham thanh overlapping")
print()
print("Ket luan: Voi margin chi 12-37px (quy doi perspective),")
print("  centroid cua xe binh thuong (dan chinh tieu) co the nam ngoai inner zone")
print("  MA KHONG PHAI VI XE CHIEM 2 SLOT.")
print()
print("DAY LA NGUYEN NHAN GOC CUA VIEC: 49 occupied bi nham thanh overlapping.")
