"""
Tinh dien tich inner zone / outer zone cua tung slot.
De hieu tai sao overlapping co nhieu false positive.
"""

"""
Tinh dien tich inner zone / outer zone cua tung slot.
De hieu tai sao overlapping co nhieu false positive.
"""

import numpy as np
import cv2

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

def create_inner_zones(areas, shrink_percentage=0.20):
    _FAR_SLOTS = set(range(0, 14)) | {17, 18}
    _FAR_SLOT_SHRINK = 0.35
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

areas = define_parking_areas()
inner_areas = create_inner_zones(areas)

print(f"{'Slot':>5} | {'Outer Area':>12} | {'Inner Area':>12} | {'Ratio (%)':>10} | {'Status':>6}")
print("-" * 60)

total_outer = 0
total_inner = 0

for i in range(len(areas)):
    outer_pts = np.array(areas[i], np.int32)
    inner_pts = np.array(inner_areas[i], np.int32)

    outer_area = cv2.contourArea(outer_pts)
    inner_area = cv2.contourArea(inner_pts)

    # Shrink percentage cua slot i
    if i <= 18:
        shrink_pct = 0.35
    else:
        shrink_pct = 0.20

    ratio = inner_area / outer_area * 100 if outer_area > 0 else 0
    total_outer += outer_area
    total_inner += inner_area

    status = "FAR" if i <= 18 else "CLOSE"
    print(f"{i:>5} | {outer_area:>12.0f} | {inner_area:>12.0f} | {ratio:>9.1f}% | {status}")

print("-" * 60)
print(f"{'AVG':>5} | {total_outer/len(areas):>12.0f} | {total_inner/len(areas):>12.0f} | {total_inner/total_outer*100:>9.1f}%")
print()

# Chi tiet hon: khoang cach tu inner boundary den outer boundary
print("\nKhoang cach tu inner boundary den outer boundary (theo chieu ngang):")
print(f"{'Slot':>5} | {'Outer width':>12} | {'Inner width':>12} | {'Margin (px)':>12} | {'Margin (%)':>10}")
print("-" * 65)

for i in range(len(areas)):
    outer_pts = np.array(areas[i], np.int32)
    inner_pts = np.array(inner_areas[i], np.int32)

    # Tim min/max x cho outer va inner
    outer_xs = [p[0] for p in areas[i]]
    inner_xs = [p[0] for p in inner_areas[i]]

    outer_min_x, outer_max_x = min(outer_xs), max(outer_xs)
    inner_min_x, inner_max_x = min(inner_xs), max(inner_xs)

    outer_width = outer_max_x - outer_min_x
    inner_width = inner_max_x - inner_min_x

    margin_left = inner_min_x - outer_min_x
    margin_right = outer_max_x - inner_max_x
    margin_avg = (margin_left + margin_right) / 2

    margin_pct = margin_avg / outer_width * 100 if outer_width > 0 else 0

    status = "FAR" if i <= 18 else "CLOSE"
    print(f"{i:>5} | {outer_width:>12.0f} | {inner_width:>12.0f} | {margin_avg:>12.1f} | {margin_pct:>9.1f}% | {status}")
