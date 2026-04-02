# Fix: Late Plate Sync Log Spam

## 🐛 Vấn Đề

### Triệu chứng
```
[QUEUE→PARKING] Late plate sync: ingress_seq 1 ← 61K62423 (0.99)
[QUEUE→PARKING] Late plate sync: ingress_seq 5 ← 61K57296 (0.98)
[QUEUE→PARKING] Late plate sync: ingress_seq 1 ← 61K62423 (0.99)
[QUEUE→PARKING] Late plate sync: ingress_seq 5 ← 61K57296 (0.98)
... (lặp lại liên tục mỗi frame)
```

### Nguyên nhân
Hàm `_sync_reserved_plate_updates()` trong `camera.py` có logic sai:

```python
# Logic CŨ (SAI)
if new_plate and (old_plate is None or new_conf >= old_conf or old_plate != new_plate):
    # Update và append vào updated_seqs
    updated_seqs.append((ingress_seq, new_plate, new_conf))
```

**Vấn đề:** Điều kiện `new_conf >= old_conf` luôn đúng khi `new_conf == old_conf`, dẫn đến:
- Mỗi frame (~30 lần/giây) hàm này chạy
- Nó cứ append cùng 1 plate vào `updated_seqs`
- Log spam không dừng

---

## ✅ Giải Pháp

### Sửa Logic Direct Sync

**File:** `services/parking_detection/camera.py`

**Thay đổi:**

```python
# Logic MỚI (ĐÚNG)
# Chỉ update nếu có thay đổi thực sự
has_change = False
if new_plate and old_plate is None:
    # Plate mới xuất hiện lần đầu
    has_change = True
elif new_plate and old_plate and new_plate != old_plate:
    # Plate thay đổi (OCR correction)
    has_change = True
elif new_plate and old_plate == new_plate and new_conf > old_conf:
    # Cùng plate nhưng confidence cao hơn (chỉ update nếu THỰC SỰ cao hơn, không phải bằng)
    has_change = True

if has_change:
    match_info['plate'] = new_plate
    match_info['conf'] = new_conf
    match_info['plate_status'] = 'confirmed' if new_conf >= 0.80 else 'provisional'
    match_info['queue_ts'] = item.get('timestamp')
    item['assigned'] = True
    updated_seqs.append((ingress_seq, new_plate, new_conf))
```

**Giải thích:**
- ✅ Chỉ append khi plate **thực sự thay đổi**
- ✅ Confidence phải **cao hơn** (không phải bằng)
- ✅ Tránh log spam

---

### Sửa Logic Fallback Sync

**Thêm check để tránh duplicate:**

```python
# Chỉ update nếu chưa có plate hoặc plate khác
old_plate = match_info.get('plate')
if old_plate and old_plate == late_plate:
    # Đã có plate này rồi, skip để tránh log spam
    continue

match_info['plate'] = late_plate
match_info['conf'] = late_conf
# ... rest of code
updated_seqs.append((ingress_seq, late_plate, late_conf))
```

---

## 📊 Kết Quả

### Trước Fix
```
[QUEUE→PARKING] Late plate sync: ingress_seq 1 ← 61K62423 (0.99)
[QUEUE→PARKING] Late plate sync: ingress_seq 5 ← 61K57296 (0.98)
[QUEUE→PARKING] Late plate sync: ingress_seq 1 ← 61K62423 (0.99)
[QUEUE→PARKING] Late plate sync: ingress_seq 5 ← 61K57296 (0.98)
... (30 lần/giây × 2 plates = 60 logs/giây)
```

### Sau Fix
```
[QUEUE→PARKING] Late plate sync: ingress_seq 1 ← 61K62423 (0.99)
[QUEUE→PARKING] Late plate sync: ingress_seq 5 ← 61K57296 (0.98)
... (chỉ log 1 lần khi plate thực sự update)
```

---

## 🎯 Các Trường Hợp Log Hợp Lệ

Sau fix, log chỉ xuất hiện khi:

1. **Plate mới xuất hiện lần đầu**
   ```
   [QUEUE→PARKING] Late plate sync: ingress_seq 10 ← 51K12345 (0.85)
   ```

2. **OCR correction (plate thay đổi)**
   ```
   [QUEUE→PARKING] Late plate sync: ingress_seq 10 ← 51K12346 (0.92)
   # OCR đã sửa từ 51K12345 → 51K12346
   ```

3. **Confidence tăng lên**
   ```
   [QUEUE→PARKING] Late plate sync: ingress_seq 10 ← 51K12345 (0.95)
   # Confidence tăng từ 0.85 → 0.95
   ```

---

## 🔍 Debug

Nếu vẫn thấy log spam, kiểm tra:

### 1. Confidence có thay đổi liên tục không?

```python
# Thêm debug log
print(f"[DEBUG] ingress_seq={ingress_seq}, old_plate={old_plate}, new_plate={new_plate}, "
      f"old_conf={old_conf:.4f}, new_conf={new_conf:.4f}, has_change={has_change}")
```

### 2. Item có bị reset không?

```python
# Kiểm tra item['assigned']
if item.get('assigned'):
    print(f"[WARNING] Item already assigned but still in queue: {item}")
```

### 3. Queue có bị duplicate không?

```python
# Đếm số lượng item với cùng ingress_seq
from collections import Counter
seqs = [_fifo_reserved_seq(p) for p in plate_fifo_queue if _fifo_reserved_seq(p)]
duplicates = {seq: count for seq, count in Counter(seqs).items() if count > 1}
if duplicates:
    print(f"[WARNING] Duplicate ingress_seq in queue: {duplicates}")
```

---

## 📝 Files Changed

- ✅ `services/parking_detection/camera.py` - Fixed `_sync_reserved_plate_updates()`

---

## 🚀 Testing

### Test Case 1: Plate mới
1. Xe vào cổng → tạo ingress_seq mới
2. OCR hoàn thành → plate sync
3. **Kỳ vọng:** Log xuất hiện 1 lần duy nhất

### Test Case 2: OCR correction
1. OCR ban đầu: 51K12345 (conf=0.85)
2. OCR cải thiện: 51K12346 (conf=0.92)
3. **Kỳ vọng:** Log xuất hiện 1 lần khi correction

### Test Case 3: Confidence tăng
1. OCR ban đầu: 51K12345 (conf=0.85)
2. OCR cải thiện: 51K12345 (conf=0.95)
3. **Kỳ vọng:** Log xuất hiện 1 lần khi confidence tăng

### Test Case 4: Không thay đổi
1. OCR: 51K12345 (conf=0.95)
2. Frame tiếp theo: vẫn 51K12345 (conf=0.95)
3. **Kỳ vọng:** KHÔNG có log

---

## ✅ Verification

Sau khi deploy fix, kiểm tra:

```bash
# Đếm số log "Late plate sync" trong 10 giây
timeout 10 python main.py 2>&1 | grep "Late plate sync" | wc -l

# Trước fix: ~600 logs (60 logs/giây × 10 giây)
# Sau fix: ~2-5 logs (chỉ khi có xe mới vào)
```

---

## 🎓 Lesson Learned

### Vấn đề với `>=` operator

```python
# SAI - sẽ luôn đúng khi bằng nhau
if new_conf >= old_conf:
    update()

# ĐÚNG - chỉ đúng khi thực sự lớn hơn
if new_conf > old_conf:
    update()
```

### Best Practice

Khi implement sync/update logic:
1. ✅ Kiểm tra **thay đổi thực sự** trước khi update
2. ✅ Dùng `>` thay vì `>=` cho numeric comparison
3. ✅ Log chỉ khi có action thực sự
4. ✅ Test với data không thay đổi để phát hiện spam

---

## 📞 Support

Nếu vẫn gặp log spam:
1. Kiểm tra debug logs ở trên
2. Verify logic trong `_sync_reserved_plate_updates()`
3. Kiểm tra `plate_fifo_queue` có duplicate không
