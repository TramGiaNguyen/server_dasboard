# Giải Pháp Toàn Diện: Frame Skipping & Mất Tracking

## 📋 Mục Lục
1. [Vấn Đề](#vấn-đề)
2. [Nguyên Nhân](#nguyên-nhân)
3. [Chẩn Đoán](#chẩn-đoán)
4. [Giải Pháp](#giải-pháp)
5. [Cấu Hình Tối Ưu](#cấu-hình-tối-ưu)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## 🔴 Vấn Đề

### Triệu Chứng
- Đối tượng "nhảy" vị trí đột ngột (teleport)
- Tracker mất ID, tạo ID mới cho cùng 1 xe
- Biển số xe bị gán nhầm
- FPS không ổn định, giật lag

### Hậu Quả
- ❌ Mất tracking continuity
- ❌ Duplicate entries trong database
- ❌ Sai lệch thống kê (1 xe tính thành nhiều xe)
- ❌ Trải nghiệm người dùng kém

---

## 🔍 Nguyên Nhân

### 1. RTSP Stream Issues (70% trường hợp)
```
Camera → Network → RTSP Buffer → Application
   ↓         ↓          ↓            ↓
Lag 20ms  Lag 50ms  Full 100ms   Process 30ms
```

**Vấn đề:**
- Network packet loss
- RTSP buffer đầy → đọc frame cũ
- Camera encoding lag
- Bandwidth không đủ

### 2. Processing Bottleneck (20% trường hợp)
```
Read Frame (5ms) → YOLO (50ms) → Tracking (10ms) → Draw (5ms)
                      ↑ BOTTLENECK
```

**Vấn đề:**
- YOLO inference chậm
- CPU/GPU quá tải
- Detection thread không sync

### 3. Tracker Configuration (10% trường hợp)
- Track buffer quá ngắn → mất ID nhanh
- Match threshold quá cao → không nhận diện lại object
- Detection confidence quá cao → miss object xa/mờ

---

## 🩺 Chẩn Đoán

### Bước 1: Kiểm Tra Frame Skip Rate

Thêm vào `services/parking_detection/camera.py`:

```python
# Sau dòng: frame_counter = 0
frame_skip_stats = {
    'prev_counter': 0,
    'skip_count': 0,
    'total_frames': 0,
    'last_report_time': time.time()
}

# Trong main loop, sau: frame_counter += 1
def check_frame_skip():
    stats = frame_skip_stats
    stats['total_frames'] += 1
    
    # Detect skip
    expected = stats['prev_counter'] + 1
    if frame_counter != expected and stats['prev_counter'] > 0:
        skip_amount = frame_counter - expected
        stats['skip_count'] += skip_amount
        print(f"⚠️  [SKIP] Frame {expected}-{frame_counter-1} skipped ({skip_amount} frames)")
    
    stats['prev_counter'] = frame_counter
    
    # Report every 5 seconds
    now = time.time()
    if now - stats['last_report_time'] >= 5.0:
        skip_rate = (stats['skip_count'] / stats['total_frames'] * 100) if stats['total_frames'] > 0 else 0
        fps = stats['total_frames'] / 5.0
        print(f"📊 [STATS] FPS: {fps:.1f}, Skip rate: {skip_rate:.1f}% ({stats['skip_count']}/{stats['total_frames']})")
        stats['skip_count'] = 0
        stats['total_frames'] = 0
        stats['last_report_time'] = now

check_frame_skip()
```

**Đánh giá:**
- ✅ Skip rate < 5%: Tốt
- ⚠️ Skip rate 5-15%: Chấp nhận được
- ❌ Skip rate > 15%: Cần khắc phục

### Bước 2: Kiểm Tra RTSP Latency

```python
# Trong main loop
rtsp_latency = rtsp_cap.get_latency()  # ms
if rtsp_latency > 200:
    print(f"⚠️  [RTSP] High latency: {rtsp_latency}ms")
```

### Bước 3: Kiểm Tra Processing Time

```python
# Trong main loop
loop_start = time.time()

# ... processing code ...

loop_time = (time.time() - loop_start) * 1000  # ms
if loop_time > 50:
    print(f"⚠️  [PERF] Slow loop: {loop_time:.1f}ms")
```

### Bước 4: Kiểm Tra Tracker Health

```python
# Sau tracking update
active_tracks = len([t for t in tracks if t.is_confirmed()])
tentative_tracks = len([t for t in tracks if t.is_tentative()])
lost_tracks = len([t for t in tracks if t.time_since_update > 5])

if frame_counter % 100 == 0:
    print(f"🎯 [TRACKER] Active: {active_tracks}, Tentative: {tentative_tracks}, Lost: {lost_tracks}")
```

---

## ✅ Giải Pháp

## Giải Pháp 1: Tối Ưu RTSP Stream (Ưu tiên cao)

### A. Sử dụng RTSPCapture với Buffer Cleaning

**File: `services/parking_detection/camera.py`**

```python
# Đã có sẵn - đảm bảo config đúng
rtsp_cap = RTSPCapture(video_path, buffer_size=2)  # Giữ buffer nhỏ
rtsp_cap.flush(wait_seconds=1.0)  # Xả buffer cũ sau sync
```

**Giải thích:**
- `buffer_size=2`: Chỉ giữ 2 frame mới nhất → latency thấp
- `flush()`: Xả frame cũ sau khi sync camera → bắt đầu với frame mới

### B. Thêm RTSP Reconnection Logic

**Tạo file: `shared/rtsp_reconnect.py`**

```python
import time
import cv2

class RobustRTSPCapture:
    """RTSP capture with auto-reconnect on failure"""
    
    def __init__(self, url, buffer_size=2, reconnect_delay=2.0):
        self.url = url
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.consecutive_failures = 0
        self.max_failures = 10
    
    def open(self):
        """Open RTSP stream with retry"""
        from shared.rtsp_capture import RTSPCapture
        
        for attempt in range(3):
            try:
                self.cap = RTSPCapture(self.url, buffer_size=self.buffer_size)
                if self.cap.open():
                    self.consecutive_failures = 0
                    print(f"✅ [RTSP] Connected: {self.url}")
                    return True
            except Exception as e:
                print(f"❌ [RTSP] Connection failed (attempt {attempt+1}/3): {e}")
                time.sleep(self.reconnect_delay)
        
        return False
    
    def read(self, timeout=0.1):
        """Read frame with auto-reconnect on failure"""
        if self.cap is None:
            if not self.open():
                return False, None
        
        ret, frame = self.cap.read(timeout=timeout)
        
        if not ret or frame is None:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.max_failures:
                print(f"⚠️  [RTSP] Too many failures ({self.consecutive_failures}), reconnecting...")
                self.cap = None
                if self.open():
                    self.cap.flush(wait_seconds=1.0)
                    return self.cap.read(timeout=timeout)
            
            return False, None
        
        self.consecutive_failures = 0
        return True, frame
    
    def flush(self, wait_seconds=1.0):
        """Flush buffer"""
        if self.cap:
            self.cap.flush(wait_seconds=wait_seconds)
```

**Sử dụng:**

```python
# Trong camera.py, thay thế RTSPCapture
from shared.rtsp_reconnect import RobustRTSPCapture

rtsp_cap = RobustRTSPCapture(video_path, buffer_size=2)
if not rtsp_cap.open():
    print(f"[ERROR] Failed to open RTSP stream: {video_path}")
    return
```

### C. Giảm RTSP Resolution (Nếu bandwidth thấp)

**File: `.env`**

```bash
# Sử dụng substream thay vì mainstream
# Mainstream: subtype=0 (1080p, 4Mbps)
# Substream:  subtype=1 (720p, 1Mbps)
PARKING_RTSP_URL=rtsp://user:pass@ip/cam/realmonitor?channel=1&subtype=1
```

---

## Giải Pháp 2: Tối Ưu Tracker Configuration

### A. Tăng Track Buffer (ByteTrack)

**File: `cfg/trackers/bytetrack_parking.yaml`**

```yaml
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 90          # Tăng từ 30 → 90 frames (3s @ 30fps)
match_thresh: 0.8
min_box_area: 10
mot20: False
```

**Giải thích:**
- `track_buffer=90`: Giữ track 3 giây sau khi mất detection
- Cho phép tracker "chờ" object xuất hiện lại sau khi bị skip frame

### B. Giảm Detection Threshold

**File: `services/parking_detection/camera.py`**

```python
# Trong ByteTrack block
results = model.track(
    detection_frame,
    persist=True,
    tracker=PARKING_TRACKER_CONFIG,
    conf=0.15,  # Giảm từ 0.20 → 0.15 để bắt object xa/mờ
    iou=0.5,
    verbose=False
)
```

### C. Thêm Kalman Filter Smoothing (DeepSORT)

**File: `services/parking_detection/camera.py`**

```python
# Trong DeepSORT block, sau khi có tracks
def smooth_bbox_with_kalman(track):
    """Use Kalman prediction to smooth bbox when detection is weak"""
    tsu = getattr(track, "time_since_update", 0)
    
    if tsu == 0:
        # Fresh detection - use original bbox
        try:
            return track.to_ltrb(orig=True, orig_strict=True)
        except:
            pass
    
    # No fresh detection - use Kalman prediction
    return track.to_ltrb()

# Áp dụng
for track in tracks:
    bbox = smooth_bbox_with_kalman(track)
    # ... use bbox ...
```

---

## Giải Pháp 3: Detection Thread Improvements

### A. Thêm Frame Interpolation

**File: `services/parking_detection/detection_thread.py`**

Thêm vào class `DetectionThread`:

```python
def __init__(self, ...):
    # ... existing code ...
    self.last_valid_result = None
    self.result_cache_ttl = 5  # frames

def get_result(self, timeout=0.001, frame_counter=None):
    """Get result with fallback to cached result"""
    try:
        result = self.output_queue.get(timeout=timeout)
        self.last_valid_result = result
        return result
    except queue.Empty:
        # No new result - use cached result if recent
        if self.last_valid_result and frame_counter:
            cached_detections, cached_frame = self.last_valid_result
            frame_diff = frame_counter - cached_frame
            
            if frame_diff <= self.result_cache_ttl:
                # Cache still valid - reuse (tracker will predict new positions)
                return self.last_valid_result
        
        return None
```

### B. Adaptive Queue Size

```python
def __init__(self, ...):
    # ... existing code ...
    self.adaptive_queue = True
    self.queue_size = 2

def _adjust_queue_size(self):
    """Adjust queue size based on processing speed"""
    if not self.adaptive_queue:
        return
    
    input_size = self.input_queue.qsize()
    output_size = self.output_queue.qsize()
    
    # If input queue often full, increase size
    if input_size >= self.queue_size - 1:
        self.queue_size = min(5, self.queue_size + 1)
        print(f"📈 [DETECTION THREAD] Increased queue size to {self.queue_size}")
    
    # If output queue often empty, decrease size
    elif output_size == 0 and self.queue_size > 2:
        self.queue_size = max(2, self.queue_size - 1)
        print(f"📉 [DETECTION THREAD] Decreased queue size to {self.queue_size}")
```

---

## Giải Pháp 4: Frame Rate Control

### A. Adaptive FPS

**File: `services/parking_detection/camera.py`**

```python
# Sau khi khởi tạo biến
target_fps = 30
adaptive_fps = True
frame_time_history = []

# Trong main loop
loop_start = time.time()

# ... processing ...

loop_time = time.time() - loop_start
frame_time_history.append(loop_time)

if len(frame_time_history) > 30:
    frame_time_history.pop(0)

# Adaptive sleep
if adaptive_fps:
    avg_loop_time = sum(frame_time_history) / len(frame_time_history)
    target_frame_time = 1.0 / target_fps
    
    if avg_loop_time < target_frame_time:
        sleep_time = target_frame_time - avg_loop_time
        time.sleep(sleep_time)
    elif avg_loop_time > target_frame_time * 1.5:
        # Processing too slow - skip next frame
        if is_stream:
            rtsp_cap.read(timeout=0.001)  # Discard one frame
```

### B. Priority Frame Processing

```python
# Chỉ process frame quan trọng
PROCESS_EVERY_N_FRAMES = 1  # Default: process every frame

if frame_counter % PROCESS_EVERY_N_FRAMES != 0:
    # Skip processing, but still update tracker with empty detections
    if PARKING_USE_BYTETRACK:
        # ByteTrack can handle empty detections
        pass
    else:
        # DeepSORT needs periodic updates
        _ds_tracks = _parking_ds.update_tracks([], frame=frame)
    continue
```

---

## Giải Pháp 5: Network Optimization

### A. QoS Configuration (Router/Switch)

```bash
# Ưu tiên RTSP traffic
# Cấu hình trên router/switch:
# - RTSP port 554: High priority
# - Camera IP: High priority
# - DSCP marking: EF (Expedited Forwarding)
```

### B. Dedicated Network

```
Camera 1 ──┐
Camera 2 ──┼── Switch ── Dedicated NIC ── Server
Camera 3 ──┘              (eth1)
                            │
                          eth0 ── Internet
```

### C. Multicast RTSP (Nếu nhiều client)

```bash
# Cấu hình camera để stream multicast
# Giảm bandwidth: 1 stream → N clients
PARKING_RTSP_URL=rtsp://ip/multicast
```

---

## 🎯 Cấu Hình Tối Ưu

### Preset 1: High Performance + Stable Network

**File: `.env`**
```bash
# Performance
PARKING_USE_HALF_PRECISION=True
PARKING_USE_CLAHE=False
PARKING_USE_DETECTION_THREAD=True

# RTSP
PARKING_RTSP_URL=rtsp://user:pass@ip/cam/realmonitor?channel=1&subtype=0  # Mainstream

# Tracking
PARKING_USE_BYTETRACK=True
```

**File: `cfg/trackers/bytetrack_parking.yaml`**
```yaml
track_buffer: 60
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
match_thresh: 0.8
```

**Kết quả:** FPS cao, tracking ổn định, skip rate <5%

---

### Preset 2: Unstable Network

**File: `.env`**
```bash
# Performance - tắt detection thread để tránh async lag
PARKING_USE_HALF_PRECISION=True
PARKING_USE_CLAHE=False
PARKING_USE_DETECTION_THREAD=False

# RTSP - dùng substream để giảm bandwidth
PARKING_RTSP_URL=rtsp://user:pass@ip/cam/realmonitor?channel=1&subtype=1  # Substream

# Tracking
PARKING_USE_BYTETRACK=True
```

**File: `cfg/trackers/bytetrack_parking.yaml`**
```yaml
track_buffer: 90        # Tăng lên 3s
track_high_thresh: 0.4  # Giảm threshold
track_low_thresh: 0.1
new_track_thresh: 0.5
match_thresh: 0.7       # Giảm để dễ match
```

**File: `services/parking_detection/camera.py`**
```python
# Giảm detection confidence
conf=0.15  # Thay vì 0.20

# Tăng buffer size
rtsp_cap = RobustRTSPCapture(video_path, buffer_size=3)  # Thay vì 2
```

**Kết quả:** FPS thấp hơn nhưng tracking ổn định hơn, skip rate <10%

---

### Preset 3: Low-End Hardware

**File: `.env`**
```bash
# Performance
PARKING_USE_HALF_PRECISION=False  # CPU không hỗ trợ
PARKING_USE_CLAHE=False
PARKING_USE_DETECTION_THREAD=False

# RTSP - substream
PARKING_RTSP_URL=rtsp://user:pass@ip/cam/realmonitor?channel=1&subtype=1

# Tracking
PARKING_USE_BYTETRACK=False  # Dùng DeepSORT nhẹ hơn
```

**File: `services/parking_detection/camera.py`**
```python
# Giảm resolution detection
DETECTION_WIDTH = 480   # Thay vì 640
DETECTION_HEIGHT = 240  # Thay vì 320

# Process mỗi 2 frames
PROCESS_INTERVAL = 2
```

**Kết quả:** FPS thấp (~10-15) nhưng ổn định, skip rate <15%

---

## 📊 Monitoring

### Dashboard Metrics

Thêm vào `api/routes.py`:

```python
@app.route('/api/parking/metrics')
def parking_metrics():
    """Real-time performance metrics"""
    return jsonify({
        'fps': shared_state.parking_fps,
        'skip_rate': shared_state.parking_skip_rate,
        'latency_ms': shared_state.parking_latency,
        'active_tracks': shared_state.parking_active_tracks,
        'lost_tracks': shared_state.parking_lost_tracks,
        'detection_time_ms': shared_state.parking_detection_time,
        'tracking_time_ms': shared_state.parking_tracking_time,
    })
```

### Grafana Dashboard (Optional)

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

**Metrics to track:**
- FPS (target: >20)
- Skip rate (target: <5%)
- Latency (target: <100ms)
- Active tracks
- Lost tracks per minute

---

## 🐛 Troubleshooting

### Vấn Đề 1: Skip rate >20%

**Chẩn đoán:**
```bash
# Kiểm tra network
ping <camera_ip>  # Latency <10ms là tốt

# Kiểm tra bandwidth
iperf3 -c <camera_ip>  # >10Mbps cho mainstream

# Kiểm tra CPU/GPU
nvidia-smi  # GPU usage <80%
htop        # CPU usage <80%
```

**Giải pháp:**
1. Dùng substream (giảm bandwidth)
2. Tắt detection thread
3. Tăng track buffer
4. Dedicated network

---

### Vấn Đề 2: Tracker mất ID liên tục

**Chẩn đoán:**
```python
# Log track lifetime
for track in tracks:
    if track.is_confirmed():
        lifetime = frame_counter - track.start_frame
        if lifetime < 30:  # <1s
            print(f"⚠️  Short-lived track: {track.track_id} ({lifetime} frames)")
```

**Giải pháp:**
1. Tăng `track_buffer` lên 90-120
2. Giảm `new_track_thresh` xuống 0.5
3. Giảm `match_thresh` xuống 0.7
4. Giảm detection `conf` xuống 0.15

---

### Vấn Đề 3: Detection thread gây lag

**Triệu chứng:**
- FPS không ổn định
- Frame counter nhảy cóc
- Queue đầy liên tục

**Giải pháp:**
```bash
# Tắt detection thread
PARKING_USE_DETECTION_THREAD=False
```

Hoặc tăng queue size:

```python
# detection_thread.py
self.input_queue = queue.Queue(maxsize=5)  # Thay vì 2
self.output_queue = queue.Queue(maxsize=5)
```

---

### Vấn Đề 4: RTSP reconnect liên tục

**Chẩn đoán:**
```bash
# Kiểm tra RTSP stream
ffmpeg -i "rtsp://user:pass@ip/cam/realmonitor?channel=1&subtype=0" -t 10 test.mp4

# Kiểm tra camera
curl http://<camera_ip>
```

**Giải pháp:**
1. Kiểm tra username/password
2. Kiểm tra camera firmware
3. Reboot camera
4. Kiểm tra firewall
5. Dùng TCP thay vì UDP:
   ```bash
   PARKING_RTSP_URL=rtsp://user:pass@ip/cam/realmonitor?channel=1&subtype=0&tcp
   ```

---

## 📈 Kết Quả Mong Đợi

### Trước Tối Ưu
- FPS: 12-18 (không ổn định)
- Skip rate: 15-30%
- Latency: 150-300ms
- Lost tracks: 5-10/phút

### Sau Tối Ưu
- FPS: 25-30 (ổn định)
- Skip rate: <5%
- Latency: 50-100ms
- Lost tracks: <2/phút

---

## 🎓 Best Practices

### 1. Network
- ✅ Dedicated network cho cameras
- ✅ Gigabit switch
- ✅ Cat6 cables
- ✅ QoS enabled
- ❌ WiFi cameras (nếu có thể)

### 2. Hardware
- ✅ GPU NVIDIA (RTX series)
- ✅ SSD cho OS
- ✅ 16GB+ RAM
- ✅ Multi-core CPU (8+ cores)

### 3. Configuration
- ✅ ByteTrack cho tracking
- ✅ Half precision trên GPU
- ✅ Buffer size nhỏ (2-3)
- ✅ Track buffer cao (60-90)
- ✅ Monitoring enabled

### 4. Maintenance
- ✅ Kiểm tra metrics hàng ngày
- ✅ Reboot camera hàng tuần
- ✅ Update firmware định kỳ
- ✅ Clean lens hàng tháng

---

## 📚 Tài Liệu Tham Khảo

- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
- [RTSP RFC 2326](https://tools.ietf.org/html/rfc2326)
- [YOLOv8 Docs](https://docs.ultralytics.com/)

---

## 🆘 Hỗ Trợ

Nếu vẫn gặp vấn đề sau khi áp dụng các giải pháp trên:

1. Thu thập logs:
   ```bash
   python main.py > parking.log 2>&1
   ```

2. Chụp screenshot metrics dashboard

3. Ghi lại:
   - FPS trung bình
   - Skip rate
   - Network topology
   - Hardware specs

4. Liên hệ support với thông tin trên
