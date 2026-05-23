# 3.8 License Plate Recognition and Cross-Camera Vehicle Association

## Mở đầu

Trong một hệ thống quản lý bãi đỗ thông minh hoàn chỉnh, việc nhận diện chủ phương tiện và theo dõi hành trình của xe từ cổng vào đến vị trí đỗ là yêu cầu bắt buộc để tính cước tự động, phát hiện quá giờ, và cảnh báo danh sách đen. Tuy nhiên, camera cổng và camera bãi đỗ hoạt động ở hai vị trí vật lý khác nhau, với các góc nhìn, độ phân giải, và điều kiện ánh sáng riêng biệt. Xe di chuyển giữa hai camera trong khoảng thời gian từ vài giây đến vài phút — trong suốt quãng đường này, hệ thống phải duy trì liên kết danh tính xe mà không có sự hỗ trợ trực tiếp từ cảm biến nào.

Bước **ghép nối liên camera** (cross-camera vehicle association) là thành phần hoàn thiện chuỗi xử lý end-to-end của hệ thống: từ nhận diện biển số tại cổng (LPR), qua hàng đợi trung gian (FIFO), đến ghép cặp với vị trí đỗ tại camera bãi, và cuối cùng là cập nhật cơ sở dữ liệu parking session. Nếu bước này thất bại, toàn bộ pipeline occupancy detection trở nên vô nghĩa vì hệ thống không thể biết chiếc xe nào đang chiếm vị trí nào, dẫn đến sai lệch thời gian đỗ, tính cước sai, và cảnh báo nhầm.

---

## 3.8.1 System Architecture for Cross-Camera Association

Tổng quan luồng xử lý ghép nối liên camera được mô tả trong Hình 10. Hệ thống bao gồm bốn thành phần chính hoạt động song song qua cơ chế đa luồng:

\begin{enumerate}
  \item **Gate Detection Thread** — Camera cổng liên tục phát hiện và theo dõi xe qua YOLO + ByteTrack. Khi xe vượt đồng thời qua cả hai vạch kẻ LINE\_1 và LINE\_2 theo hướng vào (in), hệ thống tạo một bản ghi FIFO entry với \texttt{ingress\_seq} duy nhất tăng tuần tự, thời gian \texttt{timestamp}, và trường \texttt{plate} ban đầu để trống.
  
  \item **Gate OCR Thread** — Nhận dạng biển số bằng PP-OCR v5 Server (ONNX). Kết quả OCR được ghi vào FIFO entry tương ứng theo \texttt{track\_id}. Nếu OCR không đọc được trong 8 giây, hệ thống vẫn tiếp tục với trạng thái \texttt{plate=None} và ghi nhận ảnh crop xe làm bằng chứng.
  
  \item **Parking Detection Thread** — Camera bãi đỗ phát hiện xe vào vùng entry trigger. Mỗi sự kiện trigger được đẩy vào \texttt{parking\_trigger\_queue}. Tại mỗi frame, thread này gọi \texttt{\_try\_pair\_plate\_fifo\_with\_parking\_trigger()} để ghép cặp trigger với FIFO entry chưa được gán cũ nhất, kiểm tra ràng buộc thời gian $\Delta T \leq \tau_{\max}$.
  
  \item **Database Writer** — GateDBWriter thread đợi kết quả OCR (conf $\geq 0.30$) hoặc timeout 8 giây, sau đó ghi một bản ghi \texttt{GateLog} và \texttt{ParkingSession} vào PostgreSQL. ParkingDBWriter tương ứng cập nhật slot, thời gian đỗ, và trạng thái session.
\end{enumerate}

```
┌──────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│   CAMERA CỔNG       │     │      FIFO QUEUE      │     │   CAMERA BÃI ĐỖ     │
│                      │     │  (shared/state.py)  │     │                      │
│  YOLO + ByteTrack   │────▶│                     │◀────│  YOLO + ByteTrack    │
│  ↓                  │     │  E_i = (P_i, T_in)  │     │  ↓                   │
│  LINE 1+2 crossing  │     │  + ingress_seq      │     │  Entry trigger zone  │
│  ↓                  │     │  + assigned flag    │     │  ↓                   │
│  OCR (PP-OCR v5)    │     │  + reserved_seq     │     │  Parking trigger Q   │
│  ↓                  │     │                     │     │  ↓                   │
│  plate_text + conf  │────▶│  Pairing logic       │◀────│  centroid position    │
│                      │     │  ΔT ≤ 45s check     │     │  ↓                   │
└──────────────────────┘     └──────────────────────┘     │  Match + Association  │
                              FIFO entry update ◀────────│  ↓                   │
                              (plate, conf backfill)     │  ParkingSession + DB  │
                                                         └──────────────────────┘
```

*Hình 10. Luồng xử lý ghép nối xe liên camera (cross-camera vehicle association pipeline).*

---

## 3.8.2 License Plate Recognition Module

### Mô hình OCR

Hệ thống sử dụng **PP-OCR v5 Server** thông qua **ONNX Runtime** với CUDA provider, bao gồm hai mô hình con:

- **Text Detection** (`PP-OCRv5_server_det_infer.onnx`): Dựa trên kiến trúc DB (Differentiable Binarization), phát hiện vùng chứa ký tự trên biển số. Input được resize với cạnh dài tối đa 960 px, chuẩn hóa theo mean=[0.485, 0.456, 0.406] và std=[0.229, 0.224, 0.225].
- **Text Recognition** (`PP-OCRv5_server_rec_infer.onnx`): Dựa trên kiến trúc CRNN (Convolutional Recurrent Neural Network) với CTC (Connectionist Temporal Classification) decoder. Input có chiều cao cố định 48 px, chiều rộng tối đa 320 px, xử lý theo batch size = 6.

Tham số cấu hình chính:

| Tham số | Giá trị | Mô tả |
|----------|---------|--------|
| `box_thresh` | 0.6 | Ngưỡng confidence cho vùng text được phát hiện |
| `unclip_ratio` | 1.6 | Hệ số giãn vùng phát hiện trước khi cắt |
| `drop_score` | 0.25 | Loại bỏ kết quả OCR có confidence thấp hơn ngưỡng |
| `Vehicle class` | car, bus, truck | Lọc chỉ nhận dạng phương tiện giao thông |

### Tiền xử lý ảnh biển số

Trước khi đưa vào PP-OCR, ảnh biển số được crop từ bounding box của YOLO plate detector (`best_plate_yolov8.pt`, conf=0.25) và đi qua pipeline tiền xử lý nâng cao (`enhanced_plate_preprocessing`):

1. **Upscale**: Phóng ảnh lên 4–10 lần tùy kích thước ban đầu, sử dụng `cv2.INTER_CUBIC` để bảo toàn chi tiết ký tự.
2. **Khử nhiễu**: Non-local Means denoising ($h=7$, search window 21×21) loại bỏ nhiễu sensor mà không làm mờ cạnh ký tự.
3. **Khử mờ**: Wiener deconvolution với PSF là Gaussian kernel ($k=9$, $\sigma=1.5$, $K=0.02$) khôi phục ảnh bị motion blur do xe di chuyển.
4. **Tăng tương phản**: CLAHE với `clipLimit=2.5`, `tileGridSize=(8,8)` cải thiện độ tương phản trong điều kiện ngược sáng hoặc thiếu sáng.
5. **Làm nét**: Unsharp masking ($w_1=1.6$, $w_2=-0.6$) tăng độ sắc nét của các cạnh ký tự.
6. **Chống loá**: Nếu ảnh quá sáng (mean > 180 hoặc >50% pixels sáng), áp dụng gamma correction ($\gamma=2.0$) và tăng saturation trong không gian HSV.

Pipeline này đặc biệt hiệu quả với biển số Việt Nam có kích thước nhỏ trên ảnh gốc (tỷ lệ chiều rộng biển số/chiều rộng xe chỉ 5%–95%), thường bị mờ hoặc thiếu sáng khi xe di chuyển qua cổng.

### Lọc false positive trong detection

Không phải tất cả bounding box được YOLO trả về đều là biển số thực. Hệ thống áp dụng bộ lọc hình học trước khi crop và nhận dạng:

- **Aspect ratio**: $1.05 < \text{AR} < 12.0$ (loại bỏ bounding box quá vuông hoặc quá dài)
- **Vị trí theo chiều dọc**: $0.20 < \frac{y_{\text{mid}}}{h_{\text{vehicle}}} < 0.97$ (tránh detect biển số trên nóc xe hoặc kính sau)
- **Kích thước tương đối**: $0.05 < \frac{w_{\text{box}}}{w_{\text{vehicle}}} < 0.95$

### Hậu xử lý biển số Việt Nam

Sau khi PP-OCR trả về chuỗi ký tự, hệ thống áp dụng tập luật sửa lỗi phổ biến cho biển số Việt Nam (Vietnamese plate grammar correction):

- Các cặp nhầm lẫn phổ biến: `0↔D`, `1↔L`, `1↔I`, `A↔4`, `O↔0`, `8↔B`
- Ràng buộc định dạng: 2–3 ký tự đầu là chữ, phần còn lại là số (có thể có thêm 1 chữ ở cuối cho biển 5 số)
- Loại bỏ các ký tự đặc biệt và khoảng trắng thừa

### Độ chính xác

Hệ thống không công bố số đo accuracy đơn lẻ cho module OCR trong báo cáo này. Tuy nhiên, cơ chế đảm bảo chất lượng bao gồm: majority voting theo track ID (yêu cầu tối thiểu 3 ký tự trùng khớp trên 60% số frame), provisional OCR (confidence $\geq 0.40$ được coi là tạm chấp nhận được), và confirmed OCR (confidence $\geq 0.80$ được coi là đáng tin cậy). Ngưỡng ghi vào cơ sở dữ liệu là confidence $\geq 0.30$.

---

## 3.8.3 Vehicle Association Mechanism

### Primary Method: Exact String Matching with Temporal Window

Khi một xe đi vào bãi đỗ, hệ thống tạo bản ghi nhập bế:

\[
E_i = (P_i, T_i^{\text{in}}, \text{ingress\_seq}_i)
\]

trong đó $P_i$ là biển số được nhận dạng tại cổng (có thể là \texttt{None} nếu OCR chưa hoàn thành), $T_i^{\text{in}}$ là thời gian xe vượt qua LINE\_1 tại cổng, và $\text{ingress\_seq}_i$ là số thứ tự tăng duy nhất.

Khi xe được phát hiện trong vùng entry trigger của camera bãi đỗ tại thời điểm $T_j^{\text{park}}$, hệ thống thực hiện ghép cặp nếu:

\[
\Delta T = T_j^{\text{park}} - T_i^{\text{in}} \leq \tau_{\max}
\]

trong đó $\tau_{\max} = 45$ giây là ngưỡng thời gian tối đa cho phép giữa lúc xe vào cổng và lúc xe vào bãi đỗ. Giá trị này được chọn dựa trên quan sát thực tế: thời gian di chuyển từ cổng đến vị trí đỗ xa nhất trong bãi không vượt quá 45 giây đối với xe con.

Ghép cặp thành công được định nghĩa:

\[
\text{Match}(E_i, V_j) = \begin{cases}
1, & \text{nếu } P_i = P_j \text{ và } \Delta T \leq \tau_{\max} \\
0, & \text{nếu } P_i \neq P_j \text{ hoặc } \Delta T > \tau_{\max}
\end{cases}
\]

Trong trường hợp $P_i = \text{None}$ (OCR chưa hoàn thành), ghép cặp vẫn được thực hiện dựa trên ràng buộc thời gian, và biển số sẽ được backfill sau khi OCR trả kết quả. Trường $\text{reserved\_ingress\_seq}$ được thiết lập trên FIFO entry để đánh dấu entry đã được ghép, ngăn việc gán nhiều xe cho cùng một vị trí đỗ.

### Fallback Method: ByteTrack Trajectory Association

Khi OCR thất bại hoàn toàn (biển số không thể đọc được) hoặc $\Delta T$ vượt ngưỡng, hệ thống chuyển sang cơ chế fallback dựa trên trajectory của ByteTrack:

**Cơ chế Re-anchor** — Khi một track xe mới xuất hiện tại camera bãi đỗ mà không có FIFO entry nào khớp, hệ thống kiểm tra xem track đó có phải là continuation của một track đã mất track ID tại camera cổng hay không. Điều kiện re-anchor:

\[
\text{IoU}(\text{bbox}_{\text{old}}, \text{bbox}_{\text{new}}) \geq 0.30 \quad \text{hoặc} \quad \text{score}_{\text{match}} \geq 0.60
\]

với $\text{score}_{\text{match}}$ là tổ hợp có trọng số của IoU, khoảng cách centroid, và tốc độ di chuyển dự kiến. Ngoài ra, xe nguồn phải chưa vượt qua bất kỳ đường kẻ nào (không có direction set) để tránh ghép nhầm xe đang ra với xe đang vào.

**Transit Tracking** — Trong khoảng thời gian xe di chuyển từ cổng đến bãi đỗ, ByteTrack duy trì track ID ổn định. Hệ thống theo dõi chuyển động bằng IoU và centroid distance để phát hiện trường hợp track bị mất tạm thời do occlusion. Ngưỡng IoU cho xe đang di chuyển là 0.4, cho xe đang đứng yên là 0.6.

### FIFO Queue and Timeout Mechanism

FIFO queue được implement bằng Python list với `threading.Lock` để đảm bảo tính an toàn trong môi trường đa luồng:

```python
plate_fifo_queue = []            # List[dict] — no max size
plate_fifo_lock = threading.Lock()

parking_trigger_queue = []       # List[dict]
parking_trigger_lock = threading.Lock()
```

Mỗi entry trong FIFO queue có cấu trúc:

\[
E_i = \left\{
  \begin{array}{l}
  \text{ingress\_seq}: \text{integer} \\
  \text{plate}: \text{string} \cup \{\text{None}\} \\
  \text{conf}: [0.0, 1.0] \\
  \text{timestamp}: \text{datetime} \\
  \text{assigned}: \text{boolean} \\
  \text{reserved\_ingress\_seq}: \text{integer} \cup \{\text{None}\} \\
  \text{gate\_track\_id}: \text{integer}
  \end{array}
\right\}
\]

Cơ chế timeout và dọn dẹp:

1. **Stale FIFO entry** ($\text{assigned} = \text{False}$ và tuổi $> 180$ giây): Xóa khỏi queue. Xe đã vào bãi nhưng không trigger parking, có thể đỗ ngoài vùng quản lý.
2. **Stale parking trigger** ($\Delta T < -5$ giây): Bỏ qua trigger vì xe trigger trước cả khi vào cổng — lỗi đồng bộ.
3. **Timeout FIFO entry** ($\Delta T > 45$ giây sau khi trigger): Đánh dấu FIFO entry là assigned, cho phép parking tiếp tục mà không chờ cổng.

---

## 3.8.4 Handling Recognition Failures and Robustness

### Trường hợp OCR thất bại

OCR có thể thất bại trong nhiều tình huống thực tế:

- **Góc nghiêng**: Xe đi qua cổng không thẳng hàng, biển số nghiêng $\theta > 15^\circ$ — PP-OCR phát hiện box nhưng recognition accuracy giảm mạnh.
- **Ánh sáng không đồng đều**: Xe đỗ dưới bóng cây, trời nắng gắt tạo glare, hoặc điều kiện thiếu sáng ban đêm.
- **Che khuất**: Biển số bị bẩn, bị che một phần bởi khung xe khác, hoặc dán decal không đúng chuẩn.
- **Motion blur**: Tốc độ xe qua cổng cao, camera 30fps không đủ để freezet ảnh sắc nét.
- **Ký tự không đúng chuẩn**: Biển số nước ngoài, biển số in tay, hoặc biển số mờ không thể phân biệt ký tự.

### Hybrid Fallback Strategy

Hệ thống xây dựng chiến lược fallback nhiều lớp để đảm bảo robustness:

**Lớp 1 — Majority Voting theo Track**: Với mỗi ByteTrack track ID, hệ thống thu thập kết quả OCR từ nhiều frame liên tiếp. Ký tự được giữ lại chỉ khi nó xuất hiện trong ít nhất $\lceil 0.6 \times n_{\text{frames}} \rceil$ frame. CTC canonical correction được áp dụng trước khi so sánh để chuẩn hóa các biến thể viết.

**Lớp 2 — Provisional OCR**: Nếu có ít nhất một frame đạt confidence $\geq 0.40$ trước khi xe rời khỏi vùng OCR, hệ thống coi đó là provisional result và ghi vào FIFO entry ngay. Điều này giảm latency trung bình của hệ thống vì không phải đợi tất cả các frame.

**Lớp 3 — Image Evidence on Failure**: Khi OCR không đọc được bất kỳ ký tự nào nhưng bounding box hình học hợp lệ (width $\geq 24$ px, $1.0 \leq \text{AR} \leq 10.0$), hệ thống vẫn lưu ảnh crop của vùng biển số vào thư mục `/static/gate_captures/` và ghi đường dẫn ảnh vào database. Admin có thể xác minh thủ công sau đó.

**Lớp 4 — EXIT Retry OCR**: Khi xe ra (cross LINE\_1 theo hướng ra), nếu `exit_plate` chưa được gán từ entry, hệ thống tự động queue một lượt OCR retry với ảnh từ camera cổng. Điều này tận dụng khả năng OCR tại thời điểm ra thay vì chỉ dựa vào kết quả tại thời điểm vào.

**Lớp 5 — ByteTrack Trajectory Continuity**: Nếu tất cả các lớp trên đều thất bại, hệ thống vẫn có thể ghép cặp dựa trên ByteTrack track ID và ràng buộc thời gian. Track ID được duy trì liên tục 300 frames (~10 giây) sau khi xe cuối cùng được phát hiện, giảm thiểu mất track do tạm thời occlusion.

### Minimum Dwell Time Protection

Để ngăn spurious exit events (xe chưa vào đã bị ghi ra), hệ thống áp dụng minimum dwell time:

\[
T_{\text{dwell}} \geq 30 \text{ giây}
\]

Xe chỉ được phép ghi nhận exit event nếu thời gian từ entry đến exit không nhỏ hơn 30 giây. Điều này ngăn trường hợp xe lùi nhầm qua vạch kẻ hoặc xe đi vòng qua cổng mà không thực sự vào bãi.

---

## 3.8.5 Applications Enabled by Association

Việc ghép nối thành công giữa camera cổng và camera bãi đỗ mở ra nhiều ứng dụng giá trị gia tăng:

### 3.8.5.1 Automatic Parking Duration Calculation

Khi một xe được ghép cặp tại bãi đỗ và sau đó được ghi nhận exit event tại cổng, hệ thống tự động tính thời gian đỗ:

\[
T_{\text{duration}} = T_{\text{exit}} - T_{\text{entry}}
\]

Trường `duration_minutes` trong bảng `parking_sessions` được cập nhật tự động khi vehicle exit được detect. Thời gian tính chính xác đến phút, phục vụ trực tiếp cho hệ thống tính cước.

### 3.8.5.2 Overstay Detection and Alerting

Với thời gian đỗ được theo dõi liên tục, hệ thống có thể cấu hình ngưỡng overstay cho từng loại người dùng:

| Người dùng | Ngưỡng overstay mặc định |
|------------|--------------------------|
| Sinh viên | 4 giờ |
| Nhân viên | 8 giờ |
| Khách | 2 giờ |

Khi thời gian đỗ vượt ngưỡng, sự kiện được ghi vào bảng `improper_parking_logs` với `event_type = 'overstay'`, đồng thời gửi thông báo đến mobile app của người dùng và cảnh báo cho nhân viên bảo vệ.

### 3.8.5.3 Blacklist Alerting

Khi một biển số được nhận dạng tại cổng hoặc bãi đỗ, hệ thống kiểm tra ngay lập tức với danh sách đen (blacklist). Blacklist có thể bao gồm:

- Xe không đăng ký (không có `ParkingSession` hợp lệ sau khi vào cổng)
- Xe quá hạn thanh toán
- Xe có lệnh truy nã hoặc trong danh sách theo dõi

Khi phát hiện xe trong blacklist, hệ thống phát âm thanh cảnh báo tại cổng (`warning-sound.mp3`), hiển thị cảnh báo trên giao diện web, và gửi notification đến tài khoản manager/guard.

### 3.8.5.4 Detailed Entry/Exit Statistics

Mỗi sự kiện ra/vào được ghi vào bảng `gate_logs` với đầy đủ metadata:

- Thời gian chính xác (timestamp với timezone)
- Biển số và confidence OCR
- Hướng đi (IN/OUT)
- Đường dẫn ảnh chụp tại cổng

Dữ liệu này cho phép:

- Thống kê lưu lượng xe theo giờ/ngày/tháng
- Phát hiện các pattern bất thường (xe ra vào nhiều lần trong thời gian ngắn, xe đỗ quá đêm thường xuyên)
- Truy vết lịch sử ra/vào của bất kỳ biển số nào
- Xác minh thủ công khi OCR thất bại (tra cứu ảnh theo timestamp)

---

*Hình 11 minh họa một ví dụ thành công về ghép nối biển số giữa camera cổng và camera bãi đỗ. Tại thời điểm $T_1$, xe biển số "29A-12345" được phát hiện tại cổng với confidence OCR 0.92. FIFO entry $E_1 = (\text{"29A-12345"}, T_1, 42)$ được tạo. Tại thời điểm $T_2 = T_1 + 23\text{s}$, xe được phát hiện tại vùng entry trigger của camera bãi đỗ. Vì $\Delta T = 23\text{s} \leq \tau_{\max} = 45\text{s}$ và $P_1 = P_2 = \text{"29A-12345"}$, ghép cặp thành công. Xe được gán vào slot số 7 và tracking tiếp tục cho đến khi xe rời đi.*

*Hình 12 minh họa trường hợp OCR thất bại và fallback mechanism. Tại $T_1$, xe được phát hiện tại cổng nhưng OCR không đọc được biển số. FIFO entry $E_2 = (\text{None}, T_1, 43)$ được tạo với trạng thái pending. Tại $T_2 = T_1 + 28\text{s}$, xe được phát hiện tại bãi đỗ. Vì $\Delta T = 28\text{s} \leq \tau_{\max}$, hệ thống vẫn ghép cặp dựa trên ràng buộc thời gian. Sau đó, khi xe ra tại $T_3$, cổng thực hiện EXIT retry OCR, đọc được biển số "51A-67890" và backfill vào $E_2$. ParkingSession được cập nhật đầy đủ.*
