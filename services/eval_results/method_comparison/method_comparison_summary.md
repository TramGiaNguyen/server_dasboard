## 5.4 Hiệu suất suy luận chiếm chỗ tổng thể

### Bảng 5.4. Tổng hợp các chỉ số hiệu suất chính

| Phương pháp | Accuracy | Macro-F1 (3-class) | Weighted-F1 (3-class) | Flickering Rate | Outside F1 |
|---|---|---|---|---|---|
| Rectangular ROI | 99.73% | 66.56% | 99.63% | 0.2053 | 87.27% |
| Polygon Only | 99.77% | 66.61% | 99.67% | 0.1842 | 87.27% |
| Polygon + Inner-Core | 99.74% | 87.08% | 99.79% | 0.3421 | 87.27% |
| Proposed Framework | 99.71% | 86.37% | 99.76% | 0.3000 | 87.27% |

### Bảng 5.5. Chi tiết theo lớp (Precision / Recall / F1)

| Phương pháp | Available P/R/F1 | Occupied P/R/F1 | Overlapping P/R/F1 |
|---|---|---|---|
| Rectangular ROI | 100.0/99.7/99.8 | 99.7/100.0/99.8 | 0.0/0.0/0.0 |
| Polygon Only | 100.0/99.9/100.0 | 99.7/100.0/99.9 | 0.0/0.0/0.0 |
| Polygon + Inner-Core | 100.0/99.9/100.0 | 100.0/99.7/99.8 | 44.8/97.5/61.4 |
| Proposed Framework | 99.9/99.9/99.9 | 100.0/99.7/99.8 | 43.2/95.0/59.4 |

### Bảng 5.6. Ma trận nhầm lẫn (hàng=GT, cột=Pred)

| | **available** | **occupied** | **overlapping** |
|---|---|---|---|
| **Rectangular ROI** |
| available | 3432 | 11 | 0 |
| occupied | 0 | 15517 | 0 |
| overlapping | 0 | 40 | 0 |
| **Polygon Only** |
| available | 3440 | 3 | 0 |
| occupied | 0 | 15517 | 0 |
| overlapping | 0 | 40 | 0 |
| **Polygon + Inner-Core** |
| available | 3440 | 0 | 3 |
| occupied | 0 | 15472 | 45 |
| overlapping | 0 | 1 | 39 |
| **Proposed Framework** |
| available | 3441 | 0 | 2 |
| occupied | 4 | 15465 | 48 |
| overlapping | 0 | 2 | 38 |

## 5.4.1 Độ trễ cập nhật trạng thái (State Update Latency)

Độ trễ được đo bằng số khung hình từ khi ground truth thay đổi trạng thái đến khi hệ thống phát hiện và cập nhật tương ứng. State machine trong hệ thống thực tế hoạt động theo nguyên tắc: xe đỗ vào slot được phát hiện ngay lập tức (không có độ trễ), trong khi xe rời khỏi slot cần đợi 45 khung hình liên tiếp không có detection mới được coi là available (hysteresis). Kết quả đánh giá trên 49 ground truth transitions cho thấy chế độ **Hysteresis + Inner-Core (ON_ON)** match được 87.8% transitions (43/49), trong khi chế độ **Hysteresis không có Inner-Core (ON_OFF)** chỉ match 57.1% (28/49). Inner-Core đóng vai trò then chốt: không có nó, hệ thống không phân biệt được overlapping và occupied nên bỏ qua nhiều transitions hợp lệ.

Xét về tốc độ phản hồi, chế độ ON_ON có độ trễ trung bình 548.4 khung hình (18.3 giây ở 30 FPS), trong khi ON_OFF lên đến 1567.3 khung hình (52.2 giây). Đáng chú ý, **median latency của cả hai chế độ đều bằng 0**, nghĩa là hơn một nửa các transitions được phát hiện ngay tại đúng khung hình ground truth — không có độ trễ thực tế. Ở ngưỡng 45 khung hình (1.5 giây), chế độ ON_ON đạt 81.4% transitions dưới ngưỡng, so với 67.9% của ON_OFF.

Phân tích chi tiết theo loại transition cho thấy tốc độ phản hồi phụ thuộc rất lớn vào hướng chuyển đổi. Transition **available → occupied** có median bằng 0 — xe đỗ vào slot được phát hiện và cập nhật ngay lập tức, không có độ trễ. **occupied → available** có median 10.5 khung hình (~0.35 giây) — gần như ngay lập tức vì khi xe rời đi, centroid ra khỏi inner-core và hệ thống chỉ cần đợi thêm một vài frames để đảm bảo xe đã thực sự rời đi. **overlapping → occupied** có độ trễ cao nhất (median 12,685 frames, ~423 giây) vì hệ thống cần xe giữ nguyên trạng thái overlapping trong đủ 45 khung hình liên tiếp mới chuyển sang occupied. Đây là trade-off có chủ đích: hysteresis ngăn chuyển quá nhanh sang occupied khi xe chỉ đỗ tạm ở rìa, giúp giảm false positive nhưng đánh đổi bằng độ trễ lớn cho overlapping kéo dài.

Nhìn tổng thể, Proposed Framework mang lại trải nghiệm ổn định: phần lớn transitions cập nhật gần như ngay lập tức, và chỉ trường hợp overlapping mới có độ trễ đáng kể. Trong môi trường thực tế, độ trễ này hoàn toàn chấp nhận được vì không gian đỗ xe không yêu cầu phản hồi sub-second, và sự ổn định (ít flickering) quan trọng hơn tốc độ tuyệt đối.

## 5.5 Hạn chế của đề tài

### 5.5.1 Phụ thuộc vào mô hình YOLO được huấn luyện sẵn

Hệ thống sử dụng YOLOv8l làm mô hình phát hiện phương tiện mà không huấn luyện lại (fine-tuning) trên dữ liệu bãi đỗ mục tiêu. Do đó, toàn bộ độ chính xác phát hiện xe — và gián tiếp là độ chính xác chiếm chỗ — phụ thuộc hoàn toàn vào khả năng nhận diện của YOLOv8l gốc. Mô hình này được huấn luyện trên tập dữ liệu COCO với 80 lớp, trong đó chỉ sử dụng 3 lớp `car`, `bus`, `truck`. Các phương tiện ngoài 3 lớp này (xe máy, xe đạp, xe ba bánh) không được hỗ trợ và có thể gây nhiễu hoặc bị bỏ qua. Ngoài ra, hệ thống không có cơ chế tự động giám sát và cập nhật mô hình khi độ chính xác suy giảm theo thời gian — việc cải thiện đòi hỏi thu thập ground truth mới, huấn luyện lại thủ công và triển khai lại mô hình.

### 5.5.2 Phụ thuộc vào chất lượng camera cổng cho nhận diện biển số

Hệ thống sử dụng FIFO (First-In-First-Out) để ghép cặp biển số từ camera cổng với vị trí đỗ trong bãi. Quá trình này phụ thuộc hoàn toàn vào khả năng OCR (PaddleOCR) đọc được biển số tại cổng. Nếu camera cổng bị suy giảm chất lượng do điều kiện thời tiết (ngược sáng, mưa, sương mù), camera bị rung, hoặc biển số bị che khuất một phần, OCR có thể không đọc được biển số. Trong trường hợp này, xe vào bãi đỗ mà không có biển số trong hệ thống FIFO → hệ thống chỉ ghi nhận trạng thái chiếm chỗ (có xe trong slot) nhưng không thể liên kết với chủ xe. Xe không có biển số cũng không được ghi nhận thời gian vào/ra chính xác tại cổng.

### 5.5.3 Chỉ thực nghiệm trong khuôn viên trường đại học

Toàn bộ dữ liệu đánh giá (ground truth, video, parking zones) được thu thập và xây dựng cho khuôn viên trường đại học — một môi trường có đặc điểm riêng về cấu trúc bãi đỗ, góc camera, loại phương tiện và mật độ xe. Hệ thống chưa được thử nghiệm tại các vị trí thực tế khác như bãi đỗ thương mại, chung cư, hay trung tâm thương mại. Việc triển khai tại một vị trí mới đòi hỏi thiết lập lại parking zones (vẽ polygon cho từng slot mới), thu thập video và ground truth cho vị trí mới, điều chỉnh tham số inner-core và hysteresis threshold theo đặc thù bãi đỗ, và đánh giá lại độ chính xác trên dữ liệu mới. Nói cách khác, hệ thống chưa có tính tổng quát hóa (generalization) — mỗi vị trí triển khai cần quy trình setup và đánh giá riêng.
