# License Plate Detection & Recognition - Complete Module

**Module hoàn chỉnh** để phát hiện và nhận diện biển số xe Việt Nam.

## 🎯 Tính năng

- ✅ **Pipeline hoàn chỉnh**: Vehicle Detection → Plate Detection → OCR
- ✅ **YOLO Detection**: YOLOv8 cho phát hiện phương tiện và biển số
- ✅ **2 phương pháp OCR**: PaddleOCR (chính xác) và ONNX (nhanh)
- ✅ **Models đã bao gồm**: Không cần tải thêm (~112MB)
- ✅ **Tiền xử lý ảnh**: Tăng cường chất lượng nhận diện
- ✅ **Validation**: Kiểm tra định dạng biển số hợp lệ
- ✅ **CLI & API**: Sử dụng qua command line hoặc Python code
- ✅ **Batch processing**: Xử lý nhiều ảnh cùng lúc

## 📦 Cài đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt
```

## 🚀 Sử dụng

### 1. Complete Pipeline (Khuyến nghị)

```bash
# Phát hiện xe + biển số + OCR (end-to-end)
python plate_detector.py path/to/vehicle_image.jpg --output result.jpg

# Với ONNX OCR (nhanh hơn)
python plate_detector.py image.jpg --ocr-method onnx --output result.jpg

# Hiển thị cả plate boxes
python plate_detector.py image.jpg --show-plate-box --output result.jpg
```

### 2. Python API - Complete Pipeline

```python
import cv2
from plate_detector import LicensePlateDetector

# Khởi tạo detector hoàn chỉnh
detector = LicensePlateDetector(
    ocr_method="paddle",       # hoặc "onnx"
    vehicle_conf=0.6,          # ngưỡng phát hiện xe
    plate_conf=0.25,           # ngưỡng phát hiện biển số
    ocr_conf=0.5               # ngưỡng OCR
)

# Đọc ảnh có xe
image = cv2.imread("vehicle.jpg")

# Xử lý hoàn chỉnh
vehicles = detector.process(image)

# Hiển thị kết quả
for vehicle in vehicles:
    print(f"Loại xe: {vehicle.vehicle_type}")
    print(f"Biển số: {vehicle.plate_text}")
    print(f"Độ tin cậy: {vehicle.plate_conf:.2%}")

# Vẽ kết quả
result = detector.draw_results(image, vehicles)
cv2.imwrite("result.jpg", result)
```

### 3. OCR Only (Ảnh biển số đã crop)

```bash
# Sử dụng PaddleOCR (mặc định)
python license_plate_ocr.py path/to/plate_image.jpg

# Sử dụng ONNX (nhanh hơn)
python license_plate_ocr.py path/to/plate_image.jpg --method onnx

# Với preprocessing và threshold tùy chỉnh
python license_plate_ocr.py image.jpg --preprocess --threshold 0.7
```

### 4. Python API - OCR Only

```python
import cv2
from license_plate_ocr import LicensePlateOCR

# Khởi tạo OCR engine
ocr = LicensePlateOCR(
    method="paddle",           # hoặc "onnx"
    conf_threshold=0.5,        # ngưỡng confidence
    use_preprocessing=False    # tiền xử lý ảnh
)

# Đọc ảnh biển số (đã crop)
image = cv2.imread("plate.jpg")

# Nhận diện
text, confidence = ocr.recognize(image, expand_ratio=0.15)

print(f"Biển số: {text}")
print(f"Độ tin cậy: {confidence:.2%}")
```

### 3. Sử dụng Utility Functions

```python
from ocr_utils import crop_expanded_plate, check_legit_plate

# Crop và mở rộng vùng biển số
plate_xyxy = [100, 50, 300, 150]  # [x_min, y_min, x_max, y_max]
cropped = crop_expanded_plate(plate_xyxy, vehicle_image, expand_ratio=0.15)

# Kiểm tra biển số hợp lệ
is_valid = check_legit_plate("29A12345")  # True
is_valid = check_legit_plate("ABC")       # False
```

## 📁 Cấu trúc thư mục

```
LicensePlate_OCR_Standalone/
├── __init__.py                 # Package initialization (v2.0.0)
├── plate_detector.py           # Complete detection pipeline ⭐ NEW
├── license_plate_ocr.py        # OCR interface (CLI + API)
├── paddle_ocr.py               # PaddleOCR implementation
├── ocr_utils.py                # Utility functions
├── models/                     # Pre-trained models (~112MB) ⭐ NEW
│   ├── vehicle_yolov8s_640.pt          # Vehicle detector (~85MB)
│   ├── plate_yolov8n_320_2024.pt       # Plate detector (~6MB)
│   ├── ch_PP-OCRv4_det_infer.onnx      # OCR detection (~5MB)
│   ├── ch_PP-OCRv4_rec_infer.onnx      # OCR recognition (~11MB)
│   └── updated_model_dyn.onnx          # Alternative OCR (~11MB)
├── ppocr_onnx/                 # ONNX OCR implementation
│   ├── __init__.py
│   ├── pipeline.py
│   ├── det/                    # Text detection
│   └── rec/                    # Text recognition
├── requirements.txt            # Dependencies
├── README.md                   # Tài liệu này
├── MODELS.md                   # Model documentation
├── examples.py                 # OCR-only examples
└── examples_complete.py        # Complete pipeline examples ⭐ NEW
```

## 🔧 API Reference

### `LicensePlateOCR`

**Constructor:**
```python
LicensePlateOCR(
    method="paddle",           # "paddle" hoặc "onnx"
    conf_threshold=0.5,        # 0.0 - 1.0
    use_preprocessing=False,   # True/False
    **kwargs                   # Tham số bổ sung cho engine
)
```

**Methods:**
- `recognize(plate_image, expand_ratio=0.0)` → `(text, confidence)`

### Utility Functions

- `crop_expanded_plate(plate_xyxy, image, expand_ratio)` - Crop và mở rộng vùng ảnh
- `check_legit_plate(text)` - Kiểm tra định dạng biển số hợp lệ
- `preprocess_plate_image(image)` - Tiền xử lý ảnh (grayscale, blur, threshold)

## 📊 So sánh phương pháp

| Tiêu chí | PaddleOCR | ONNX |
|----------|-----------|------|
| **Tốc độ** | Chậm hơn | Nhanh hơn 2-3x |
| **Độ chính xác** | Cao | Tương đương |
| **Kích thước** | ~200MB | ~50MB |
| **Khuyến nghị** | Độ chính xác quan trọng | Tốc độ quan trọng |

## 🎨 Ví dụ nâng cao

### Batch processing nhiều ảnh

```python
import cv2
import glob
from license_plate_ocr import LicensePlateOCR

ocr = LicensePlateOCR(method="paddle", conf_threshold=0.6)

for img_path in glob.glob("plates/*.jpg"):
    image = cv2.imread(img_path)
    text, conf = ocr.recognize(image)
    
    if text:
        print(f"{img_path}: {text} ({conf:.2%})")
    else:
        print(f"{img_path}: KHÔNG NHẬN DIỆN ĐƯỢC")
```

### Tích hợp với YOLO detection

```python
from ultralytics import YOLO
from license_plate_ocr import LicensePlateOCR
from ocr_utils import crop_expanded_plate

# Load models
vehicle_detector = YOLO("yolov8n.pt")
plate_detector = YOLO("plate_detector.pt")
ocr = LicensePlateOCR()

# Detect vehicles
vehicles = vehicle_detector(image)

for vehicle_box in vehicles.boxes.xyxy:
    # Crop vehicle
    vehicle_img = image[int(vehicle_box[1]):int(vehicle_box[3]), 
                       int(vehicle_box[0]):int(vehicle_box[2])]
    
    # Detect plate
    plates = plate_detector(vehicle_img)
    
    for plate_box in plates.boxes.xyxy:
        # Crop plate với mở rộng 15%
        plate_img = crop_expanded_plate(
            plate_box.cpu().numpy(), 
            vehicle_img, 
            expand_ratio=0.15
        )
        
        # OCR
        text, conf = ocr.recognize(plate_img)
        print(f"Biển số: {text} ({conf:.2%})")
```

## 🐛 Xử lý lỗi thường gặp

### Lỗi: "Cannot load image"
```python
# Kiểm tra đường dẫn file
import os
if not os.path.exists(image_path):
    print("File không tồn tại!")
```

### Lỗi: "OCR failed"
```python
# Thử với preprocessing
ocr = LicensePlateOCR(use_preprocessing=True)
text, conf = ocr.recognize(image)
```

### Confidence thấp
```python
# Giảm threshold hoặc tăng expand_ratio
ocr = LicensePlateOCR(conf_threshold=0.3)
text, conf = ocr.recognize(image, expand_ratio=0.2)
```

## 📝 Định dạng biển số hợp lệ

- **Format 1**: 2 chữ cái + 4 số (VD: `29A1234`, `AB1234`)
- **Format 2**: 1 chữ cái + 4+ số (VD: `A12345`)

## 🔗 Nguồn gốc

Module này được trích xuất từ project [AI-Traffic-Analysis](https://github.com/...) với đầy đủ chức năng OCR biển số xe.

## 📄 License

Kế thừa license từ project gốc.
