# Models Information

## 📦 ONNX Models Included

Module này đã bao gồm sẵn các ONNX models cho OCR biển số xe:

### PaddleOCR v4 Models

Located in `models/` directory:

| Model File | Size | Description |
|------------|------|-------------|
| `ch_PP-OCRv4_det_infer.onnx` | ~4.7 MB | Text Detection Model |
| `ch_PP-OCRv4_rec_infer.onnx` | ~10.8 MB | Text Recognition Model |
| `updated_model_dyn.onnx` | ~10.8 MB | Alternative Recognition Model |

### Model Usage

#### 1. ONNX Method (Recommended for Speed)

```python
from license_plate_ocr import LicensePlateOCR
import os

# Get model paths
model_dir = "models"
det_model = os.path.join(model_dir, "ch_PP-OCRv4_det_infer.onnx")
rec_model = os.path.join(model_dir, "ch_PP-OCRv4_rec_infer.onnx")

# Initialize with ONNX models
ocr = LicensePlateOCR(
    method="onnx",
    det_model=det_model,
    rec_model=rec_model,
    conf_threshold=0.5
)

# Use it
text, conf = ocr.recognize(image)
```

#### 2. PaddleOCR Method (Auto-download models)

```python
# PaddleOCR will automatically download models on first run
ocr = LicensePlateOCR(method="paddle", conf_threshold=0.5)
text, conf = ocr.recognize(image)
```

**Note**: PaddleOCR sẽ tự động tải models (~200MB) vào thư mục cache khi chạy lần đầu tiên.

## 🔧 Model Configuration

### ONNX Pipeline Configuration

```python
from ppocr_onnx import DetAndRecONNXPipeline

pipeline = DetAndRecONNXPipeline(
    box_thresh=0.6,                    # Detection confidence threshold
    unclip_ratio=1.6,                  # Text box expansion ratio
    text_det_onnx_model="models/ch_PP-OCRv4_det_infer.onnx",
    text_rec_onnx_model="models/ch_PP-OCRv4_rec_infer.onnx",
    text_rec_dict="ppocr_onnx/ppocr_keys_v1.txt"
)

results = pipeline.detect_and_ocr(image, drop_score=0.5)
```

## 📊 Model Performance

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| **ONNX** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 💾 Low | Real-time processing |
| **PaddleOCR** | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | 💾💾 Medium | High accuracy needed |

## 🚀 Quick Start with Models

### Example 1: Using ONNX Models

```python
import cv2
from license_plate_ocr import LicensePlateOCR

# Initialize with local ONNX models
ocr = LicensePlateOCR(
    method="onnx",
    det_model="models/ch_PP-OCRv4_det_infer.onnx",
    rec_model="models/ch_PP-OCRv4_rec_infer.onnx",
    conf_threshold=0.5
)

# Process image
image = cv2.imread("plate.jpg")
text, confidence = ocr.recognize(image)

print(f"License Plate: {text}")
print(f"Confidence: {confidence:.2%}")
```

### Example 2: Batch Processing with ONNX

```python
import cv2
import glob
from license_plate_ocr import LicensePlateOCR

# Initialize once
ocr = LicensePlateOCR(
    method="onnx",
    det_model="models/ch_PP-OCRv4_det_infer.onnx",
    rec_model="models/ch_PP-OCRv4_rec_infer.onnx"
)

# Process multiple images
for img_path in glob.glob("plates/*.jpg"):
    image = cv2.imread(img_path)
    text, conf = ocr.recognize(image)
    print(f"{img_path}: {text} ({conf:.2%})")
```

## 📝 Model Details

### Detection Model (ch_PP-OCRv4_det_infer.onnx)
- **Purpose**: Detect text regions in the image
- **Input**: RGB image (any size)
- **Output**: Bounding boxes of text regions
- **Architecture**: Based on DB (Differentiable Binarization)

### Recognition Model (ch_PP-OCRv4_rec_infer.onnx)
- **Purpose**: Recognize characters in detected text regions
- **Input**: Cropped text region images
- **Output**: Text string + confidence score
- **Architecture**: Based on CRNN (Convolutional Recurrent Neural Network)
- **Character Set**: Vietnamese + English alphanumeric

## 🔍 Troubleshooting

### Issue: "Cannot find model file"
```python
import os
# Check if models exist
model_path = "models/ch_PP-OCRv4_det_infer.onnx"
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
```

### Issue: "ONNX Runtime error"
```bash
# Make sure onnxruntime is installed
pip install onnxruntime
```

### Issue: Low accuracy
```python
# Try adjusting parameters
ocr = LicensePlateOCR(
    method="onnx",
    det_model="models/ch_PP-OCRv4_det_infer.onnx",
    rec_model="models/ch_PP-OCRv4_rec_infer.onnx",
    conf_threshold=0.3,  # Lower threshold
    use_preprocessing=True  # Enable preprocessing
)
```

## 📦 Model Files Size

Total size: **~26.4 MB**

This makes the module lightweight and easy to distribute!
