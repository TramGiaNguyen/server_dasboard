"""
Example: Basic usage of License Plate OCR
"""

import cv2
from license_plate_ocr import LicensePlateOCR
from ocr_utils import check_legit_plate

def example_basic():
    """Basic OCR example."""
    print("="*60)
    print("EXAMPLE 1: Basic OCR")
    print("="*60)
    
    # Initialize OCR
    ocr = LicensePlateOCR(
        method="paddle",
        conf_threshold=0.5
    )
    
    # Load image (replace with your image path)
    image_path = "test_plate.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        print("Please provide a valid license plate image.")
        return
    
    # Recognize
    text, confidence = ocr.recognize(image)
    
    # Display results
    print(f"\nBiển số: {text if text else 'KHÔNG NHẬN DIỆN ĐƯỢC'}")
    print(f"Độ tin cậy: {confidence:.2%}")
    print(f"Hợp lệ: {check_legit_plate(text) if text else 'N/A'}")


def example_with_preprocessing():
    """OCR with image preprocessing."""
    print("\n" + "="*60)
    print("EXAMPLE 2: OCR with Preprocessing")
    print("="*60)
    
    # Initialize with preprocessing enabled
    ocr = LicensePlateOCR(
        method="paddle",
        conf_threshold=0.5,
        use_preprocessing=True  # Enable preprocessing
    )
    
    image_path = "test_plate.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    
    # Recognize with expanded crop
    text, confidence = ocr.recognize(image, expand_ratio=0.15)
    
    print(f"\nBiển số: {text if text else 'KHÔNG NHẬN DIỆN ĐƯỢC'}")
    print(f"Độ tin cậy: {confidence:.2%}")


def example_compare_methods():
    """Compare PaddleOCR vs ONNX."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Compare PaddleOCR vs ONNX")
    print("="*60)
    
    image_path = "test_plate.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    
    # Test PaddleOCR
    print("\n[PaddleOCR]")
    ocr_paddle = LicensePlateOCR(method="paddle", conf_threshold=0.5)
    text1, conf1 = ocr_paddle.recognize(image)
    print(f"Kết quả: {text1} ({conf1:.2%})")
    
    # Test ONNX (requires ONNX models)
    print("\n[ONNX]")
    try:
        ocr_onnx = LicensePlateOCR(method="onnx", conf_threshold=0.5)
        text2, conf2 = ocr_onnx.recognize(image)
        print(f"Kết quả: {text2} ({conf2:.2%})")
    except Exception as e:
        print(f"ONNX không khả dụng: {e}")


def example_batch_processing():
    """Process multiple images."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing")
    print("="*60)
    
    import glob
    
    # Initialize OCR once
    ocr = LicensePlateOCR(method="paddle", conf_threshold=0.6)
    
    # Process all images in a folder
    image_paths = glob.glob("test_images/*.jpg")
    
    if not image_paths:
        print("No images found in test_images/ folder")
        print("Create the folder and add some license plate images.")
        return
    
    results = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is not None:
            text, conf = ocr.recognize(image)
            results.append((img_path, text, conf))
    
    # Display results
    print(f"\nProcessed {len(results)} images:\n")
    for path, text, conf in results:
        status = "✓" if text else "✗"
        print(f"{status} {path}: {text if text else 'FAILED'} ({conf:.2%})")


if __name__ == "__main__":
    print("\n" + "🚗 LICENSE PLATE OCR - EXAMPLES 🚗".center(60))
    print()
    
    # Run examples
    example_basic()
    example_with_preprocessing()
    example_compare_methods()
    example_batch_processing()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
