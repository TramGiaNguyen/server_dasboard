"""
OCR Comparison Test Script
Test different preprocessing methods on the same plate image
"""

import os
import sys
import cv2

# Add OCR module path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ocr_module_path = os.path.join(base_dir, 'services', 'ocr', 'LicensePlate_OCR_Standalone')
sys.path.insert(0, ocr_module_path)

from license_plate_ocr import LicensePlateOCR

def test_ocr_on_images():
    """Test OCR on all preprocessed images and compare results"""
    
    # Initialize OCR engine (ONNX method - same as production)
    print("=" * 60)
    print("Initializing OCR Engine...")
    print("=" * 60)
    
    ocr = LicensePlateOCR(
        method="onnx",
        conf_threshold=0.3,  # Lower threshold to catch more text
        use_preprocessing=False  # Disable built-in preprocessing to test raw images
    )
    
    # Test images with different preprocessing methods
    test_images = [
        ("01_up_unsharp.png", "Upscale + Unsharp Mask"),
        ("02_clahe_sharp.png", "CLAHE + Sharpen"),
        ("03_adaptive_thresh.png", "Adaptive Threshold"),
        ("04_otsu.png", "Otsu Binarization"),
        ("05_laplacian_enhance.png", "Laplacian Enhancement"),
        ("06_wiener_clahe_sharp.png", "Wiener + CLAHE + Sharpen"),
        ("07_blackhat_combo.png", "Blackhat Morphology Combo"),
    ]
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "=" * 60)
    print("OCR COMPARISON RESULTS")
    print("=" * 60)
    
    results = []
    
    for filename, method_name in test_images:
        img_path = os.path.join(test_dir, filename)
        
        if not os.path.exists(img_path):
            print(f"\n[SKIP] {filename} - File not found")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"\n[ERROR] {filename} - Failed to load image")
            continue
        
        h, w = img.shape[:2]
        
        # Run OCR
        text, confidence = ocr.recognize(img, expand_ratio=0.0)
        
        # Store result
        results.append({
            'filename': filename,
            'method': method_name,
            'text': text if text else "(No text detected)",
            'confidence': confidence,
            'size': f"{w}x{h}"
        })
        
        # Print result
        status = "✓" if text else "✗"
        print(f"\n{status} [{method_name}]")
        print(f"   File: {filename} ({w}x{h})")
        print(f"   Text: {text if text else '(No text detected)'}")
        print(f"   Confidence: {confidence:.2%}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Ranked by Confidence")
    print("=" * 60)
    
    # Sort by confidence (descending)
    results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    for i, r in enumerate(results_sorted, 1):
        status = "✓" if r['text'] != "(No text detected)" else "✗"
        print(f"{i}. {status} {r['method']}: {r['text']} ({r['confidence']:.2%})")
    
    # Best method
    if results_sorted and results_sorted[0]['confidence'] > 0:
        best = results_sorted[0]
        print(f"\n🏆 BEST METHOD: {best['method']}")
        print(f"   Result: {best['text']} (Confidence: {best['confidence']:.2%})")
    else:
        print("\n⚠️ No method successfully detected text")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_ocr_on_images()
