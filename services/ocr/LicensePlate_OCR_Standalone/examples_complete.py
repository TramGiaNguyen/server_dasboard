"""
Complete End-to-End Examples
Vehicle Detection → Plate Detection → OCR
"""

import cv2
from plate_detector import LicensePlateDetector, VehicleInfo
from ocr_utils import check_legit_plate


def example_complete_pipeline():
    """Example: Complete end-to-end detection pipeline."""
    print("="*70)
    print("EXAMPLE: Complete Pipeline (Vehicle → Plate → OCR)")
    print("="*70)
    
    # Initialize complete detector
    detector = LicensePlateDetector(
        ocr_method="paddle",
        vehicle_conf=0.6,
        plate_conf=0.25,
        ocr_conf=0.5
    )
    
    # Load test image
    image_path = "test_vehicle.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"\n⚠ Error: Cannot load {image_path}")
        print("Please provide an image with vehicles.")
        return
    
    # Process image
    print("\nProcessing image...")
    vehicles = detector.process(image)
    
    # Display results
    print(f"\n✓ Found {len(vehicles)} vehicle(s):\n")
    for idx, vehicle in enumerate(vehicles, 1):
        print(f"Vehicle #{idx}:")
        print(f"  └─ Type: {vehicle.vehicle_type}")
        print(f"  └─ Confidence: {vehicle.vehicle_conf:.2%}")
        
        if vehicle.plate_text:
            print(f"  └─ License Plate: {vehicle.plate_text}")
            print(f"  └─ OCR Confidence: {vehicle.plate_conf:.2%}")
            print(f"  └─ Valid: {check_legit_plate(vehicle.plate_text)}")
        else:
            print(f"  └─ License Plate: NOT DETECTED")
        print()
    
    # Draw and save results
    result = detector.draw_results(image, vehicles)
    cv2.imwrite("result_complete.jpg", result)
    print("✓ Result saved to: result_complete.jpg\n")


def example_step_by_step():
    """Example: Step-by-step processing."""
    print("="*70)
    print("EXAMPLE: Step-by-Step Processing")
    print("="*70)
    
    detector = LicensePlateDetector(ocr_method="paddle")
    
    image_path = "test_vehicle.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"\n⚠ Error: Cannot load {image_path}")
        return
    
    # Step 1: Detect vehicles
    print("\nStep 1: Detecting vehicles...")
    vehicles = detector.detect_vehicles(image)
    print(f"  ✓ Found {len(vehicles)} vehicle(s)")
    
    if not vehicles:
        return
    
    # Step 2: Detect plates
    print("\nStep 2: Detecting license plates...")
    detector.detect_plates(vehicles)
    plates_found = sum(1 for v in vehicles if v.plate_image is not None)
    print(f"  ✓ Found {plates_found} plate(s)")
    
    # Step 3: Recognize text
    print("\nStep 3: Recognizing text...")
    detector.recognize_plates(vehicles)
    recognized = sum(1 for v in vehicles if v.plate_text)
    print(f"  ✓ Recognized {recognized} plate(s)")
    
    # Show details
    print("\nDetailed Results:")
    for idx, vehicle in enumerate(vehicles, 1):
        print(f"\n  Vehicle #{idx}: {vehicle.vehicle_type}")
        if vehicle.plate_text:
            print(f"    → Plate: {vehicle.plate_text} ({vehicle.plate_conf:.2%})")


def example_only_ocr():
    """Example: Using only OCR (no detection)."""
    print("="*70)
    print("EXAMPLE: OCR Only (Pre-cropped Plate Image)")
    print("="*70)
    
    from license_plate_ocr import LicensePlateOCR
    
    # Initialize OCR only
    ocr = LicensePlateOCR(method="paddle", conf_threshold=0.5)
    
    # Load pre-cropped plate image
    plate_image = cv2.imread("test_plate.jpg")
    
    if plate_image is None:
        print("\n⚠ Error: Cannot load test_plate.jpg")
        print("Please provide a cropped license plate image.")
        return
    
    # Recognize
    text, conf = ocr.recognize(plate_image)
    
    print(f"\nLicense Plate: {text if text else 'NOT DETECTED'}")
    print(f"Confidence: {conf:.2%}")
    print(f"Valid Format: {check_legit_plate(text) if text else 'N/A'}")


def example_batch_processing():
    """Example: Process multiple images."""
    print("="*70)
    print("EXAMPLE: Batch Processing Multiple Images")
    print("="*70)
    
    import glob
    
    detector = LicensePlateDetector(ocr_method="paddle")
    
    image_paths = glob.glob("test_images/*.jpg")
    
    if not image_paths:
        print("\n⚠ No images found in test_images/ folder")
        print("Create the folder and add vehicle images.")
        return
    
    print(f"\nProcessing {len(image_paths)} images...\n")
    
    all_results = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is not None:
            vehicles = detector.process(image)
            all_results.append((img_path, vehicles))
    
    # Summary
    print("="*70)
    print("BATCH RESULTS")
    print("="*70)
    
    for img_path, vehicles in all_results:
        print(f"\n{img_path}:")
        if vehicles:
            for idx, v in enumerate(vehicles, 1):
                plate_info = v.plate_text if v.plate_text else "NO PLATE"
                print(f"  Vehicle #{idx}: {v.vehicle_type} | {plate_info}")
        else:
            print("  No vehicles detected")


def example_custom_config():
    """Example: Custom configuration."""
    print("="*70)
    print("EXAMPLE: Custom Configuration")
    print("="*70)
    
    # Custom detector with specific settings
    detector = LicensePlateDetector(
        ocr_method="onnx",           # Use ONNX for speed
        vehicle_conf=0.7,            # Higher vehicle confidence
        plate_conf=0.3,              # Lower plate confidence (more sensitive)
        ocr_conf=0.6,                # Medium OCR confidence
        device="cpu"                 # Force CPU
    )
    
    print("\nConfiguration:")
    print(f"  OCR Method: onnx")
    print(f"  Vehicle Confidence: 0.7")
    print(f"  Plate Confidence: 0.3")
    print(f"  OCR Confidence: 0.6")
    print(f"  Device: cpu")
    
    image_path = "test_vehicle.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"\n⚠ Error: Cannot load {image_path}")
        return
    
    vehicles = detector.process(image)
    print(f"\n✓ Detected {len(vehicles)} vehicle(s)")


def example_visualize_all():
    """Example: Visualize with all boxes."""
    print("="*70)
    print("EXAMPLE: Visualization with All Boxes")
    print("="*70)
    
    detector = LicensePlateDetector()
    
    image_path = "test_vehicle.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"\n⚠ Error: Cannot load {image_path}")
        return
    
    vehicles = detector.process(image)
    
    # Draw with both vehicle and plate boxes
    result = detector.draw_results(
        image,
        vehicles,
        show_vehicle_box=True,
        show_plate_box=True  # Show plate boxes too
    )
    
    cv2.imwrite("result_with_boxes.jpg", result)
    print("\n✓ Result saved to: result_with_boxes.jpg")
    print("  (Green boxes = vehicles, Blue boxes = plates)")


if __name__ == "__main__":
    print("\n" + "🚗 COMPLETE DETECTION PIPELINE - EXAMPLES 🚗".center(70))
    print()
    
    # Run examples
    try:
        example_complete_pipeline()
    except Exception as e:
        print(f"Example 1 failed: {e}\n")
    
    try:
        example_step_by_step()
    except Exception as e:
        print(f"Example 2 failed: {e}\n")
    
    try:
        example_only_ocr()
    except Exception as e:
        print(f"Example 3 failed: {e}\n")
    
    try:
        example_batch_processing()
    except Exception as e:
        print(f"Example 4 failed: {e}\n")
    
    try:
        example_custom_config()
    except Exception as e:
        print(f"Example 5 failed: {e}\n")
    
    try:
        example_visualize_all()
    except Exception as e:
        print(f"Example 6 failed: {e}\n")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
