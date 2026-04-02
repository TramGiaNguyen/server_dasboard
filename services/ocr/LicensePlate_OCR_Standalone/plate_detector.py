"""
License Plate Detection Module
Complete pipeline: Vehicle Detection → Plate Detection → OCR
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import os

from ocr_utils import crop_expanded_plate, check_legit_plate
from license_plate_ocr import LicensePlateOCR


class VehicleInfo:
    """Data class for vehicle detection results."""
    def __init__(self):
        self.bbox = None  # [x1, y1, x2, y2]
        self.vehicle_type = ""
        self.vehicle_conf = 0.0
        self.vehicle_image = None
        self.plate_bbox = None  # [x1, y1, x2, y2] relative to vehicle image
        self.plate_image = None
        self.plate_text = ""
        self.plate_conf = 0.0
        self.track_id = None   # set by gate/parking camera for per-track caching



class LicensePlateDetector:
    """
    Complete license plate detection and recognition pipeline.
    
    Pipeline:
    1. Detect vehicles in image (YOLO)
    2. Detect license plates in each vehicle (YOLO)
    3. Recognize plate text (OCR)
    """
    
    def __init__(
        self,
        vehicle_model: Optional[str] = None,
        plate_model: Optional[str] = None,
        ocr_method: str = "paddle",
        vehicle_conf: float = 0.6,
        plate_conf: float = 0.25,
        ocr_conf: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize complete detection pipeline.
        
        Args:
            vehicle_model: Path to vehicle detection YOLO model, or None to skip (use external detection)
            plate_model: Path to plate detection YOLO model
            ocr_method: OCR method ("paddle" or "onnx")
            vehicle_conf: Vehicle detection confidence threshold
            plate_conf: Plate detection confidence threshold
            ocr_conf: OCR confidence threshold
            device: Device to use ("auto", "cpu", "cuda", "0", "1", etc.)
        """
        # Get default model paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_vehicle = os.path.join(current_dir, "models", "vehicle_yolov8s_640.pt")
        default_plate = os.path.join(current_dir, "models", "best_plate_yolov8.pt")

        # vehicle_model=None → skip loading (caller uses external vehicle detection, e.g. shared yolov8l)
        if vehicle_model is None:
            self.vehicle_detector = None
            print("Vehicle detector: skipped (external detection will be used)")
        else:
            vehicle_model = vehicle_model or default_vehicle
            print(f"Loading vehicle detector: {os.path.basename(vehicle_model)}")
            self.vehicle_detector = YOLO(vehicle_model, task='detect')

        plate_model = plate_model or default_plate
        
        # Auto-detect device
        if device == "auto":
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        
        # Initialize plate detector and OCR (always needed for plate detection + recognition)
        print(f"Loading plate detector: {os.path.basename(plate_model)}")
        self.plate_detector = YOLO(plate_model, task='detect')
        
        print(f"Loading OCR engine: {ocr_method}")
        self.ocr = LicensePlateOCR(method=ocr_method, conf_threshold=ocr_conf)
        
        # Configuration
        self.vehicle_conf = vehicle_conf
        self.plate_conf = plate_conf
        self.device = device
        
        # Vehicle types
        self.vehicle_types = ['bus', 'car', 'motorcycle', 'truck', 'bicycle']
        
        print(f"✓ Pipeline ready! Device: {device}")
    
    def detect_vehicles(self, image: np.ndarray, filter_classes: Optional[List[int]] = None) -> List[VehicleInfo]:
        """
        Detect vehicles in image.
        
        Args:
            image: Input image (BGR)
            filter_classes: List of class indices to filter (e.g., [1, 0, 3] for car, bus, truck)
                          If None, detects all vehicle types
            
        Returns:
            List of VehicleInfo objects
        """
        if self.vehicle_detector is None:
            return []
        # Prepare predict arguments
        predict_args = {
            'verbose': False,
            'device': self.device,
            'imgsz': 640,
            'conf': self.vehicle_conf
        }
        
        # Add class filter if specified (reduces post-processing and improves speed)
        if filter_classes is not None:
            predict_args['classes'] = filter_classes
        
        results = self.vehicle_detector(image, **predict_args)[0]
        
        vehicles = []
        boxes = results.boxes
        
        if len(boxes) == 0:
            return vehicles
        
        for idx, bbox in enumerate(boxes.xyxy):
            bbox = bbox.cpu().numpy().astype(int)
            
            vehicle = VehicleInfo()
            vehicle.bbox = bbox
            vehicle.vehicle_conf = float(boxes.conf[idx])
            
            # Get vehicle type
            cls_idx = int(boxes.cls[idx])
            if cls_idx < len(self.vehicle_types):
                vehicle.vehicle_type = self.vehicle_types[cls_idx]
            else:
                vehicle.vehicle_type = f"vehicle_{cls_idx}"
            
            # Crop vehicle image
            vehicle.vehicle_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            
            vehicles.append(vehicle)
        
        return vehicles
    
    def detect_plates(self, vehicles: List[VehicleInfo]) -> None:
        """
        Detect license plates in vehicle images.
        
        Args:
            vehicles: List of VehicleInfo objects (modified in-place)
        """
        if not vehicles:
            return
        
        # Batch process all vehicle images
        vehicle_images = [v.vehicle_image for v in vehicles]
        
        results = self.plate_detector(
            vehicle_images,
            verbose=False,
            imgsz=320,
            device=self.device,
            conf=self.plate_conf
        )
        
        for idx, vehicle in enumerate(vehicles):
            boxes = results[idx].boxes
            
            if len(boxes.xyxy) > 0:
                # Get first plate detection
                plate_bbox = boxes.xyxy[0].cpu().numpy().astype(int)
                vehicle.plate_bbox = plate_bbox
                
                # Crop and expand plate region (50% expansion for better OCR)
                vehicle.plate_image = crop_expanded_plate(
                    plate_bbox,
                    vehicle.vehicle_image,
                    expand_ratio=0.50
                )
    
    def recognize_plates(self, vehicles: List[VehicleInfo]) -> None:
        """
        Recognize text on license plates.
        
        Args:
            vehicles: List of VehicleInfo objects (modified in-place)
        """
        for vehicle in vehicles:
            if vehicle.plate_image is not None:
                text, conf = self.ocr.recognize(vehicle.plate_image)
                vehicle.plate_text = text
                vehicle.plate_conf = conf
    
    def process(self, image: np.ndarray) -> List[VehicleInfo]:
        """
        Complete pipeline: detect vehicles → detect plates → recognize text.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of VehicleInfo objects with all information filled
        """
        # Step 1: Detect vehicles
        vehicles = self.detect_vehicles(image)
        
        if not vehicles:
            return []
        
        # Step 2: Detect plates
        self.detect_plates(vehicles)
        
        # Step 3: Recognize text
        self.recognize_plates(vehicles)
        
        return vehicles
    
    def draw_results(
        self,
        image: np.ndarray,
        vehicles: List[VehicleInfo],
        show_vehicle_box: bool = True,
        show_plate_box: bool = False
    ) -> np.ndarray:
        """
        Draw detection results on image.
        
        Args:
            image: Input image (BGR)
            vehicles: List of VehicleInfo objects
            show_vehicle_box: Whether to draw vehicle bounding boxes
            show_plate_box: Whether to draw plate bounding boxes
            
        Returns:
            Image with drawn results
        """
        result = image.copy()
        
        for vehicle in vehicles:
            # Draw vehicle box
            if show_vehicle_box and vehicle.bbox is not None:
                x1, y1, x2, y2 = vehicle.bbox
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare label
                label = vehicle.vehicle_type
                if vehicle.plate_text and check_legit_plate(vehicle.plate_text):
                    label += f" | {vehicle.plate_text}"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    result,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width + 10, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    result,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            
            # Draw plate box (relative to vehicle)
            if show_plate_box and vehicle.plate_bbox is not None and vehicle.bbox is not None:
                vx1, vy1, _, _ = vehicle.bbox
                px1, py1, px2, py2 = vehicle.plate_bbox
                
                # Convert to absolute coordinates
                abs_px1 = vx1 + px1
                abs_py1 = vy1 + py1
                abs_px2 = vx1 + px2
                abs_py2 = vy1 + py2
                
                cv2.rectangle(
                    result,
                    (abs_px1, abs_py1),
                    (abs_px2, abs_py2),
                    (255, 0, 0),
                    2
                )
        
        return result


def main():
    """CLI for testing complete pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete License Plate Detection & Recognition Pipeline"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", help="Path to save output image")
    parser.add_argument("--ocr-method", choices=["paddle", "onnx"], default="paddle")
    parser.add_argument("--show-plate-box", action="store_true", help="Show plate boxes")
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot load image from {args.image}")
        return
    
    # Initialize detector
    print("\n" + "="*60)
    print("INITIALIZING DETECTION PIPELINE")
    print("="*60)
    detector = LicensePlateDetector(ocr_method=args.ocr_method)
    
    # Process
    print("\n" + "="*60)
    print("PROCESSING IMAGE")
    print("="*60)
    vehicles = detector.process(image)
    
    # Display results
    print("\n" + "="*60)
    print(f"RESULTS: Found {len(vehicles)} vehicle(s)")
    print("="*60)
    
    for idx, vehicle in enumerate(vehicles, 1):
        print(f"\nVehicle #{idx}:")
        print(f"  Type: {vehicle.vehicle_type} ({vehicle.vehicle_conf:.2%})")
        if vehicle.plate_text:
            print(f"  License Plate: {vehicle.plate_text} ({vehicle.plate_conf:.2%})")
            print(f"  Valid Format: {check_legit_plate(vehicle.plate_text)}")
        else:
            print(f"  License Plate: NOT DETECTED")
    
    # Draw and save/show
    result = detector.draw_results(
        image,
        vehicles,
        show_plate_box=args.show_plate_box
    )
    
    if args.output:
        cv2.imwrite(args.output, result)
        print(f"\n✓ Output saved to: {args.output}")
    else:
        cv2.imshow("Detection Results", result)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
