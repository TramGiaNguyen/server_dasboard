"""
License Plate OCR - Main Interface
Standalone module for Vietnamese license plate recognition
Author: Extracted from AI-Traffic-Analysis project
"""

import argparse
import re
from functools import lru_cache
from typing import Optional, Tuple, Literal

import cv2
import numpy as np

from ocr_utils import (
    check_legit_plate, 
    crop_expanded_plate, 
    preprocess_plate_image, 
    enhanced_plate_preprocessing,
    laplacian_enhance_preprocessing,
    adaptive_thresh_preprocessing
)
from collections import Counter


class LicensePlateOCR:
    """
    Main interface for license plate OCR.
    Supports both PaddleOCR (default) and ONNX methods.
    """
    
    def __init__(
        self, 
        method: Literal["paddle", "onnx"] = "paddle",
        conf_threshold: float = 0.5,
        use_preprocessing: bool = False,
        **kwargs
    ):
        """
        Initialize OCR engine.
        
        Args:
            method: OCR method to use ("paddle" or "onnx")
            conf_threshold: Minimum confidence threshold (0.0-1.0)
            use_preprocessing: Whether to apply image preprocessing
            **kwargs: Additional arguments for OCR engine
        """
        self.method = method
        self.conf_threshold = conf_threshold
        self.use_preprocessing = use_preprocessing
        
        if method == "paddle":
            self.ocr = self._init_paddle_ocr(**kwargs)
        elif method == "onnx":
            self.ocr = self._init_onnx_ocr(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'paddle' or 'onnx'")
    
    @staticmethod
    @lru_cache(maxsize=1)
    def _init_paddle_ocr(**kwargs):
        """Initialize PaddleOCR engine (cached)."""
        # Lazy import so ONNX mode does not require paddleocr package.
        from paddleocr import PaddleOCR
        default_config = {
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_textline_orientation': False,
        }
        default_config.update(kwargs)
        return PaddleOCR(**default_config)
    
    @staticmethod
    def _init_onnx_ocr(**kwargs):
        """Initialize ONNX OCR engine."""
        from ppocr_onnx import DetAndRecONNXPipeline
        import os
        
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Default model paths (use local models if available)
        default_det = os.path.join(current_dir, "models", "PP-OCRv5_server_det_infer.onnx")
        default_rec = os.path.join(current_dir, "models", "PP-OCRv5_server_rec_infer.onnx")
        default_dict = os.path.join(current_dir, "ppocr_onnx", "ppocrv5_dict.txt")
        
        default_config = {
            'box_thresh': 0.6,
            'unclip_ratio': 1.6,
            'text_det_onnx_model': kwargs.get('det_model', default_det),
            'text_rec_onnx_model': kwargs.get('rec_model', default_rec),
            'text_rec_dict': kwargs.get('dict_path', default_dict)
        }
        return DetAndRecONNXPipeline(**default_config)
    
    def recognize(
        self, 
        plate_image: np.ndarray,
        expand_ratio: float = 0.0
    ) -> Tuple[str, float]:
        """
        Recognize license plate text from image.
        
        Args:
            plate_image: Input image (BGR format)
            expand_ratio: Ratio to expand image borders (0.0-0.3)
            
        Returns:
            Tuple of (plate_text, confidence_score)
            Returns ("", 0.0) if recognition fails or confidence is too low
        """
        if plate_image is None or not isinstance(plate_image, np.ndarray):
            return "", 0.0
        
        h, w = plate_image.shape[:2]
        original_image = plate_image.copy()
        
        # Step 1: Anti-glare handling - detect overexposed images first
        gray_check = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_check)
        bright_pixels_ratio = np.sum(gray_check > 200) / gray_check.size
        
        if mean_brightness > 180 or bright_pixels_ratio > 0.5:
            # Image is overexposed - apply gamma correction
            gamma = 2.0
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            plate_image = cv2.LUT(plate_image, table)
            
            # Boost saturation to recover color info
            hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
            plate_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Determine upscale factor based on plate size
        if h < 40:
            scale = 10  # 10x for very small plates
        elif h < 80:
            scale = 8   # 8x for small plates
        else:
            scale = 4
        
        # Step 2: Use ONLY Wiener-CLAHE-Sharp preprocessing as requested
        # Removed ensemble/voting logic to improve performance and specificity
        
        try:
            # Apply preprocessing
            preprocessed = enhanced_plate_preprocessing(plate_image.copy(), scale=scale)
            
            # Expand if needed
            if expand_ratio > 0:
                ph, pw = preprocessed.shape[:2]
                plate_xyxy = [0, 0, pw, ph]
                preprocessed = crop_expanded_plate(plate_xyxy, preprocessed, expand_ratio)
            
            # Run OCR
            if self.method == "paddle":
                text, conf = self._recognize_paddle(preprocessed)
            else:
                text, conf = self._recognize_onnx(preprocessed)
            
            return text, conf
            
        except Exception as e:
            print(f"[DEBUG OCR] Preprocessing/OCR failed: {e}")
            return "", 0.0
    
    def _recognize_paddle(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """Recognize using PaddleOCR."""
        try:
            results = self.ocr.ocr(plate_image, cls=False)
        except Exception as e:
            print(f"OCR failed: {e}")
            return "", 0.0
        
        if not results or not results[0]:
            return "", 0.0
        
        # PaddleOCR returns: [[[bbox], (text, confidence)], ...]
        # Extract texts and confidences
        rec_texts = []
        rec_scores = []
        for line in results[0]:
            if line and len(line) >= 2:
                text, score = line[1]
                rec_texts.append(text)
                rec_scores.append(score)
        
        # Combine all recognized texts
        plate_info = " ".join(rec_texts) if rec_texts else ""
        
        # Post-processing
        plate_info = self._postprocess_text(plate_info)
        
        # Calculate average confidence
        conf = float(sum(rec_scores) / len(rec_scores)) if rec_scores else 0.0
        
        # Validate result
        if conf >= self.conf_threshold and check_legit_plate(plate_info):
            return plate_info, conf
        
        return "", 0.0
    
    def _recognize_onnx(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """Recognize using ONNX."""
        try:
            results = self.ocr.detect_and_ocr(
                plate_image, 
                drop_score=self.conf_threshold
            )
        except Exception as e:
            print(f"ONNX OCR failed: {e}")
            return "", 0.0
        
        if not results:
            # print(f"[DEBUG OCR] No text detected in plate image")
            return "", 0.0
        
        # Vietnamese plates have 2 rows (e.g., "60A" on top, "62602" on bottom)
        # Combine ALL detected text lines instead of just taking the best one
        all_texts = []
        all_scores = []
        for r in results:
            all_texts.append(r.text)
            all_scores.append(r.score)
        
        # Sort by Y position (top to bottom) - boxes are sorted in pipeline.py
        # Combine all texts into one plate string
        combined_text = "".join(all_texts)
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        plate_info = self._postprocess_text(combined_text)
        
        # DEBUG: Log raw OCR result before validation
        print(f"[DEBUG OCR] Raw texts: {all_texts} -> Combined: '{plate_info}' (avg score: {avg_score:.2f})")
        
        if check_legit_plate(plate_info):
            return plate_info, avg_score
        else:
            pass  # Silently reject invalid plates
            # print(f"[DEBUG OCR] Rejected by check_legit_plate: '{plate_info}'")
        
        return "", 0.0
    
    @staticmethod
    def _postprocess_text(text: str) -> str:
        """
        Post-process OCR text with Vietnamese license plate grammar validation.
        
        Vietnamese plate formats:
        - Standard: NN + [A-Z except I,O,Q,W] + 5 digits (e.g., 61K-345.64 -> 61K34564)
        - Special series: NNKT, NNLD, NNDA, NNR, NNHC + 5 digits
        
        OCR Confusion Mapping based on common misreads.
        """
        if not text:
            return ""
        
        # Step 1: Normalize - remove all non-alphanumeric, uppercase
        text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        
        if len(text) < 5:
            return text
        
        # Vietnamese plate valid series letters (I, O, Q, W are NOT used)
        VALID_SERIES_LETTERS = set('ABCDEFGHKLMNPSTUVXYZ')
        
        # Special 2-letter series
        SPECIAL_SERIES = {'KT', 'LD', 'DA', 'HC', 'NG', 'QT', 'CV', 'NN', 'LB', 'R'}
        
        # OCR confusion maps based on common misreads
        # Digit -> Letter (for series position - position 3)
        DIGIT_TO_LETTER = {
            '0': 'D',   # or K/C
            '1': 'L',   # or T
            '2': 'Z',
            '4': 'A',
            '5': 'S',
            '6': 'G',   # or L/C
            '7': 'T',   # or Y
            '8': 'B',
            '9': 'G',
        }
        
        # Letter -> Digit (for numeric positions - positions 4-8)
        LETTER_TO_DIGIT = {
            'A': '4',
            'B': '8',
            'C': '0',
            'D': '0',
            'G': '6',
            'I': '1',
            'J': '3',   # common misread
            'L': '6',   # common misread (user's case: L4 -> 64)
            'O': '0',
            'Q': '0',
            'R': '2',   # can look like 2
            'S': '5',
            'T': '7',
            'U': '0',   # can look like 0
            'Y': '4',   # can look like 4
            'Z': '2',
        }
        
        # --- Parse and correct based on grammar ---
        corrected_text = text
        
        # Province code (first 2 chars should be digits 11-99)
        first_two = text[:2]
        corrected_province = ""
        for char in first_two:
            if char.isdigit():
                corrected_province += char
            elif char in LETTER_TO_DIGIT:
                corrected_province += LETTER_TO_DIGIT[char]
            else:
                corrected_province += char
        
        # Check remaining chars
        remaining = text[2:]
        
        if not remaining:
            return corrected_province
        
        # Check for 2-letter special series (KT, LD, DA, HC, etc.)
        series = ""
        number_part = ""
        
        if len(remaining) >= 2:
            potential_series = remaining[:2].upper()
            if potential_series in SPECIAL_SERIES:
                # Format: NN + 2-letter series + 5 digits
                series = potential_series
                number_part = remaining[2:]
            else:
                # Format: NN + 1-letter series + 5 digits (standard)
                series_char = remaining[0]
                
                # If series char is a digit, correct to letter
                if series_char.isdigit():
                    series_char = DIGIT_TO_LETTER.get(series_char, series_char)
                
                # Validate series letter
                if series_char not in VALID_SERIES_LETTERS:
                    # Try to map invalid letters
                    invalid_to_valid = {'I': 'L', 'O': 'D', 'Q': 'G', 'W': 'M'}
                    series_char = invalid_to_valid.get(series_char, series_char)
                
                series = series_char
                number_part = remaining[1:]
        else:
            series = remaining[0] if remaining else ""
            if series.isdigit():
                series = DIGIT_TO_LETTER.get(series, series)
            number_part = remaining[1:] if len(remaining) > 1 else ""
        
        # Correct the number part (should be all digits)
        corrected_numbers = ""
        for char in number_part:
            if char.isdigit():
                corrected_numbers += char
            elif char in LETTER_TO_DIGIT:
                corrected_numbers += LETTER_TO_DIGIT[char]
            else:
                # Unknown char, try to keep if looks like digit
                corrected_numbers += char
        
        # Reconstruct the plate
        corrected_text = corrected_province + series + corrected_numbers
        
        # Validate length: standard plate = 8 chars (NN + L + 5 digits)
        # Truncate if too long (likely OCR noise)
        if len(corrected_text) > 10:
            corrected_text = corrected_text[:10]
        
        return corrected_text


def main():
    """Command-line interface for testing."""
    parser = argparse.ArgumentParser(
        description="License Plate OCR - Standalone Module"
    )
    parser.add_argument(
        "image", 
        help="Path to license plate image"
    )
    parser.add_argument(
        "--method", 
        choices=["paddle", "onnx"], 
        default="paddle",
        help="OCR method to use (default: paddle)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Apply image preprocessing"
    )
    parser.add_argument(
        "--expand", 
        type=float, 
        default=0.0,
        help="Expand image ratio (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot load image from {args.image}")
        return
    
    # Initialize OCR
    print(f"Initializing {args.method.upper()} OCR engine...")
    ocr = LicensePlateOCR(
        method=args.method,
        conf_threshold=args.threshold,
        use_preprocessing=args.preprocess
    )
    
    # Recognize
    print("Processing...")
    text, confidence = ocr.recognize(image, expand_ratio=args.expand)
    
    # Display results
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"License Plate: {text if text else 'NOT DETECTED'}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Valid Format: {check_legit_plate(text) if text else 'N/A'}")
    print("="*50)


if __name__ == "__main__":
    main()
