"""
LicensePlate OCR Standalone Module
Initialization file
"""

from .license_plate_ocr import LicensePlateOCR
from .plate_detector import LicensePlateDetector, VehicleInfo
from .ocr_utils import (
    crop_expanded_plate,
    check_legit_plate,
    preprocess_plate_image
)

__version__ = "2.0.0"
__all__ = [
    "LicensePlateOCR",
    "LicensePlateDetector",
    "VehicleInfo",
    "crop_expanded_plate",
    "check_legit_plate",
    "preprocess_plate_image"
]
