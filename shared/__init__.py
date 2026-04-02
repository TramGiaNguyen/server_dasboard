# Shared module initialization
from .state import current_parking_status, gate_ocr_results
from .models import check_gpu, initialize_model

__all__ = [
    'current_parking_status',
    'gate_ocr_results',
    'check_gpu',
    'initialize_model'
]
