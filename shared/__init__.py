# Shared module initialization
from .state import current_parking_status, gate_ocr_results

# Note: models.py (check_gpu, initialize_model) requires torch/ultralytics
# Import directly from shared.models if needed, not exposed here to avoid
# forcing heavy dependencies on lightweight consumers (e.g., mobile backend)

__all__ = [
    'current_parking_status',
    'gate_ocr_results',
]
