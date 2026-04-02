# Model initialization utilities
import torch
from ultralytics import YOLO
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH


def check_gpu():
    """Check for GPU availability and print GPU information"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU configuration.")
    else:
        print("CUDA is available.")
        print("Current GPU device:", torch.cuda.current_device())
        print("GPU device name:", torch.cuda.get_device_name(0))


def initialize_model(use_half=False):
    """Initialize YOLO model. Each caller gets a fresh instance because
    model.track() stores internal tracker state — cameras must NOT share.
    model.fuse() merges Conv+BN layers for faster inference (~5-10% speedup).
    
    Args:
        use_half: Enable FP16 half precision for 20-40% speedup on GPU (minimal accuracy loss)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL_PATH).to(device)
    model.fuse()
    
    # Enable half precision (FP16) for GPU inference speedup
    if use_half and device == 'cuda':
        model.model.half()
        print("[MODEL] Half precision (FP16) enabled - expect 20-40% speedup")
    
    return model



