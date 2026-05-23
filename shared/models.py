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


def initialize_model(model_path=None, use_half=False):
    """Initialize a YOLO model.

    Each caller gets a fresh instance because model.track() stores internal
    tracker state across calls — cameras must NOT share the same instance.
    model.fuse() merges Conv+BN layers for ~5-10% faster inference.

    Args:
        model_path: Path (or bare filename) to the YOLO weights. If None or empty,
            defaults to the global MODEL_PATH (parking model). Pass a different
            path to use a smaller model on a different camera (e.g. yolov8s.pt
            for the gate camera).
        use_half: Enable FP16 half precision for 20-40% speedup on GPU (minimal
            accuracy loss). Silently ignored on CPU.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = model_path or MODEL_PATH
    print(f"[MODEL] Loading YOLO weights: {path} (device={device}, half={use_half})")
    model = YOLO(path).to(device)
    model.fuse()

    # Enable half precision (FP16) for GPU inference speedup
    if use_half and device == 'cuda':
        model.model.half()
        print(f"[MODEL] FP16 enabled for {os.path.basename(str(path))} — expect 20-40% speedup")

    return model



