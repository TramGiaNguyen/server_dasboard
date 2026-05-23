"""
Runtime limits — centralized thread/process configuration.

Why this exists:
  On multi-core hosts (especially 16+ logical cores) PyTorch and ONNX Runtime
  each spawn one worker thread per logical core by default. When multiple
  inference engines run concurrently (parking YOLO + gate YOLO + 3 OCR ONNX
  sessions) the result is hundreds of OS threads fighting for the same cores,
  context switches dominate and total throughput is LOWER than on a 6-core
  host. This module enforces an explicit, hardware-independent thread budget.

Usage:
  Must be invoked BEFORE any `import torch`, `import numpy`, `import cv2` or
  `import onnxruntime`, because OMP / MKL / OpenBLAS read their thread limits
  only once at library load time.

  In main.py:
      import os
      os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
      from shared.runtime_limits import setup_runtime_limits
      setup_runtime_limits()
      # ... now safe to import torch / cv2 / etc.

  After torch is imported, also call apply_torch_limits() once to set
  per-process knobs that the env vars do not cover.
"""
import os


_DEFAULT_TORCH_THREADS = 4
_DEFAULT_ONNX_INTRA_THREADS = 2
_DEFAULT_ONNX_INTER_THREADS = 1

_applied = False


def _maybe_load_dotenv() -> None:
    """Load .env if python-dotenv is installed. Safe to call before torch import."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '.env',
        ))
    except Exception:
        pass


def setup_runtime_limits() -> dict:
    """Set OMP / MKL / BLAS thread caps. Must run before torch / numpy import.

    Returns the resolved limits dict so callers can log them.
    """
    global _applied
    _maybe_load_dotenv()

    torch_threads = int(os.getenv("TORCH_NUM_THREADS", str(_DEFAULT_TORCH_THREADS)))
    onnx_intra = int(os.getenv("ONNX_NUM_THREADS", str(_DEFAULT_ONNX_INTRA_THREADS)))
    onnx_inter = int(os.getenv("ONNX_INTER_OP_THREADS", str(_DEFAULT_ONNX_INTER_THREADS)))

    if torch_threads < 1:
        torch_threads = 1
    if onnx_intra < 1:
        onnx_intra = 1
    if onnx_inter < 1:
        onnx_inter = 1

    env_caps = {
        "OMP_NUM_THREADS": str(torch_threads),
        "MKL_NUM_THREADS": str(torch_threads),
        "OPENBLAS_NUM_THREADS": str(torch_threads),
        "NUMEXPR_NUM_THREADS": str(torch_threads),
        "VECLIB_MAXIMUM_THREADS": str(torch_threads),
    }
    for k, v in env_caps.items():
        os.environ.setdefault(k, v)

    _applied = True
    return {
        "torch_threads": torch_threads,
        "onnx_intra_op": onnx_intra,
        "onnx_inter_op": onnx_inter,
    }


def apply_torch_limits() -> None:
    """Cap torch's own intra / inter op threads. Safe to call multiple times."""
    try:
        import torch
        torch_threads = int(os.getenv("TORCH_NUM_THREADS", str(_DEFAULT_TORCH_THREADS)))
        torch.set_num_threads(max(1, torch_threads))
        torch.set_num_interop_threads(max(1, min(2, torch_threads)))
    except Exception:
        pass


def get_onnx_session_options():
    """Build a configured ort.SessionOptions with consistent thread caps.

    Returns None if onnxruntime is not installed yet.
    """
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = int(os.getenv(
            "ONNX_NUM_THREADS", str(_DEFAULT_ONNX_INTRA_THREADS),
        ))
        so.inter_op_num_threads = int(os.getenv(
            "ONNX_INTER_OP_THREADS", str(_DEFAULT_ONNX_INTER_THREADS),
        ))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return so
    except Exception:
        return None


def is_applied() -> bool:
    return _applied
