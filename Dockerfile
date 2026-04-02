# =============================================================================
# Dockerfile for Smart Parking Main Application
# =============================================================================
# Multi-stage build for smaller production image
#
# Stages:
#   1. builder  - Install all dependencies including build tools
#   2. runtime  - Minimal image with only runtime dependencies
# =============================================================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BUILD_DEPS="build-essential libgl1 libglib2.0-0"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends $BUILD_DEPS \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install PyTorch with CUDA support FIRST (from PyTorch repo, not PyPI)
# Using CUDA 12.4 as specified in requirements.txt
RUN pip install --upgrade pip \
    && pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Copy requirements (remove torch/torchvision from it)
COPY requirements.txt .

# Remove torch/torchvision/onnxruntime-gpu lines to avoid re-install
RUN grep -v -E "^(torch|torchvision|onnxruntime-gpu)" requirements.txt > /tmp/requirements_filtered.txt \
    && pip install --target=/app/.deps -r /tmp/requirements_filtered.txt

# Download YOLO model during build (optional - can also mount at runtime)
# RUN pip install ultralytics && python -c "from ultralytics import YOLO; YOLO('yolov8l.pt')"


# Stage 2: Runtime (for CPU)
FROM python:3.11-slim AS runtime-cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # OpenCV headless mode
    OPENCV_VIDEOIO_PRIORITY_MSMF=0 \
    OPENCV_VIDEOIO_PRIORITY_V4L2=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /app/.deps /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . .

# Ensure static directories exist
RUN mkdir -p static/models static/sound static/video && \
    touch static/models/.gitkeep static/sound/.gitkeep static/video/.gitkeep

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Expose port
EXPOSE 5000

# Default command - can be overridden in docker-compose
CMD ["python", "main.py"]


# Stage 2: Runtime (for GPU/CUDA) - extends runtime-cpu with CUDA libraries
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS runtime-gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_VIDEOIO_PRIORITY_MSMF=0 \
    OPENCV_VIDEOIO_PRIORITY_V4L2=1 \
    # NVIDIA environment
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python from builder image
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy installed Python packages from builder
COPY --from=builder /app/.deps /usr/local/lib/python3.11/site-packages

# Create app directory
RUN mkdir -p /app && \
    groupadd -r appgroup && useradd -r -g appgroup appuser && \
    chown -R appuser:appgroup /app

# Copy application code
COPY --from=builder /app /app

# Ensure static directories exist
RUN mkdir -p static/models static/sound static/video && \
    touch static/models/.gitkeep static/sound/.gitkeep static/video/.gitkeep

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Expose port
EXPOSE 5000

# Default command - can be overridden in docker-compose
CMD ["python", "main.py"]
