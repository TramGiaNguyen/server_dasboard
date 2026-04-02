# =============================================================================
# Smart Parking System - Main Application Dockerfile
# =============================================================================
# Multi-stage build for optimized image size
# Supports both CPU and GPU (CUDA 12.4) runtime
# =============================================================================

# =============================================================================
# Stage 1: Builder CPU - Install dependencies for CPU runtime
# =============================================================================
FROM python:3.11-slim AS builder-cpu

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
WORKDIR /build
COPY requirements.txt .

# Install PyTorch CPU version (smaller, faster for CPU-only deployments)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies (excluding torch/torchvision from requirements.txt)
RUN grep -v "^torch" requirements.txt | grep -v "^torchvision" > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt

# =============================================================================
# Stage 1b: Builder GPU - Install dependencies for GPU runtime with CUDA 12.8
# =============================================================================
FROM nvidia/cuda:12.8.0-base-ubuntu22.04 AS builder-gpu

# Install Python 3.11 and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
WORKDIR /build
COPY requirements.txt .

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies (excluding torch/torchvision from requirements.txt)
RUN grep -v "^torch" requirements.txt | grep -v "^torchvision" > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt

# =============================================================================
# Stage 2: Runtime - CPU version (default)
# =============================================================================
FROM python:3.11-slim AS runtime-cpu

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder-cpu
COPY --from=builder-cpu /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    FLASK_ENV=production

# Create app directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/gate_captures static/parking_captures static/models

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')" || exit 1

# Expose port
EXPOSE 5001

# Default command
CMD ["python", "main.py"]

# =============================================================================
# Stage 3: Runtime - GPU version (CUDA 12.8)
# =============================================================================
FROM nvidia/cuda:12.8.0-base-ubuntu22.04 AS runtime-gpu

# Install Python 3.11 and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libpq5 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy virtual environment from builder-gpu
COPY --from=builder-gpu /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    FLASK_ENV=production \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create app directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/gate_captures static/parking_captures static/models

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')" || exit 1

# Expose port
EXPOSE 5001

# Default command
CMD ["python", "main.py"]
