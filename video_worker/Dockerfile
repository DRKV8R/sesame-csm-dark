FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Video processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    # OpenCV dependencies
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    # System utilities
    git \
    curl \
    wget \
    unzip \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV VIDEO_MODEL_PATH=Wan-AI/Wan2.1-T2V-1.3B
ENV MAX_VIDEO_DURATION=6
ENV GENERATION_LIMIT=100
ENV FFMPEG_BINARY=/usr/bin/ffmpeg
ENV HF_HOME=/workspace/hf_cache
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Install Python dependencies
COPY video-worker/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY video-worker/handler.py ./handler.py
COPY video-worker/training.py ./training.py

# Create workspace directories
RUN mkdir -p /workspace/video_loras /workspace/hf_cache /workspace/temp && \
    chmod -R 755 /workspace

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash runpod && \
    chown -R runpod:runpod /app /workspace

# Switch to non-root user
USER runpod

# Pre-warm imports
RUN python -c "import torch; import cv2; from PIL import Image; import numpy as np; print('Video imports successful')"

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)"

# Labels
LABEL maintainer="David Hamilton 2025"
LABEL description="WAN 2.1 Video Training & Generation Worker"
LABEL version="1.0.0"
LABEL repository="https://github.com/drkv8r/sesame-csm-dark"

# Start the handler
CMD ["python", "-u", "handler.py"]
