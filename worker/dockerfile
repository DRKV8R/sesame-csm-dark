FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # FFmpeg and audio libraries
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    libavfilter-dev \
    # Audio processing
    sox \
    libsox-fmt-all \
    # OpenCV dependencies
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    # System utilities
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV NO_TORCH_COMPILE=1
ENV PYTHONPATH=/app/csm_repo:$PYTHONPATH
ENV VOICE_MODEL_REPO=BiggestLab/csm-1b
ENV VIDEO_MODEL_PATH=Wan-AI/Wan2.1-T2V-1.3B
ENV DEFAULT_SPEAKER_ID=0
ENV MAX_LENGTH_MS=30000
ENV MAX_VIDEO_DURATION=6
ENV GENERATION_LIMIT=100
ENV FFMPEG_BINARY=/usr/bin/ffmpeg
ENV TORCHAUDIO_BACKEND=ffmpeg
ENV HF_HOME=/workspace/hf_cache
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Install Python dependencies
COPY config/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Clone and setup CSM repository
RUN git clone https://github.com/SesameAILabs/csm.git csm_repo && \
    cd csm_repo && \
    pip install --no-cache-dir -e . && \
    python -c "from generator import load_csm_1b; print('CSM import successful')"

# Copy application files
COPY src/unified_handler.py ./src/handler.py

# Create workspace directory structure
RUN mkdir -p /workspace/voice_loras /workspace/video_loras /workspace/hf_cache && \
    chmod -R 755 /workspace

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app /workspace

# Switch to non-root user
USER appuser

# Pre-warm imports to reduce cold start
RUN python -c "import torch; import torchaudio; import librosa; import numpy as np; from PIL import Image; print('All imports successful')"

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from handler import health_check; result = health_check(); exit(0 if result['status'] == 'healthy' else 1)"

# Labels
LABEL maintainer="David Hamilton 2025"
LABEL description="Unified AI Handler - Voice & Video with LoRA Training"
LABEL version="2.0.0"
LABEL repository="https://github.com/drkv8r/sesame-csm-dark"

# Start the handler
CMD ["python", "-u", "/app/src/handler.py"]
