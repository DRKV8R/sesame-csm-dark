FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies with FFmpeg best practices
RUN apt-get update && apt-get install -y \
    # FFmpeg and audio libraries
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    libavfilter-dev \
    # Audio processing tools
    sox \
    libsox-fmt-all \
    # System utilities
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Verify FFmpeg installation with comprehensive codec support
RUN ffmpeg -version && \
    ffmpeg -codecs | grep -E "(aac|mp3|wav|pcm|flac)" && \
    sox --version

# Set environment variables for optimal performance
ENV NO_TORCH_COMPILE=1
ENV PYTHONPATH=/app/csm_repo:$PYTHONPATH
ENV MODEL_REPO=BiggestLab/csm-1b
ENV DEFAULT_SPEAKER_ID=0
ENV MAX_LENGTH_MS=30000
ENV GENERATION_LIMIT=100
ENV FFMPEG_BINARY=/usr/bin/ffmpeg
ENV TORCHAUDIO_BACKEND=ffmpeg
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Install Python dependencies
COPY config/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Clone and setup CSM repository with error handling
RUN git clone https://github.com/SesameAILabs/csm.git csm_repo || \
    (echo "Failed to clone CSM repo" && exit 1) && \
    cd csm_repo && \
    pip install --no-cache-dir -e . && \
    # Verify installation
    python -c "from generator import load_csm_1b; print('CSM import successful')"

# Copy application files
COPY src/ ./src/

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app && \
    # Create cache directories
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser/.cache

# Switch to non-root user
USER appuser

# Pre-warm common Python imports to reduce cold start
RUN python -c "import torch; import torchaudio; import librosa; import numpy as np; print('All imports successful')"

# Health check with comprehensive testing
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from handler import health_check; result = health_check(); exit(0 if result['status'] == 'healthy' else 1)"

# Set labels for better container management
LABEL maintainer="David Hamilton 2025"
LABEL description="Sesame CSM 1B Voice Cloning with FFmpeg optimization"
LABEL version="1.0.0"
LABEL repository="https://github.com/drkv8r/sesame-csm-dark"

# Start the handler
CMD ["python", "-u", "/app/src/handler.py"]
