FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies with FFmpeg best practices
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Verify FFmpeg installation and codecs
RUN ffmpeg -version && \
    ffmpeg -codecs | grep -E "(aac|mp3|wav|pcm)"

# Copy and install Python requirements
COPY config/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clone and setup CSM repository
RUN git clone https://github.com/SesameAILabs/csm.git csm_repo && \
    cd csm_repo && \
    pip install --no-cache-dir -e .

# Set environment variables for optimal performance
ENV NO_TORCH_COMPILE=1
ENV PYTHONPATH=/app/csm_repo:$PYTHONPATH
ENV MODEL_REPO=BiggestLab/csm-1b
ENV DEFAULT_SPEAKER_ID=0
ENV MAX_LENGTH_MS=30000
ENV GENERATION_LIMIT=100
ENV FFMPEG_BINARY=/usr/bin/ffmpeg
ENV TORCHAUDIO_BACKEND=ffmpeg

# Copy application files
COPY src/ ./src/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import torchaudio; print('OK')"

CMD ["python", "-u", "/app/src/handler.py"]
