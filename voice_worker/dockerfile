FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    sox \
    libsox-fmt-all \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ---- Environment variables ----
ENV NO_TORCH_COMPILE=1
ENV PYTHONPATH=/app:/app/csm_repo:$PYTHONPATH
ENV VOICE_MODEL_REPO=sesame/csm-1b
ENV GENERATION_LIMIT=200
ENV MAX_LENGTH_MS=30000
ENV FFMPEG_BINARY=/usr/bin/ffmpeg
ENV TORCHAUDIO_BACKEND=ffmpeg
ENV HF_HOME=/workspace/hf_cache
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1

# ---- Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# ---- CSM repo ----
RUN git clone https://github.com/SesameAILabs/csm.git csm_repo && \
    cd csm_repo && \
    pip install --no-cache-dir -e . && \
    python -c "from generator import load_csm_1b; print('CSM import successful')"

# ---- Application files ----
COPY handler.py ./handler.py

# ---- Workspace setup ----
RUN mkdir -p /workspace/voice_loras /workspace/hf_cache /workspace/temp && \
    chmod -R 755 /workspace

# ---- Non-root user ----
RUN useradd -m -u 1000 -s /bin/bash runpod && \
    chown -R runpod:runpod /app /workspace

USER runpod

# ---- Pre-warm common imports ----
RUN python -c "import torch; import torchaudio; import librosa; print('Voice imports successful')"

# ---- Health check ----
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)"

LABEL maintainer="David Hamilton 2025"
LABEL description="Sesame CSM Voice Training and Generation Worker"
LABEL version="1.0.0"
LABEL repository="https://github.com/drkv8r/sesame_csm_dark"
LABEL org.opencontainers.image.title="Sesame CSM Voice AI Portfolio"
LABEL org.opencontainers.image.authors="David Hamilton"
LABEL org.opencontainers.image.url="https://your-resume-or-portfolio-site"
LABEL org.opencontainers.image.description="Portfolio build for event-quality, serverless, AI voice generation"

CMD ["python", "-u", "handler.py"]
