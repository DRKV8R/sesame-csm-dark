{
  "name": "Sesame CSM 1B Voice Studio",
  "description": "Professional voice cloning with zero idle costs",
  "dockerImage": "davidhamilton/sesame-csm:latest",
  "containerDiskInGb": 20,
  "volumeDiskInGb": 0,
  "env": {
    "MODEL_REPO": "BiggestLab/csm-1b",
    "DEFAULT_SPEAKER_ID": "0",
    "MAX_LENGTH_MS": "30000",
    "GENERATION_LIMIT": "100",
    "NO_TORCH_COMPILE": "1",
    "FFMPEG_BINARY": "/usr/bin/ffmpeg",
    "TORCHAUDIO_BACKEND": "ffmpeg"
  },
  "ports": "8000/http",
  "volumeMountPath": "/workspace",
  "startSSH": false,
  "category": "AI/ML",
  "readme": "# Sesame CSM 1B Voice Studio\n\nCreated by David Hamilton 2025\n\nProfessional voice cloning with:\n- Zero idle costs ($0.000 when stopped)\n- 2-5s inference time\n- 15-30s cold start\n- FFmpeg optimized audio processing\n\nSource: https://github.com/yourusername/sesame-csm-dark"
}
