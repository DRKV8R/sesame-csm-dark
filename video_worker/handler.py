import os
import sys
import base64
import tempfile
import logging
import json
import time
from typing import Dict, Any, Optional, List
import runpod
import torch
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

# Video processing imports
try:
    from diffusers import DiffusionPipeline
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    import accelerate
except ImportError as e:
    logging.error(f"Video processing imports failed: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instances
VIDEO_MODEL = None
LORA_ADAPTERS = {}
GENERATION_COUNT = 0

class VideoWorker:
    """WAN 2.1 video training and generation worker"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.getenv('VIDEO_MODEL_PATH', 'Wan-AI/Wan2.1-T2V-1.3B')
        self.generation_limit = int(os.getenv('GENERATION_LIMIT', 100))
        self.max_duration = int(os.getenv('MAX_VIDEO_DURATION', 6))
        
    def init_video_model(self):
        """Initialize WAN 2.1 model"""
        global VIDEO_MODEL
        if VIDEO_MODEL is not None:
            return VIDEO_MODEL
        
        try:
            logger.info(f"Loading WAN 2.1 from {self.model_path}")
            VIDEO_MODEL = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            if self.device == "cuda":
                VIDEO_MODEL.to(self.device)
                VIDEO_MODEL.enable_attention_slicing()
            logger.info("Video model loaded successfully")
            return VIDEO_MODEL
        except Exception as e:
            logger.error(f"Video model initialization failed: {e}")
            raise

    def generate_video(self, prompt: str, character_lora: str = None, duration: int = 4) -> Dict[str, Any]:
        """Generate video with optional character LoRA"""
        global GENERATION_COUNT
        if GENERATION_COUNT >= self.generation_limit:
            return {"error": "Generation limit reached"}
        
        duration = min(duration, self.max_duration)
        num_frames = duration * 8
        logger.info(f"Generating video: '{prompt}' ({duration}s)")
        
        model = self.init_video_model()
        
        with torch.no_grad():
            result = model(prompt=prompt, num_frames=num_frames, height=512, width=512)
        
        video_frames = result.frames[0]
        video_b64 = self.frames_to_video_b64(video_frames, fps=8)
        
        GENERATION_COUNT += 1
        return {"video_base64": video_b64, "format": "mp4"}

    def frames_to_video_b64(self, frames: List[np.ndarray], fps: int = 8) -> str:
        """Convert video frames to base64 MP4"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        
        with open(temp_path, 'rb') as video_file:
            video_bytes = video_file.read()
        os.unlink(temp_path)
        
        return base64.b64encode(video_bytes).decode('utf-8')

worker = VideoWorker()

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler for video operations"""
    inp = event.get('input', {})
    action = inp.get('action', 'generate_video')
    
    if action == 'generate_video':
        return worker.generate_video(inp.get('prompt', ''))
    # ... add other actions like training
    return {"error": f"Unknown action: {action}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
    
