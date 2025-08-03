import os
import sys
import base64
import tempfile
import logging
import json
import time
from typing import Dict, Any, Optional
import runpod
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime, timedelta

# Add CSM to path
sys.path.append('/app/csm_repo')

try:
    from generator import load_csm_1b, Segment
except ImportError as e:
    logging.error(f"Failed to import CSM: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
MODEL = None
GENERATION_COUNT = 0
LAST_RESET = datetime.now()

class AudioProcessor:
    """FFmpeg-optimized audio processing"""
    
    @staticmethod
    def verify_ffmpeg():
        """Verify FFmpeg installation"""
        try:
            import subprocess
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            logger.info(f"FFmpeg version: {result.stdout.split()[2]}")
            return True
        except Exception as e:
            logger.error(f"FFmpeg verification failed: {e}")
            return False
    
    @staticmethod
    def process_audio(audio_b64: str, target_sr: int = 24000) -> tuple:
        """Process audio with FFmpeg backend"""
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_b64)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name
            
            # Load with librosa (uses FFmpeg backend)
            audio, sr = librosa.load(
                temp_path, 
                sr=target_sr, 
                mono=True,
                res_type='kaiser_fast'
            )
            
            # Clean up
            os.unlink(temp_path)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            logger.info(f"Processed audio: shape={audio_tensor.shape}, sr={sr}")
            return audio_tensor, sr
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise

class GenerationLimiter:
    """Manage generation limits and billing"""
    
    def __init__(self):
        self.daily_limit = int(os.getenv('GENERATION_LIMIT', 100))
        self.cost_per_generation = 0.02
        
    def check_limit(self) -> bool:
        """Check if generation limit is reached"""
        global GENERATION_COUNT, LAST_RESET
        
        # Reset daily counter
        if datetime.now() - LAST_RESET > timedelta(days=1):
            GENERATION_COUNT = 0
            LAST_RESET = datetime.now()
            logger.info("Daily generation count reset")
        
        if self.daily_limit != -1 and GENERATION_COUNT >= self.daily_limit:
            logger.warning(f"Generation limit reached: {GENERATION_COUNT}/{self.daily_limit}")
            return False
            
        return True
    
    def increment_count(self):
        """Increment generation counter"""
        global GENERATION_COUNT
        GENERATION_COUNT += 1
        logger.info(f"Generation count: {GENERATION_COUNT}/{self.daily_limit}")

def init_model(repo: str = None) -> Any:
    """Initialize CSM model with error handling"""
    global MODEL
    
    if MODEL is not None:
        return MODEL
    
    try:
        repo = repo or os.getenv('MODEL_REPO', 'BiggestLab/csm-1b')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading CSM 1B from {repo} on {device}")
        MODEL = load_csm_1b(device=device, repo=repo)
        
        logger.info("Model loaded successfully")
        return MODEL
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

def prepare_context(audio_b64: str, text: str, speaker: int) -> Optional[Segment]:
    """Prepare voice cloning context"""
    try:
        audio_tensor, _ = AudioProcessor.process_audio(audio_b64, target_sr=24000)
        return Segment(text=text, speaker=speaker, audio=audio_tensor)
    except Exception as e:
        logger.warning(f"Context preparation failed: {e}")
        return None

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler function"""
    start_time = time.time()
    
    try:
        # Verify FFmpeg on first run
        if not AudioProcessor.verify_ffmpeg():
            return {"error": "FFmpeg not available - audio processing disabled"}
        
        # Check generation limits
        limiter = GenerationLimiter()
        if not limiter.check_limit():
            return {
                "error": "Daily generation limit reached",
                "limit": limiter.daily_limit,
                "count": GENERATION_COUNT
            }
        
        # Parse input
        inp = event.get('input', {})
        text = inp.get('text')
        
        if not text:
            return {"error": "text parameter is required"}
        
        if len(text) > 1000:
            return {"error": "text too long (max 1000 characters)"}
        
        # Get parameters with defaults
        repo = inp.get('model_repo') or os.getenv('MODEL_REPO', 'BiggestLab/csm-1b')
        speaker_id = inp.get('speaker_id', int(os.getenv('DEFAULT_SPEAKER_ID', 0)))
        max_ms = min(
            inp.get('max_length_ms', int(os.getenv('MAX_LENGTH_MS', 30000))),
            30000  # Hard limit
        )
        
        # Process voice cloning context
        context = []
        if inp.get('reference_audio') and inp.get('reference_text'):
            ctx_segment = prepare_context(
                inp['reference_audio'],
                inp['reference_text'],
                speaker_id
            )
            if ctx_segment:
                context.append(ctx_segment)
                logger.info("Voice cloning context prepared")
        
        # Generate speech
        model = init_model(repo)
        
        generation_start = time.time()
        audio = model.generate(
            text=text,
            speaker=speaker_id,
            context=context,
            max_audio_length_ms=max_ms
        )
        generation_time = time.time() - generation_start
        
        # Save output with FFmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(
                tmp_file.name,
                audio.unsqueeze(0).cpu(),
                24000,
                backend="ffmpeg"
            )
            
            # Read and encode
            with open(tmp_file.name, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            os.unlink(tmp_file.name)
        
        # Increment counter
        limiter.increment_count()
        
        total_time = time.time() - start_time
        
        return {
            "audio_base64": audio_b64,
            "sample_rate": 24000,
            "format": "wav",
            "text": text,
            "speaker_id": speaker_id,
            "context_used": len(context) > 0,
            "generation_time": round(generation_time, 2),
            "total_time": round(total_time, 2),
            "generation_count": GENERATION_COUNT,
            "estimated_cost": round(limiter.cost_per_generation, 4),
            "created_by": "David Hamilton 2025"
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            "error": str(e),
            "type": type(e).__name__,
            "total_time": round(time.time() - start_time, 2)
        }

if __name__ == "__main__":
    logger.info("Starting Sesame CSM 1B handler - Created by David Hamilton 2025")
    logger.info(f"FFmpeg available: {AudioProcessor.verify_ffmpeg()}")
    runpod.serverless.start({"handler": handler})
