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
from io import BytesIO

# Voice processing imports
try:
    import torchaudio
    import librosa
    import soundfile as sf
    import whisper
    from generator import load_csm_1b, Segment
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except ImportError as e:
    logging.error(f"Voice processing imports failed: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instances
VOICE_MODEL = None
WHISPER_MODEL = None
LORA_ADAPTERS = {}
GENERATION_COUNT = 0

class VoiceWorker:
    """Sesame CSM voice training and generation worker"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_repo = os.getenv('VOICE_MODEL_REPO', 'BiggestLab/csm-1b')
        self.generation_limit = int(os.getenv('GENERATION_LIMIT', 200))
        self.max_length_ms = int(os.getenv('MAX_LENGTH_MS', 30000))
        
    def init_voice_model(self):
        """Initialize Sesame CSM model"""
        global VOICE_MODEL
        
        if VOICE_MODEL is not None:
            return VOICE_MODEL
        
        try:
            logger.info(f"Loading Sesame CSM from {self.model_repo}")
            VOICE_MODEL = load_csm_1b(device=self.device, repo=self.model_repo)
            VOICE_MODEL.eval()
            
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()
            
            logger.info("Voice model loaded successfully")
            return VOICE_MODEL
            
        except Exception as e:
            logger.error(f"Voice model initialization failed: {e}")
            raise
    
    def init_whisper_model(self):
        """Initialize Whisper for transcription"""
        global WHISPER_MODEL
        
        if WHISPER_MODEL is not None:
            return WHISPER_MODEL
        
        try:
            logger.info("Loading Whisper model")
            WHISPER_MODEL = whisper.load_model("base", device=self.device)
            logger.info("Whisper model loaded successfully")
            return WHISPER_MODEL
            
        except Exception as e:
            logger.error(f"Whisper initialization failed: {e}")
            raise
    
    def transcribe_audio(self, audio_b64: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        try:
            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name
            
            whisper_model = self.init_whisper_model()
            result = whisper_model.transcribe(temp_path)
            os.unlink(temp_path)
            
            return {
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"]
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def prepare_training_data(self, audio_b64: str, text: str = None) -> Dict[str, Any]:
        """Prepare audio data for LoRA training"""
        try:
            logger.info("Preparing voice training data")
            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name
            
            audio, sr = librosa.load(temp_path, sr=24000, mono=True)
            duration = len(audio) / sr
            
            if duration < 30:
                raise ValueError("Audio must be at least 30 seconds for LoRA training")
            
            if not text:
                transcription = self.transcribe_audio(audio_b64)
                text = transcription["text"]
            
            audio = librosa.util.normalize(audio)
            audio_tensor = torch.from_numpy(audio).float()
            os.unlink(temp_path)
            
            return {
                "audio_tensor": audio_tensor,
                "text": text,
                "duration": duration,
                "sample_rate": sr,
                "status": "ready"
            }
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            raise
    
    def train_voice_lora(self, training_data: Dict[str, Any], voice_name: str) -> Dict[str, Any]:
        """Train LoRA adapter for voice cloning"""
        logger.info(f"Starting LoRA training for voice: {voice_name}")
        # Placeholder for actual training logic
        time.sleep(5) # Simulate training time
        lora_id = f"voice_{voice_name}_{int(time.time())}"
        LORA_ADAPTERS[lora_id] = {"name": voice_name}
        return {"status": "success", "lora_id": lora_id}

    def generate_voice(self, text: str, voice_lora: str = None, speaker_id: int = 0) -> Dict[str, Any]:
        """Generate voice with optional LoRA"""
        global GENERATION_COUNT
        if GENERATION_COUNT >= self.generation_limit:
            return {"error": "Generation limit reached"}
        
        logger.info(f"Generating voice: '{text[:50]}...'")
        model = self.init_voice_model()
        
        audio = model.generate(text=text, speaker=speaker_id)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio.unsqueeze(0).cpu(), 24000)
            with open(tmp_file.name, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            os.unlink(tmp_file.name)
        
        GENERATION_COUNT += 1
        return {"audio_base64": audio_b64}

worker = VoiceWorker()

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler for voice operations"""
    inp = event.get('input', {})
    action = inp.get('action', 'generate_voice')
    
    if action == 'generate_voice':
        return worker.generate_voice(inp.get('text', ''))
    elif action == 'train_lora':
        return worker.train_voice_lora(inp, inp.get('voice_name', 'default'))
    # ... add other actions
    return {"error": f"Unknown action: {action}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
