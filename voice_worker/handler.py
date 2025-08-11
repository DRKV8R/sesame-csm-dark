import os
import sys
import base64
import tempfile
import logging
import time
from typing import Dict, Any, List
import runpod
import torch
import torchaudio
from faster_whisper import WhisperModel
from transformers import CsmForConditionalGeneration, AutoProcessor

# --- Logging Setup ---
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# --- Global Model Instances ---
VOICE_MODEL = None
PROCESSOR = None
WHISPER_MODEL = None

# --- Main Worker Class ---
class VoiceWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "sesame/csm-1b"
        self.whisper_model_size = "base"

    def init_models(self):
        """Initializes CSM, Processor, and Whisper models if not loaded."""
        global VOICE_MODEL, PROCESSOR, WHISPER_MODEL
        
        if VOICE_MODEL is None or PROCESSOR is None:
            logger.info(f"Loading Sesame CSM model and processor from '{self.model_id}'...")
            PROCESSOR = AutoProcessor.from_pretrained(self.model_id)
            VOICE_MODEL = CsmForConditionalGeneration.from_pretrained(self.model_id, device_map=self.device)
            logger.info("Sesame CSM model and processor loaded successfully.")

        if WHISPER_MODEL is None:
            logger.info(f"Loading faster-whisper model '{self.whisper_model_size}'...")
            compute_type = "float16" if self.device == "cuda" else "float32"
            WHISPER_MODEL = WhisperModel(self.whisper_model_size, device=self.device, compute_type=compute_type)
            logger.info("faster-whisper model loaded successfully.")

    def generate_voice(self, text_prompt: str, context_history: List[Dict], speaker_id: int) -> Dict[str, Any]:
        """Generates voice using the official Hugging Face CSM implementation."""
        self.init_models()
        
        conversation = []
        
        # 1. Build the context from history
        for turn in context_history:
            if "audio_b64" in turn and turn["audio_b64"]:
                try:
                    # Decode and load audio from base64
                    audio_bytes = base64.b64decode(turn["audio_b64"])
                    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_file.seek(0)
                        # Use the processor's feature extractor to load the audio
                        audio_array, sr = torchaudio.load(tmp_file.name)
                    
                    conversation.append({
                        "role": f"{turn.get('speaker', 0)}",
                        "content": [
                            {"type": "text", "text": turn.get("text", "")},
                            {"type": "audio", "path": audio_array.numpy()} # Pass the numpy array
                        ],
                    })
                except Exception as e:
                    logger.error(f"Failed to process context turn: {e}")

        # 2. Add the final text prompt
        conversation.append({
            "role": f"{speaker_id}",
            "content": [{"type": "text", "text": text_prompt}]
        })
        
        logger.info(f"Generating audio with {len(conversation)} turns in total.")
        
        # Process the entire conversation using the new chat template format
        inputs = PROCESSOR.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.device)

        # Infer the model
        audio = VOICE_MODEL.generate(**inputs, output_audio=True)
        
        # Save the audio to a base64 string
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            PROCESSOR.save_audio(audio, tmp_file.name)
            tmp_file.seek(0)
            audio_b64 = base64.b64encode(tmp_file.read()).decode('utf-8')
            
        logger.info("Voice generation successful.")
        return {"audio_base64": audio_b64}

# --- RunPod Handler ---
worker = VoiceWorker()

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler for voice operations."""
    inp = event.get('input', {})
    
    try:
        text = inp.get('text')
        context_history = inp.get('context_history', [])
        speaker_id = inp.get('speaker_id', 0)
        
        if not text:
            return {"error": "Input 'text' is required."}
        
        return worker.generate_voice(text, context_history, speaker_id)

    except Exception as e:
        logger.exception("Error during voice generation")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
