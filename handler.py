import os
import sys
import base64
import tempfile
import logging
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import runpod
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np

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

# Global state
MODEL = None
GENERATION_COUNT = 0
LAST_RESET = datetime.now()

class AudioProcessor:
    """FFmpeg-optimized audio processing with quality analysis"""
    
    @staticmethod
    def verify_ffmpeg():
        """Verify FFmpeg installation and capabilities"""
        try:
            import subprocess
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            version_line = result.stdout.split('\n')[0]
            logger.info(f"FFmpeg verified: {version_line}")
            
            # Check for required codecs
            codecs_result = subprocess.run(
                ['ffmpeg', '-codecs'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            required_codecs = ['aac', 'mp3', 'pcm', 'wav']
            available_codecs = codecs_result.stdout.lower()
            
            for codec in required_codecs:
                if codec not in available_codecs:
                    logger.warning(f"Codec {codec} may not be available")
            
            return True
            
        except Exception as e:
            logger.error(f"FFmpeg verification failed: {e}")
            return False
    
    @staticmethod
    def analyze_audio_quality(audio_path: str) -> Dict[str, Any]:
        """Analyze audio quality for voice cloning suitability"""
        try:
            # Load audio with librosa for analysis
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            # Quality metrics
            quality_metrics = {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1 if len(y.shape) == 1 else y.shape[0],
                'rms_energy': float(np.sqrt(np.mean(y**2))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            }
            
            # Quality assessment
            issues = []
            score = 100
            
            if duration < 30:
                issues.append("Audio too short (minimum 30 seconds)")
                score -= 50
            elif duration < 60:
                issues.append("Short audio may limit voice quality")
                score -= 20
            
            if sr < 22050:
                issues.append("Low sample rate may affect quality")
                score -= 15
            
            if quality_metrics['rms_energy'] < 0.01:
                issues.append("Audio level too low")
                score -= 20
            elif quality_metrics['rms_energy'] > 0.8:
                issues.append("Audio may be clipping")
                score -= 30
            
            quality_metrics['score'] = max(0, score)
            quality_metrics['issues'] = issues
            quality_metrics['suitable'] = score >= 60
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Audio quality analysis failed: {e}")
            return {
                'duration': 0,
                'score': 0,
                'issues': [f"Analysis failed: {str(e)}"],
                'suitable': False
            }
    
    @staticmethod
    def process_audio(audio_b64: str, target_sr: int = 24000) -> tuple:
        """Process audio with FFmpeg backend and quality optimization"""
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_b64)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name
            
            # Analyze quality first
            quality_info = AudioProcessor.analyze_audio_quality(temp_path)
            
            if not quality_info['suitable']:
                logger.warning(f"Audio quality issues: {quality_info['issues']}")
                # Continue processing but log warnings
            
            # Load and process with librosa (uses FFmpeg)
            audio, sr = librosa.load(
                temp_path, 
                sr=target_sr, 
                mono=True,
                res_type='kaiser_fast'
            )
            
            # Audio enhancement for voice cloning
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Light noise reduction if needed
            if quality_info['rms_energy'] > 0:
                # Simple spectral gating for noise reduction
                noise_threshold = np.percentile(np.abs(audio), 10)
                audio = np.where(np.abs(audio) < noise_threshold, 
                                audio * 0.3, audio)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            logger.info(f"Processed audio: shape={audio_tensor.shape}, sr={sr}, quality_score={quality_info['score']}")
            return audio_tensor, sr, quality_info
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise

class GenerationLimiter:
    """Manage generation limits and billing tracking"""
    
    def __init__(self):
        self.daily_limit = int(os.getenv('GENERATION_LIMIT', 100))
        self.cost_per_generation = 0.02
        self.reset_hour = 0  # Reset at midnight UTC
        
    def should_reset_counter(self) -> bool:
        """Check if we should reset the daily counter"""
        global LAST_RESET
        now = datetime.now()
        
        # Reset daily at midnight or if it's been more than 24 hours
        if (now.hour == self.reset_hour and LAST_RESET.day != now.day) or \
           (now - LAST_RESET > timedelta(days=1)):
            return True
        return False
    
    def check_limit(self) -> Dict[str, Any]:
        """Check generation limits and return status"""
        global GENERATION_COUNT, LAST_RESET
        
        # Reset counter if needed
        if self.should_reset_counter():
            GENERATION_COUNT = 0
            LAST_RESET = datetime.now()
            logger.info("Daily generation count reset")
        
        # Check limits
        if self.daily_limit > 0 and GENERATION_COUNT >= self.daily_limit:
            return {
                'allowed': False,
                'reason': 'Daily generation limit reached',
                'current_count': GENERATION_COUNT,
                'daily_limit': self.daily_limit,
                'reset_time': (LAST_RESET + timedelta(days=1)).isoformat()
            }
        
        return {
            'allowed': True,
            'current_count': GENERATION_COUNT,
            'daily_limit': self.daily_limit,
            'remaining': max(0, self.daily_limit - GENERATION_COUNT) if self.daily_limit > 0 else 'unlimited'
        }
    
    def increment_count(self) -> Dict[str, Any]:
        """Increment generation counter and return usage info"""
        global GENERATION_COUNT
        GENERATION_COUNT += 1
        
        usage_info = {
            'generation_number': GENERATION_COUNT,
            'daily_limit': self.daily_limit,
            'estimated_cost': round(GENERATION_COUNT * self.cost_per_generation, 4),
            'idle_cost': 0.0000
        }
        
        logger.info(f"Generation #{GENERATION_COUNT}/{self.daily_limit}, cost: ${usage_info['estimated_cost']}")
        return usage_info

def init_model(repo: str = None) -> Any:
    """Initialize CSM model with error handling and optimization"""
    global MODEL
    
    if MODEL is not None:
        logger.info("Using cached model")
        return MODEL
    
    try:
        repo = repo or os.getenv('MODEL_REPO', 'BiggestLab/csm-1b')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading CSM 1B from {repo} on {device}")
        
        # Load model with optimizations
        MODEL = load_csm_1b(device=device, repo=repo)
        
        # Model optimization for inference
        if hasattr(MODEL, 'eval'):
            MODEL.eval()
        
        # Enable inference mode optimizations
        torch.set_grad_enabled(False)
        
        if device == "cuda":
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Clear cache
            torch.cuda.empty_cache()
        
        logger.info("Model loaded and optimized successfully")
        return MODEL
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

def prepare_context(audio_b64: str, text: str, speaker: int) -> Optional[Segment]:
    """Prepare voice cloning context with quality validation"""
    try:
        audio_tensor, sr, quality_info = AudioProcessor.process_audio(audio_b64, target_sr=24000)
        
        if not quality_info['suitable']:
            logger.warning(f"Using low-quality audio for context: {quality_info['issues']}")
        
        return Segment(text=text, speaker=speaker, audio=audio_tensor)
        
    except Exception as e:
        logger.warning(f"Context preparation failed: {e}")
        return None

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced handler with generation limits and quality control"""
    start_time = time.time()
    
    try:
        # Verify FFmpeg on first run
        if not AudioProcessor.verify_ffmpeg():
            logger.warning("FFmpeg verification failed - some audio processing may be limited")
        
        # Check generation limits
        limiter = GenerationLimiter()
        limit_check = limiter.check_limit()
        
        if not limit_check['allowed']:
            return {
                "error": limit_check['reason'],
                "details": limit_check,
                "handler_info": "David Hamilton 2025"
            }
        
        # Parse and validate input
        inp = event.get('input', {})
        text = inp.get('text', '').strip()
        
        if not text:
            return {"error": "text parameter is required"}
        
        if len(text) > 1000:
            return {"error": "text too long (maximum 1000 characters)"}
        
        # Get parameters with validation
        repo = inp.get('model_repo') or os.getenv('MODEL_REPO', 'BiggestLab/csm-1b')
        speaker_id = inp.get('speaker_id', int(os.getenv('DEFAULT_SPEAKER_ID', 0)))
        max_ms = min(
            inp.get('max_length_ms', int(os.getenv('MAX_LENGTH_MS', 30000))),
            30000  # Hard limit for cost control
        )
        
        # Validate speaker_id
        if not isinstance(speaker_id, int) or speaker_id < 0 or speaker_id > 10:
            return {"error": "speaker_id must be an integer between 0 and 10"}
        
        # Process voice cloning context
        context = []
        context_quality = None
        
        if inp.get('reference_audio') and inp.get('reference_text'):
            try:
                ctx_segment = prepare_context(
                    inp['reference_audio'],
                    inp['reference_text'],
                    speaker_id
                )
                if ctx_segment:
                    context.append(ctx_segment)
                    logger.info("Voice cloning context prepared successfully")
                    
                    # Get quality info for response
                    _, _, context_quality = AudioProcessor.process_audio(inp['reference_audio'])
                    
            except Exception as e:
                logger.warning(f"Context preparation failed: {e}")
        
        # Initialize model
        model = init_model(repo)
        
        # Generate speech
        generation_start = time.time()
        
        try:
            audio = model.generate(
                text=text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=max_ms
            )
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return {"error": f"Speech generation failed: {str(e)}"}
        
        generation_time = time.time() - generation_start
        
        # Save output with FFmpeg backend
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
                # Use FFmpeg backend for encoding
                torchaudio.save(
                    tmp_file.name,
                    audio.unsqueeze(0).cpu(),
                    24000,
                    backend="ffmpeg"
                )
                
                # Read and encode to base64
                with open(tmp_file.name, 'rb') as f:
                    audio_b64 = base64.b64encode(f.read()).decode('utf-8')
                
            finally:
                # Always clean up
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
        
        # Update usage tracking
        usage_info = limiter.increment_count()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Build comprehensive response
        response = {
            "audio_base64": audio_b64,
            "sample_rate": 24000,
            "format": "wav",
            "text": text,
            "speaker_id": speaker_id,
            "max_length_ms": max_ms,
            "context_used": len(context) > 0,
            "generation_time": round(generation_time, 2),
            "total_time": round(total_time, 2),
            "usage": usage_info,
            "model_repo": repo,
            "created_by": "David Hamilton 2025"
        }
        
        # Add context quality info if available
        if context_quality:
            response["context_quality"] = {
                "score": context_quality['score'],
                "suitable": context_quality['suitable'],
                "duration": round(context_quality['duration'], 1)
            }
        
        logger.info(f"Generation completed successfully in {total_time:.2f}s")
        return response
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Handler error: {e}")
        
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "total_time": round(total_time, 2),
            "handler_info": "David Hamilton 2025"
        }

# Health check endpoint
def health_check():
    """Health check for RunPod"""
    try:
        # Check model availability
        model_status = MODEL is not None
        
        # Check FFmpeg
        ffmpeg_status = AudioProcessor.verify_ffmpeg()
        
        # Check CUDA if available
        cuda_status = torch.cuda.is_available()
        if cuda_status:
            torch.cuda.empty_cache()
        
        return {
            "status": "healthy",
            "model_loaded": model_status,
            "ffmpeg_available": ffmpeg_status,
            "cuda_available": cuda_status,
            "generation_count": GENERATION_COUNT,
            "handler_version": "David Hamilton 2025"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    logger.info("Starting Sesame CSM 1B Handler - Created by David Hamilton 2025")
    logger.info(f"FFmpeg available: {AudioProcessor.verify_ffmpeg()}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Start RunPod serverless
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    })
