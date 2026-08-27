import time
import uuid
import tempfile
import os
import subprocess
import numpy as np
from scipy.io import wavfile
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-api")

app = FastAPI(title="Logistics Voice AI API")

class Prediction(BaseModel):
    prediction: str
    confidence: float

class AnalysisResponse(BaseModel):
    contact_id: str
    gender: Prediction
    age_bracket: Prediction
    processing_ms: int
    audio_quality: str

class AudioPipeline:
    def __init__(self):
        logger.info("Initializing Pure-Numpy Audio Pipeline...")

    def assess_quality(self, y):
        """Calculates SNR to handle logistics background noise."""
        signal_power = np.mean(y**2)
        noise_power = np.percentile(y**2, 10)
        if noise_power == 0: noise_power = 1e-10
        snr = 10 * np.log10(signal_power / noise_power)
        
        if snr < 10: return "insufficient"
        elif snr < 20: return "degraded"
        return "good"

    def predict(self, file_path: str):
        wav_path = file_path
        created_temp_wav = False

        if not file_path.lower().endswith(".wav"):
            wav_path = file_path + "_converted.wav"
            try:
                subprocess.run([
                    "ffmpeg", "-i", file_path, 
                    "-ac", "1", "-ar", "16000", 
                    wav_path, "-y"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                created_temp_wav = True
            except Exception:
                raise HTTPException(status_code=400, detail="Audio conversion failed. Ensure ffmpeg is installed.")

        try:
            sr, y = wavfile.read(wav_path)
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            y = y.astype(np.float32) / (np.max(np.abs(y)) + 1e-10) # Normalize
        finally:
            if created_temp_wav and os.path.exists(wav_path):
                os.remove(wav_path)

        quality = self.assess_quality(y)
        if quality == "insufficient":
            return {"gender": "unknown", "g_conf": 0.0, "age": "unknown", "a_conf": 0.0, "quality": quality}

        # --- 1. OPTIMIZED Pitch Extraction (Fast Autocorrelation via FFT) ---
        min_period = sr // 300 # Max search: 300 Hz
        max_period = sr // 40  # Min search: lowered to 40 Hz for deep/distorted voices
        
        # FFT changes time complexity from O(N^2) to O(N log N) - massively faster!
        n = len(y)
        f = np.fft.fft(y, n=2 * n)
        corr = np.fft.ifft(f * np.conjugate(f)).real
        valid_corr = corr[min_period:max_period]
        
        median_pitch = sr / (np.argmax(valid_corr) + min_period) if len(valid_corr) > 0 else 0

        # Gender Heuristic (Lowered floor to 50 Hz for heavy PA / Radio voices)
        if 50 <= median_pitch <= 170:
            gender, g_conf = "male", 0.85
        elif 170 < median_pitch <= 280:
            gender, g_conf = "female", 0.82
        else:
            gender, g_conf = "unknown", 0.45

        # --- 2. Spectral Centroid (FFT) ---
        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1.0/sr)
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

        if centroid < 1500: age, a_conf = "46-60", 0.65
        elif centroid < 2500: age, a_conf = "31-45", 0.70
        else: age, a_conf = "18-30", 0.60

        return {"gender": gender, "g_conf": g_conf, "age": age, "a_conf": a_conf, "quality": quality}

pipeline = AudioPipeline()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    contact_id: str = Form(default_factory=lambda: str(uuid.uuid4())),
    file: UploadFile = File(...)
):
    start_time = time.time()
    
    # Grab the original extension (e.g., .mp3) so the pipeline knows to trigger ffmpeg
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".tmp"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = pipeline.predict(tmp_path)
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio file")
    finally:
        os.remove(tmp_path) 
    
    return AnalysisResponse(
        contact_id=contact_id,
        gender=Prediction(prediction=results["gender"], confidence=results["g_conf"]),
        age_bracket=Prediction(prediction=results["age"], confidence=results["a_conf"]),
        processing_ms=int((time.time() - start_time) * 1000),
        audio_quality=results["quality"]
    )
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = pipeline.predict(tmp_path)
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio file")
    finally:
        os.remove(tmp_path) 
    
    return AnalysisResponse(
        contact_id=contact_id,
        gender=Prediction(prediction=results["gender"], confidence=results["g_conf"]),
        age_bracket=Prediction(prediction=results["age"], confidence=results["a_conf"]),
        processing_ms=int((time.time() - start_time) * 1000),
        audio_quality=results["quality"]
    )