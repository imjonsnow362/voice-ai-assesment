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

# --- API Schemas ---
class Prediction(BaseModel):
    prediction: str
    confidence: float

class AnalysisResponse(BaseModel):
    contact_id: str
    gender: Prediction
    age_bracket: Prediction
    processing_ms: int
    audio_quality: str

# --- Inference Engine ---
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
        wav_path = file_path + "_converted.wav"
        try:
            # Use ffmpeg natively to decode ANY compressed codec to 16kHz WAV
            subprocess.run([
                "ffmpeg", "-i", file_path, 
                "-ac", "1", "-ar", "16000", 
                wav_path, "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            sr, y = wavfile.read(wav_path)
            y = y.astype(np.float32) / (np.max(np.abs(y)) + 1e-10) # Normalize
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        quality = self.assess_quality(y)
        if quality == "insufficient":
            return {"gender": "unknown", "g_conf": 0.0, "age": "unknown", "a_conf": 0.0, "quality": quality}

        # --- 1. Pitch Extraction (Pure Math Autocorrelation) ---
        min_period = sr // 300 # Max 300 Hz
        max_period = sr // 50  # Min 50 Hz
        
        corr = np.correlate(y, y, mode='full')
        corr = corr[len(corr)//2:]
        valid_corr = corr[min_period:max_period]
        
        median_pitch = sr / (np.argmax(valid_corr) + min_period) if len(valid_corr) > 0 else 0

        # Gender Heuristic
        if 85 <= median_pitch <= 165:
            gender, g_conf = "male", 0.85
        elif 165 < median_pitch <= 255:
            gender, g_conf = "female", 0.82
        else:
            gender, g_conf = "unknown", 0.45

        # --- 2. Spectral Centroid (Pure Math FFT) ---
        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1.0/sr)
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

        if centroid < 1500: age, a_conf = "46-60", 0.65
        elif centroid < 2500: age, a_conf = "31-45", 0.70
        else: age, a_conf = "18-30", 0.60

        return {"gender": gender, "g_conf": g_conf, "age": age, "a_conf": a_conf, "quality": quality}

pipeline = AudioPipeline()

# --- Endpoints ---
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    contact_id: str = Form(default_factory=lambda: str(uuid.uuid4())),
    file: UploadFile = File(...)
):
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = pipeline.predict(tmp_path)
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio")
    finally:
        os.remove(tmp_path) 
    
    return AnalysisResponse(
        contact_id=contact_id,
        gender=Prediction(prediction=results["gender"], confidence=results["g_conf"]),
        age_bracket=Prediction(prediction=results["age"], confidence=results["a_conf"]),
        processing_ms=int((time.time() - start_time) * 1000),
        audio_quality=results["quality"]
    )