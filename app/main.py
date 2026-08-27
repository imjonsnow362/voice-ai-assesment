import time
import uuid
import tempfile
import os
import librosa
import numpy as np
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
        # In production, we'd preload ONNX-quantized models (e.g., wav2vec2) here.
        # Using a mathematical feature-extraction approach here to guarantee <500ms CPU execution.
        logger.info("Initializing Audio Pipeline...")

    def assess_quality(self, y):
        """Calculates Signal-to-Noise Ratio (SNR) to handle logistics truck/road noise."""
        signal_power = np.mean(y**2)
        noise_power = np.percentile(y**2, 10) # Assume 10th percentile is background noise floor
        if noise_power == 0: noise_power = 1e-10
        snr = 10 * np.log10(signal_power / noise_power)
        
        if snr < 10: return "insufficient"
        elif snr < 20: return "degraded"
        return "good"

    def predict(self, file_path: str):
        # Resample to 16kHz (Standard for voice AI)
        y, sr = librosa.load(file_path, sr=16000)
        
        quality = self.assess_quality(y)
        if quality == "insufficient":
            return {"gender": "unknown", "g_conf": 0.0, "age": "unknown", "a_conf": 0.0, "quality": quality}

        # Extract pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]
        median_pitch = np.median(pitches) if len(pitches) > 0 else 0

        # Gender Heuristic based on human vocal fold frequencies
        if 85 <= median_pitch <= 165:
            gender, g_conf = "male", 0.85
        elif 165 < median_pitch <= 255:
            gender, g_conf = "female", 0.82
        else:
            gender, g_conf = "unknown", 0.45

        # Age Heuristic using spectral centroid proxy
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
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
    
    # PRIVACY CONSTRAINT: Save to temp file, process, and permanently delete immediately.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = pipeline.predict(tmp_path)
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid or corrupted audio file")
    finally:
        os.remove(tmp_path) # Assures PII is not stored on disk
    
    processing_ms = int((time.time() - start_time) * 1000)
    
    return AnalysisResponse(
        contact_id=contact_id,
        gender=Prediction(prediction=results["gender"], confidence=results["g_conf"]),
        age_bracket=Prediction(prediction=results["age"], confidence=results["a_conf"]),
        processing_ms=processing_ms,
        audio_quality=results["quality"]
    )

# BONUS: WebSocket Endpoint for Real-Time Chunks
@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            start = time.time()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
                
            try:
                results = pipeline.predict(tmp_path)
                await websocket.send_json({
                    "gender": results["gender"],
                    "audio_quality": results["quality"],
                    "processing_ms": int((time.time() - start) * 1000)
                })
            finally:
                os.remove(tmp_path)
    except Exception:
        logger.info("WebSocket closed")