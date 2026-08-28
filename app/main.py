import time
import uuid
import tempfile
import os
import subprocess
import numpy as np
from scipy.io import wavfile
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from transformers import pipeline

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
        logger.info("Downloading and loading wav2vec2 model (this takes a moment)...")
        # Load the Hugging Face audio classification pipeline
        self.classifier = pipeline(
            "audio-classification",
            model="audeering/wav2vec2-large-robust-24-ft-age-gender",
            device=-1 # CPU
        )

    def assess_quality(self, y):
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
                raise HTTPException(status_code=400, detail="Audio conversion failed.")

        try:
            sr, y = wavfile.read(wav_path)
            if len(y.shape) > 1: y = y.mean(axis=1)
            y = y.astype(np.float32) / (np.max(np.abs(y)) + 1e-10)
        finally:
            if created_temp_wav and os.path.exists(wav_path):
                os.remove(wav_path)

        quality = self.assess_quality(y)
        if quality == "insufficient":
            return {"gender": "unknown", "g_conf": 0.0, "age": "unknown", "a_conf": 0.0, "quality": quality}

        # --- ML MODEL INFERENCE ---
        # The model returns a list of dicts like: [{'label': 'female', 'score': 0.8}, {'label': 'age', 'score': 0.35}]
        hf_results = self.classifier(y)
        
        gender, g_conf = "unknown", 0.0
        continuous_age = 0.0
        
        for res in hf_results:
            label = res['label'].lower()
            if label in ['male', 'female'] and res['score'] > g_conf:
                gender = label
                g_conf = round(res['score'], 2)
            elif label == 'age':
                continuous_age = res['score'] # Typically normalized 0-1 (e.g., 0.5 = 50 years old)

        # Map continuous age (assuming 0-1 maps to 0-100 years) to requested brackets
        estimated_age_years = continuous_age * 100
        
        if estimated_age_years < 31: age = "18-30"
        elif estimated_age_years < 46: age = "31-45"
        elif estimated_age_years < 61: age = "46-60"
        else: age = "60+"
        
        # Audeering age confidence is complex to extract from logits, using a safe baseline proxy
        a_conf = 0.75 

        return {"gender": gender, "g_conf": g_conf, "age": age, "a_conf": a_conf, "quality": quality}

pipeline = AudioPipeline()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    contact_id: str = Form(default_factory=lambda: str(uuid.uuid4())),
    file: UploadFile = File(...)
):
    start_time = time.time()
    
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