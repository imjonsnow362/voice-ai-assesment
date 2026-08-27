from fastapi.testclient import TestClient
from app.main import app
import numpy as np
import soundfile as sf
import io

client = TestClient(app)

def test_analyze_audio():
    # Generate 1 sec of dummy audio (sine wave) simulating voice
    sr = 16000
    t = np.linspace(0, 1, sr)
    y = np.sin(2 * np.pi * 120 * t) # 120Hz tone (Male pitch range)
    
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    
    response = client.post(
        "/analyze",
        data={"contact_id": "integration-test"},
        files={"file": ("test.wav", buf, "audio/wav")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["contact_id"] == "integration-test"
    assert data["gender"]["prediction"] == "male"
    assert "audio_quality" in data