from fastapi.testclient import TestClient
from app.main import app
import numpy as np
from scipy.io import wavfile
import io

client = TestClient(app)

def test_analyze_audio():
    sr = 16000
    t = np.linspace(0, 1, sr)
    y = np.sin(2 * np.pi * 120 * t).astype(np.float32) # 120Hz tone
    
    buf = io.BytesIO()
    wavfile.write(buf, sr, y)
    buf.seek(0)
    
    response = client.post(
        "/analyze",
        data={"contact_id": "integration-test"},
        files={"file": ("test.wav", buf, "audio/wav")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["gender"]["prediction"] == "male"