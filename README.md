# Logistics Voice AI API

A real-time, low-latency audio processing API designed for logistics and dispatch call centers. It ingests audio streams, assesses background noise (SNR), and infers the caller's gender and age bracket.

## 🚀 Setup & Execution

### Option 1: Docker (Recommended for Reviewers)
The service is fully containerized and requires no external dependencies.
```bash
docker compose up --build
```
The API will be available at `http://localhost:8000`.

### Option 2: Local / Native
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Ensure ffmpeg is installed natively (e.g., brew install ffmpeg on macOS)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🧠 Design Decisions & Model Rationale

* **Model Choice (Pretrained vs. Heuristics):** I implemented the `audeering/wav2vec2-large-robust-21-ft-age-gender` model via Hugging Face. While mathematical heuristics (FFT/Autocorrelation) offer sub-50ms CPU latency, they struggle with real-world logistics noise. This specific `wav2vec2` model was explicitly fine-tuned on noisy speech datasets (like MSP-Podcast), making it highly resilient to truck and warehouse background noise. It outputs continuous age and gender logits, providing a massive accuracy upgrade over hand-coded thresholds.
* **Logistics Noise Handling:** Before inference, the pipeline calculates the Signal-to-Noise Ratio (SNR) by comparing signal power against a baseline noise floor. If the SNR drops below acceptable thresholds, it safely flags the `audio_quality` as `degraded` or `insufficient` rather than forcing the model to hallucinate predictions on pure static.
* **Privacy & PII (Strict Compliance):** Caller audio is treated as highly sensitive PII. The audio chunk is temporarily saved to an OS temp file via FastAPI's `UploadFile`, processed, and immediately permanently deleted in a `finally` block before the HTTP response is dispatched. Zero data persistence occurs on disk.

## 🏗️ Architecture & Scaling Strategy (1,000 Concurrent Calls)

To scale this service to handle 1,000 concurrent inbound logistics calls, the current monolithic REST architecture must be decoupled. My production architecture would look like this:

1. **Ingestion Layer (Web Tier):** FastAPI WebSocket servers load-balanced via Nginx/ALB. They do not process audio; they only accept chunks and publish them to a message broker.
2. **Message Broker:** Apache Kafka or Redis Streams to queue the incoming audio chunks safely.
3. **Inference Layer (Worker Tier):** A pool of GPU-backed workers (e.g., Celery or Faust). These workers pull from Kafka, run the heavy machine learning inference (e.g., PyTorch/ONNX models), and push results back to a response queue.
4. **Auto-Scaling:** Utilize **KEDA** (Kubernetes Event-driven Autoscaling) to scale the worker pods horizontally based on the Kafka queue depth, ensuring we never drop calls during traffic spikes.

## ⚠️ Known Limitations

* **Model Loading Overhead:** Because it is a 1.2GB Transformer model, the initial cold-start load takes time. In production, this would be mitigated by pre-warming GPU worker nodes.
* **Codec Conversion Overhead:** The pipeline relies on `ffmpeg` under the hood to transcode non-WAV formats. At high scale, this transcoding step should be moved to a dedicated preprocessing microservice to avoid bottlenecking the inference nodes.

## 📡 API Contract

**POST `/analyze`**
Accepts a `multipart/form-data` upload containing an audio file.

**Sample Response:**
```json
{
  "contact_id": "550e8400-e29b-41d4-a716-446655440000",
  "gender": {
    "prediction": "male",
    "confidence": 0.85
  },
  "age_bracket": {
    "prediction": "31-45",
    "confidence": 0.70
  },
  "processing_ms": 42,
  "audio_quality": "good"
}
```

## 🧪 Testing
Run the automated test suite using pytest:
```bash
pytest tests/
```