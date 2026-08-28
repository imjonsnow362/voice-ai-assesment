# Logistics Voice AI API

A real-time, low-latency audio processing API designed for logistics and dispatch call centers. It ingests audio streams, assesses background noise (SNR), and infers the caller's gender and age bracket.

## 🚀 Setup & Execution

### Option 1: Docker (Recommended for Reviewers)
The service is fully containerized and requires no external dependencies.
```bash
docker compose up --build
```
The API will be available at `http://localhost:8000`.

### Option 2: Local / Native (For development on older machines)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Ensure ffmpeg is installed natively (e.g., brew install ffmpeg on macOS)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🧠 Design Decisions & Model Rationale

* **Model Choice (CPU & Latency Optimization):** While foundational models like `SpeechBrain` or `wav2vec2` offer high accuracy, they often exceed 1GB in size and easily breach the 500ms CPU inference SLA. Because this assessment prioritizes reasoning and architecture, I built an acoustic feature extraction pipeline using `librosa`. By tracking fundamental frequencies (pitch) and spectral centroids, we achieve incredibly fast inference (<50ms) entirely on CPU. The `AudioPipeline` class uses the Strategy pattern—in a GPU-enabled production environment, swapping this for an ONNX-quantized HuggingFace model requires updating exactly one function.
* **Logistics Noise Handling:** Logistics calls feature heavy background noise (trucks, wind, road). The pipeline calculates the Signal-to-Noise Ratio (SNR) by comparing signal power against a baseline noise floor. If the SNR drops below acceptable thresholds, it safely flags the `audio_quality` as `degraded` or `insufficient` rather than hallucinating predictions.
* **Privacy & PII (Strict Compliance):** Caller audio is treated as highly sensitive PII. The audio chunk is temporarily saved to an OS temp file via FastAPI's `UploadFile`, processed, and immediately permanently deleted in a `finally` block before the HTTP response is dispatched. Zero data persistence occurs on disk.
* **Real-Time Streaming:** I included a bonus WebSocket endpoint (`/ws/analyze`) to demonstrate how progressive predictions would work for real-time streaming chunks during live calls.

## 🏗️ Architecture & Scaling Strategy (1,000 Concurrent Calls)

To scale this service to handle 1,000 concurrent inbound logistics calls, the current monolithic REST architecture must be decoupled. My production architecture would look like this:

1. **Ingestion Layer (Web Tier):** FastAPI WebSocket servers load-balanced via Nginx/ALB. They do not process audio; they only accept chunks and publish them to a message broker.
2. **Message Broker:** Apache Kafka or Redis Streams to queue the incoming audio chunks safely.
3. **Inference Layer (Worker Tier):** A pool of GPU-backed workers (e.g., Celery or Faust). These workers pull from Kafka, run the heavy machine learning inference (e.g., PyTorch/ONNX models), and push results back to a response queue.
4. **Auto-Scaling:** Utilize **KEDA** (Kubernetes Event-driven Autoscaling) to scale the worker pods horizontally based on the Kafka queue depth, ensuring we never drop calls during traffic spikes.

## ⚠️ Known Limitations

* **Heuristic Accuracy:** The current `librosa` implementation relies on vocal pitch (fundamental frequency) for gender and spectral centroids for age. While lightning-fast, this is a proxy, not true AI. It may misclassify edge cases (e.g., high-pitched male voices, children, or highly distorted phone lines). 
* **Codec Conversion Overhead:** Currently, `librosa` relies on `ffmpeg` under the hood to transcode non-WAV formats. At high scale, this transcoding step should be moved to a dedicated preprocessing microservice to avoid CPU bottlenecking on the inference nodes.

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