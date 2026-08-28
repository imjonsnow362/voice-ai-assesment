FROM python:3.10-slim

# ffmpeg handles compressed codecs (mp3, m4a, etc.)
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run with 1 worker for demo, but easily scalable
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]