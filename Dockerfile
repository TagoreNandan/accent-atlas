###########
# Accent Atlas Production Dockerfile (Multi-stage)
# Smaller final image; gunicorn WSGI server; runtime-only deps.
###########

FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5050 \
    CHUNK_SEC=0.8 \
    CHUNK_HOP=0.4 \
    MIN_CHUNKS=1 \
    CALIB_TEMPERATURE=1.3 \
    MIN_DURATION_SEC=1.2 \
    MIN_RMS=0.003
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# Copy application source
COPY . .

EXPOSE 5050

# Use gunicorn for production (Render overrides startCommand if using render.yaml)
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers=2", "server.app:app"]
