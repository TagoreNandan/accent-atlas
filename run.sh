#!/usr/bin/env bash
set -euo pipefail
# Cross-platform (Unix) launcher for Accent Detector
PYTHON=${PYTHON:-python}
PORT=${PORT:-5050}
MODEL_PATH=${MFCC_MODEL_PATH:-"results/presentation_imbalance/mfcc_prosody_model.joblib"}
export MFCC_MODEL_PATH="$MODEL_PATH"
export PORT
export CHUNK_SEC=${CHUNK_SEC:-0.8}
export CHUNK_HOP=${CHUNK_HOP:-0.4}
export MIN_CHUNKS=${MIN_CHUNKS:-1}
export CALIB_TEMPERATURE=${CALIB_TEMPERATURE:-1.3}
export MIN_DURATION_SEC=${MIN_DURATION_SEC:-1.2}
export MIN_RMS=${MIN_RMS:-0.003}

if [ ! -d .venv ]; then
  echo "[run] Creating virtualenv" >&2
  $PYTHON -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

echo "[run] Starting server on port $PORT with model=$MODEL_PATH"
exec python server/app.py
