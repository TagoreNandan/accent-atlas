# Deployment Guide (Accent Atlas)

This document walks you from zero to a public URL hosting the existing accent model.

## 1. Local Run (Baseline)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MFCC_MODEL_PATH="results/presentation_imbalance/mfcc_prosody_model.joblib"
export PORT=5050
python server/app.py
```
Visit: http://localhost:5050 and test `/health`.

Test prediction:
```bash
curl -F 'file=@sample.wav' http://localhost:5050/predict
```

## 2. Choose Model Artifact
Recommended: `results/presentation_imbalance/mfcc_prosody_model.joblib` (has scaler + clf + metadata). Calibrated variants end with `_calibrated.joblib`.

## 3. Docker (Local)
```bash
docker build -t accent-atlas .
docker run -p 5050:5050 \
  -e MFCC_MODEL_PATH=results/presentation_imbalance/mfcc_prosody_model.joblib \
  accent-atlas
```
Health check:
```bash
curl http://localhost:5050/health
```

## 4. Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial accent atlas"
# Create repo on GitHub first (or use gh CLI)
git remote add origin https://github.com/YOUR_USER/accent-atlas.git
git branch -M main
git push -u origin main
```

## 5. Render Deployment (Managed Hosting)
### Option A: Use `render.yaml`
Render can auto-detect `render.yaml`:
- Create new Web Service → Select repository.
- Enable settings sync (if UI presents option).

`render.yaml` specifies:
- Build: `pip install -r requirements.txt`
- Start: `gunicorn --bind 0.0.0.0:$PORT --workers=2 server.app:app`
- Env: `MFCC_MODEL_PATH=results/presentation_imbalance/mfcc_prosody_model.joblib`
- Health: `/health`

### Option B: Manual Form
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT --workers=2 server.app:app`
- Add env var: `MFCC_MODEL_PATH` = path above

Render assigns a URL like: `https://accent-atlas.onrender.com`

### Rate Limiting & Version
The app now includes:
- Global limits: `60/min`, `1000/hour` per IP (default).
- `/predict` specific limit: `10/min` per IP.
- `/version` endpoint exposing model metadata (classes, feature_dim, cross-val stats if present).

Check:
```bash
curl https://accent-atlas.onrender.com/version | jq
```

## 6. Post-Deploy Verification
```bash
curl https://accent-atlas.onrender.com/health
curl -F 'file=@sample.wav' https://accent-atlas.onrender.com/predict
```
If you get `model_not_loaded`, verify the path exists in the repo and env var matches.

## 7. Custom Domain (Optional)
On Render dashboard:
- Settings → Custom Domains → Add domain (e.g. accentatlas.example.com)
- Update DNS: Add CNAME to Render's provided target.
- Wait for SSL issuance (5–30 min).

## 8. Scaling & Performance
- Workers: Increase `--workers` as traffic grows (2–4 typical). Each worker loads model once.
- Memory: Model is tiny; 512MB instance sufficient.
- Concurrency: Flask + gunicorn handles multiple requests; avoid blocking operations inside requests.

## 9. Logging & Monitoring
- Render dashboard shows stdout/stderr.
- Structured JSON logging active (python-json-logger) with fields: request_id, path, status, duration_ms.
- Sample minimal fallback:
```python
import logging
logging.basicConfig(level=logging.INFO)
```
- Optionally integrate with a service (Logtail, Axiom) by sending logs.

## 10. Security Hardening
- Upload size limit: Enforced at 16 MB.
- Rate limiting: Active (Flask-Limiter). Adjust limits in `server/app.py`.
- Disable debug: Already off.
- Consider adding WAF (Cloudflare) for elevated traffic.

## 11. Updating the Model
1. Train new model locally; produce new `.joblib` file.
2. Replace old file in `results/presentation_imbalance/`.
3. Commit + push.
4. Redeploy (Render auto-deploys if enabled).
5. Verify `/health` shows new path.

## 12. Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| 404 on `/predict` | Wrong URL or service not deployed | Check base URL, redeploy |
| 503 model_not_loaded | Env var path incorrect | Confirm file path + env var name |
| Slow first request | Cold start / model load | Accept initial latency; subsequent fast |
| Libsndfile error | Missing system deps | Ensure Dockerfile includes `libsndfile1` (already) |

## 13. Environment Variables Reference
| Variable | Purpose | Default |
|----------|---------|---------|
| `MFCC_MODEL_PATH` | Joblib model bundle path | (Must set) |
| `PORT` | Server port (Render sets automatically) | 5050 |
| `CHUNK_SEC` | Chunk window for accent robustness | 0.8 |
| `CHUNK_HOP` | Hop size between chunks | 0.4 |
| `MIN_CHUNKS` | Minimum chunks to average | 1 |
| `CALIB_TEMPERATURE` | Probability temperature scaling | 1.3 |

## 14. Next Enhancements
- Add `/version` endpoint returning model metadata.
- Implement file size + duration pre-check before feature extraction.
- Add simple caching for identical audio hashes.

---
**You are deployed when:** `/health` returns `{ "status": "ok" }` on the public URL and `/predict` returns JSON for an uploaded sample.
**You are production-ready when:** `/version` returns model metadata and logs display structured JSON entries with `request_id`.
