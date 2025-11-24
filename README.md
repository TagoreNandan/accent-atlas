<<<<<<< HEAD
# accent-atlas
=======
# Accent Atlas: Indic Regional Accent Detection

A lightweight, CPU-friendly web application that detects Indian regional accents from short speech samples and recommends culturally relevant cuisines.

## Overview

Accent Atlas combines efficient MFCC-based feature extraction with a calibrated MLP classifier to achieve **97.8% accuracy** on six Indian regional accents without requiring GPU infrastructure. The system is privacy-respecting, deployable locally, and integrates accent detection with culturally aware cuisine recommendations.

**Key Features:**
- 🎯 **97.8% accuracy** on balanced test set (40-dim MFCC + MLP)
- ⚡ **100 ms latency** on CPU (single-pass inference)
- 🔒 **Privacy-first**: ephemeral in-memory audio processing
- 🎨 **Clean UI**: drag-and-drop upload, in-browser recording, light/dark theme
- 🍜 **Cuisine recommendations**: accent → region → traditional dishes
- 📦 **Minimal dependencies**: Flask, librosa, scikit-learn (no GPU required)

---

## Quick Start

### Prerequisites

- Python 3.9+
- FFmpeg (for audio processing)

### Installation

**Option 1: pip (Minimal MFCC-only)**
```bash
pip install Flask numpy librosa scikit-learn joblib scipy
```

**Option 2: pip (Full with HuBERT)**
```bash
pip install -r requirements.txt
```

**Option 3: Conda**
```bash
conda env create -f environment.yml
conda activate accent-atlas
```

### Running the Application

```bash
# Set model path
export MFCC_MODEL_PATH="results/presentation_imbalance/mfcc_mlp_calibrated.joblib"
export PORT=5050

# Start Flask server
python server/app.py
```

Open browser: **http://localhost:5050**

---

## API Usage

### POST /predict

Upload audio file and get accent prediction with cuisine suggestions.

**Request:**
```bash
curl -F 'file=@audio.wav' http://localhost:5050/predict
```

**Response:**
```json
# Accent Atlas: Indic Regional Accent Detection

A lightweight, CPU-friendly web application that detects Indian regional accents from short speech samples and recommends culturally relevant cuisines.

## Overview

Accent Atlas combines efficient MFCC-based feature extraction with a calibrated MLP classifier to achieve **97.8% accuracy** on six Indian regional accents without requiring GPU infrastructure. The system is privacy-respecting, deployable locally, and integrates accent detection with culturally aware cuisine recommendations.

**Key Features:**
- 🎯 97.8% accuracy on balanced test set (40-dim MFCC + MLP)
- ⚡ ~100 ms latency on CPU (single-pass inference)
- 🔒 Privacy-first: ephemeral in-memory audio processing
- 🎨 Clean UI: drag-and-drop upload, in-browser recording
- 🍜 Cuisine recommendations: accent → region → traditional dishes
- 📦 Minimal dependencies: Flask, librosa, scikit-learn (no GPU required)

---

## Quick Start

### Prerequisites
- Python 3.9+
- FFmpeg (for audio processing)

### Installation

Option 1: pip (Minimal MFCC-only)
```bash
pip install Flask numpy librosa scikit-learn joblib scipy
```

Option 2: pip (Full)
```bash
pip install -r requirements.txt
```

Option 3: Conda
```bash
conda env create -f environment.yml
conda activate accent-atlas
```

### Running the Application
```bash
export MFCC_MODEL_PATH="results/presentation_imbalance/mfcc_mlp_calibrated.joblib"
export PORT=5050
python server/app.py
```
Open: http://localhost:5050

---

## API Usage

### POST /predict
```bash
curl -F 'file=@audio.wav' http://localhost:5050/predict
```
Response (example):
```json
{
  "predicted_accent": "kerala",
  "top3": [
    {"label": "kerala", "score": 0.9870},
    {"label": "tamil", "score": 0.0095},
    {"label": "karnataka", "score": 0.0035}
  ],
  "suggested_foods": ["Appam", "Stew", "Fish Curry", "Puttu", "Erissery"]
}
```

### GET /health
```bash
curl http://localhost:5050/health
```

---

## Supported Accents & Cuisines
| Accent | Region | Sample Dishes |
|--------|--------|---------------|
| Andhra Pradesh | South-Central | Biryani, Mirchi Ka Salan, Gongura Pickle |
| Gujarat | Western | Dhokla, Fafda, Undhiyu, Thepla |
| Karnataka | Southern | Ragi Mudde, Bisi Bele Bath, Uppittu |
| Kerala | South-Western | Appam, Stew, Fish Curry, Puttu |
| Tamil Nadu | Southern | Dosa, Sambar, Idli, Rasam, Payasam |
| Jharkhand | Eastern | Litti Chikha, Dhuska, Bamboo Shoot Curry |

---

## Project Structure (excerpt)
```
accent-atlas/
├── README.md
├── requirements.txt
├── server/
│   ├── app.py
│   └── templates/index.html
├── results/                # Metrics & trained models
└── models/                 # Alternative model artifacts
```

---

## License
Proprietary / Internal (update if open-sourcing).

## Status
Experimental deployment-ready prototype.
| Karnataka | 0.97 | 0.96 | 0.965 |
| Kerala | 0.99 | 0.99 | 0.990 |
| Tamil Nadu | 0.95 | 0.94 | 0.945 |
| Jharkhand | 0.98 | 0.99 | 0.985 |

---

## Feature Engineering

### MFCC (40-dim)
- 20 Mel-Frequency Cepstral Coefficients
- Mean + std aggregation per coefficient
- Mel-scale warping for perceptual alignment
- Effective for CPU-bound, low-latency deployment

### MFCC+Prosody (52-dim)
- MFCC (40-dim) + pitch + energy statistics
- Modest improvement (~0.6% accuracy gain)
- Trade-off: +30% feature dimension

### HuBERT Embeddings (Exploratory)
- 1,536-dim pooled representation (mean+std)
- Self-supervised pre-trained model (facebook/hubert-base-ls960)
- Layer-wise analysis: mid-layers (4–7) optimal (~93–95% accuracy)
- Use case: future fusion, layer selection

---

## Training & Evaluation

### Dataset

- **Source**: IndicAccentDB (curated subset)
- **Classes**: 6 Indian regional accents
- **Total Samples**: ~1,200 (200 per class, balanced)
- **Split**: 60% train / 20% calibration / 20% test
- **Preprocessing**: 16 kHz mono, optional trim/silence

### Model

- **Baseline**: Logistic Regression (87% accuracy, calibrated)
- **Primary**: MLP (1×64 ReLU, Adam optimizer, 97.8% accuracy, calibrated)
- **Calibration**: Platt scaling improves probability reliability (ECE: 0.156 → 0.042)

### Robustness

| Condition | Accuracy Drop |
|-----------|---------------|
| SNR 20 dB (noise) | -3.6% |
| SNR 10 dB (noise) | -8.3% |
| Short clips (2–5 s) | -4.0% |
| Chunked inference | +0.3% (improved) |

---

## Deployment

### Local Development
```bash
python server/app.py
# Server runs on http://localhost:5050
```

### Docker
```bash
docker-compose up -d
# Access at http://localhost:5050
```

### Production (Gunicorn)
```bash
gunicorn --bind 0.0.0.0:8000 --workers 2 server.app:app
```

### Memory-Constrained Hosting (Render / Free Tiers)
If you see worker SIGKILL / OOM events (500 errors with `Worker was sent SIGKILL` in logs), enable lightweight settings via environment variables to reduce peak RAM:

```bash
LIGHT_MODE=1          # Skips expensive silence splitting and TTA augmentations
YIN_ENABLED=0         # Disables fundamental frequency (f0) extraction (librosa.yin)
TTA_ENABLED=0         # Disables test-time augmentation variants entirely
DEBUG_FEATURE_LOG=0   # Stops saving per-prediction npz debug artifacts
MAX_INFERENCE_SEC=6.0 # Caps processed audio length (default 8.0)
```

Set these in the Render dashboard under Environment Variables, then redeploy. This typically cuts memory consumption by 30–50% for longer clips and avoids OOM kills on small instances.

Recommended minimal config for constrained instances:
```bash
LIGHT_MODE=1
YIN_ENABLED=0
TTA_ENABLED=0
DEBUG_FEATURE_LOG=0
MAX_INFERENCE_SEC=6.0
```

You can re-enable pieces incrementally if stability is confirmed.

### Mobile (TensorFlow Lite)
See `docs/DEPLOYMENT.md` for iOS/Android setup.

---

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Test specific endpoint:
```bash
pytest tests/test_api.py::test_predict_response_schema -v
```

Test with coverage:
```bash
pytest tests/ --cov=models --cov=server
```

---

## Limitations

1. **Age Generalization**: MFCC drops 15–20% on children's speech (ages 6–14); HuBERT generalizes better (+5.6%).
2. **Accent Coverage**: Limited to 6 regional accents; generalization to other varieties untested.
3. **Training Data**: Balanced dataset (~1,200 samples) is modest; larger datasets may improve performance.
4. **Noise**: Tested on synthetic SNR 20–10 dB; real-world field recordings may differ.
5. **Code-Switching**: Limited evaluation on multilingual input (Hindi-English mixes).
6. **Memory Constraints**: On very low-RAM hosts (e.g., free tiers) advanced features (YIN pitch stats, TTA) may trigger worker OOM; use the memory-constrained configuration section above.

---

## Future Work

### High Priority (2–4 weeks)
- [ ] Age normalization for cross-demographic robustness
- [ ] Real-world field testing with background noise
- [ ] Extended accent coverage (Marathi, Punjabi, Odia, Bengali)

### Medium Priority (4–6 weeks)
- [ ] HuBERT layer fusion (expected 98%+ accuracy)
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Uncertainty thresholding for low-confidence predictions

### Long-term (8–12 weeks)
- [ ] Multi-task learning (accent + language ID + emotion)
- [ ] Federated learning for privacy-preserving improvement
- [ ] Cross-linguistic benchmarks

See `docs/FUTURE_ROADMAP.md` for detailed roadmap.

---

## Privacy & Ethics

- **No Persistent Storage**: Audio is processed in-memory and deleted after inference.
- **CPU-Only**: No cloud uploads or GPU processing required.
- **Accent Classification Only**: System does not infer demographic attributes, identity, or sensitive personal information.
- **Ethical Use**: Intended for educational and cultural personalization; avoid stereotyping or discriminatory applications.

---

## Libraries & Dependencies

**Core:**
- Flask 2.3.2 – Web framework
- librosa 0.10.0 – Audio processing
- scikit-learn 1.3.0 – ML models
- NumPy 1.24.3 – Numerical computing
- joblib 1.3.1 – Model serialization

**Optional (HuBERT):**
- PyTorch 2.0.1 – Tensor operations
- Transformers 4.30.2 – Pre-trained models

**Development:**
- pytest 7.4.0 – Testing
- Jupyter 1.0.0 – Notebooks
- matplotlib, seaborn – Visualization

See `requirements.txt` for full list.

---

## References

- **HuBERT**: Huang et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units" (Facebook AI, 2021)
- **MFCC**: Davis & Mermelstein, "Comparison of Parametric Representations for Monosyllabic Word Recognition" (IEEE Trans. ASSP, 1980)
- **Platt Scaling**: Platt et al., "Probabilistic Outputs for Support Vector Machines" (2000)
- **IndicAccentDB**: Regional speech corpus for Indian languages

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -am 'Add my feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

MIT License – see `LICENSE` file for details.

---

## Contact & Support

- **Issues**: Open a GitHub issue for bugs or questions
- **Discussions**: Use GitHub Discussions for feature requests
- **Email**: [Your contact info]

---

## Acknowledgments

- IndicAccentDB for curated speech data
- Hugging Face for HuBERT pre-trained models
- scikit-learn and librosa communities for robust ML/audio libraries

---

**Last Updated**: November 2024
**Version**: 1.0.0
>>>>>>> 7621df4 (Initial deployment)
