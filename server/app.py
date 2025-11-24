import os
import tempfile
import pathlib
import uuid
import time
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import librosa
import scipy.signal as sig
from flask import Flask, request, jsonify, render_template, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pythonjsonlogger import jsonlogger
from joblib import load

# Configuration
TARGET_SR = 16000
N_MFCC = 20
# Chunked inference to improve accent sensitivity on longer audio
CHUNK_SEC = float(os.environ.get("CHUNK_SEC", 1.5))
CHUNK_HOP = float(os.environ.get("CHUNK_HOP", 0.75))
MIN_CHUNKS = int(os.environ.get("MIN_CHUNKS", 2))
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # project root

def _resolve_model_path() -> str:
    env_path = os.environ.get("MFCC_MODEL_PATH", "").strip()
    if env_path:
        p = pathlib.Path(env_path)
        if p.is_file():
            return str(p)
    # Fallback candidates (relative to BASE_DIR)
    candidates = [
        BASE_DIR / "results" / "presentation_imbalance" / "mfcc_prosody_model.joblib",
        BASE_DIR / "results" / "presentation_imbalance" / "mfcc_model_targeted_aug.joblib",
        BASE_DIR / "results" / "presentation_imbalance" / "mfcc_model.joblib",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError(
        "Model bundle not found. Set MFCC_MODEL_PATH to a valid joblib file."
    )

try:
    MODEL_PATH = _resolve_model_path()
except Exception as e:
    # Defer error until first /predict call rather than crash import on some platforms
    MODEL_PATH = str(BASE_DIR / "results" / "presentation_imbalance" / "_missing_model.joblib")

# Test-Time Augmentation / Logging configuration
TTA_ENABLED = int(os.environ.get("TTA_ENABLED", 1))  # 1 to enable TTA averaging
TTA_MAX_VARIANTS = int(os.environ.get("TTA_MAX_VARIANTS", 6))  # number of augmented variants (excluding original)
TTA_PROB_THRESH = float(os.environ.get("TTA_PROB_THRESH", 0.94))  # only apply TTA if max prob below threshold
TTA_USE_CHUNKS = int(os.environ.get("TTA_USE_CHUNKS", 0))  # if 1: run TTA on chunked segments instead of full audio
DEBUG_LOG = int(os.environ.get("DEBUG_FEATURE_LOG", 1))  # if 1: write debug feature/prob logs
DEBUG_DIR = BASE_DIR / "results" / "debug_features"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Input validation thresholds
MIN_DURATION_SEC = float(os.environ.get("MIN_DURATION_SEC", 1.5))  # minimum usable duration after trimming
MIN_RMS = float(os.environ.get("MIN_RMS", 0.004))  # minimum RMS energy threshold (post-normalization)
CALIB_TEMPERATURE = float(os.environ.get("CALIB_TEMPERATURE", 1.0))  # temperature scaling for probabilities

UPLOAD_DIR = os.environ.get("UPLOAD_DIR") or str(BASE_DIR / "uploads")
pathlib.Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload cap

# ---- Rate Limiting ----
limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute", "1000 per hour"])

# ---- JSON Logging ----
root_logger = logging.getLogger()
if not root_logger.handlers:
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

@app.before_request
def _inject_request_id():
    g.request_id = uuid.uuid4().hex
    request.start_time = time.time()

@app.after_request
def _after_request(resp):
    rid = getattr(g, 'request_id', None)
    if rid:
        resp.headers['X-Request-ID'] = rid
    duration = (time.time() - getattr(request, 'start_time', time.time())) * 1000.0
    app.logger.info("request_complete", extra={
        'request_id': rid,
        'path': request.path,
        'method': request.method,
        'status': resp.status_code,
        'duration_ms': round(duration, 2)
    })
    return resp


# -------- Audio + Features ---------

def _normalize_rms(y: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Normalize audio to target RMS level in dBFS; avoid exploding near-silence."""
    eps = 1e-9
    rms = np.sqrt(np.mean(y**2) + eps)
    curr_db = 20.0 * np.log10(max(rms, eps))
    gain_db = target_dbfs - curr_db
    gain = 10.0 ** (gain_db / 20.0)
    # clamp overly large gain to avoid amplifying silence
    gain = float(np.clip(gain, 0.1, 10.0))
    y = y * gain
    # hard clip
    y = np.clip(y, -1.0, 1.0)
    return y


def load_audio(file_path: str, sr: int = TARGET_SR, max_sec: float = 8.0) -> Tuple[np.ndarray, int]:
    """Load audio mono, trim silence, normalize loudness, resample, cap length (cross-platform)."""
    y, orig_sr = librosa.load(file_path, sr=None, mono=True)
    # trim leading/trailing silence
    try:
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception:
        pass
    # optional voice-activity-like splitting (keep voiced segments)
    try:
        intervals = librosa.effects.split(y, top_db=35)
        if len(intervals) > 0:
            parts = [y[s:e] for s, e in intervals]
            y = np.concatenate(parts) if len(parts) > 0 else y
    except Exception:
        pass
    # normalize RMS
    y = _normalize_rms(y, -20.0)
    # cap to max length at original SR first
    if max_sec is not None and orig_sr and len(y) > int(max_sec * orig_sr):
        y = y[: int(max_sec * orig_sr)]
    # resample if needed
    if orig_sr != sr:
        y = sig.resample_poly(y, sr, orig_sr)
    # min duration check after processing
    if len(y) < int(0.5 * sr):
        # too short after trimming; return as-is and let caller handle
        return y, sr
    return y, sr


def _mfcc_features_from_signal(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """MFCC-only feature vector from an audio array (20 mean + 20 std = 40 dims)."""
    if y.size < 1024:
        y = np.pad(y, (0, max(0, 1024 - y.size)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    return np.concatenate([mean, std])


def extract_mfcc(file_path: str, sr: int = TARGET_SR, n_mfcc: int = N_MFCC, max_sec: float = 8.0) -> np.ndarray:
    """MFCC-only feature vector (20 mean + 20 std = 40 dims)."""
    y, sr = load_audio(file_path, sr=sr, max_sec=max_sec)
    return _mfcc_features_from_signal(y, sr, n_mfcc)


def _mfcc_prosody_features_from_signal(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """MFCC (40) + prosody/spectral stats (12) + zcr (1) + f0 stats (3) = 52 dims from an audio array.

    Prosody/spectral: rms mean/std, centroid mean/std, rolloff mean/std, bandwidth mean/std.
    """
    # Ensure minimum length
    if y.size < 1024:
        y = np.pad(y, (0, max(0, 1024 - y.size)))

    # MFCC base (40)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mf_feat = np.concatenate([mf.mean(axis=1), mf.std(axis=1)])  # (40,)

    # Prosody/spectral
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # f0 via YIN (robust to NaNs)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[np.isfinite(f0)]
        if f0.size == 0:
            f0_med = f0_mean = f0_std = 0.0
        else:
            f0_med = float(np.median(f0))
            f0_mean = float(np.mean(f0))
            f0_std = float(np.std(f0))
    except Exception:
        f0_med = f0_mean = f0_std = 0.0

    spec_stats = [
        float(np.mean(zcr)),
        float(np.mean(rms)), float(np.std(rms)),
        float(np.mean(cent)), float(np.std(cent)),
        float(np.mean(roll)), float(np.std(roll)),
        float(np.mean(bw)), float(np.std(bw)),
        f0_med, f0_mean, f0_std,
    ]

    return np.concatenate([mf_feat, np.array(spec_stats, dtype=np.float32)])  # (52,)


def extract_mfcc_prosody(file_path: str, sr: int = TARGET_SR, n_mfcc: int = N_MFCC, max_sec: float = 8.0) -> np.ndarray:
    y, sr = load_audio(file_path, sr=sr, max_sec=max_sec)
    return _mfcc_prosody_features_from_signal(y, sr, n_mfcc)


def _generate_tta_variants(y: np.ndarray, sr: int) -> List[np.ndarray]:
    """Generate lightweight augmented variants of the signal.

    Augmentations kept subtle to preserve accent cues while reducing device/noise shift.
    Order: pitch shifts, time stretch, small noise, slight gain perturbations.
    """
    variants: List[np.ndarray] = []
    if y.size == 0:
        return variants
    # Safeguard: librosa effects may raise on very short clips; catch and skip.
    def safe(fn):
        try:
            v = fn()
            if isinstance(v, np.ndarray) and v.size > 0:
                variants.append(np.clip(v, -1.0, 1.0))
        except Exception:
            pass

    # Pitch shifts +/- 1 semitone
    safe(lambda: librosa.effects.pitch_shift(y, sr=sr, n_steps=1))
    safe(lambda: librosa.effects.pitch_shift(y, sr=sr, n_steps=-1))
    # Time stretch slight
    safe(lambda: librosa.effects.time_stretch(y, rate=0.97))
    safe(lambda: librosa.effects.time_stretch(y, rate=1.03))
    # Add low-level noise
    noise_scale = 0.003
    variants.append(np.clip(y + noise_scale * np.random.randn(len(y)), -1.0, 1.0))
    # Slight gain perturbation
    variants.append(np.clip(y * 1.05, -1.0, 1.0))
    # Cap total variants
    if len(variants) > TTA_MAX_VARIANTS:
        variants = variants[:TTA_MAX_VARIANTS]
    return variants


def _apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to a probability vector in probability space.

    Because we do not have raw logits, we approximate by raising probabilities to 1/T.
    This preserves ordering but adjusts confidence calibration. T>1 makes distribution softer.
    """
    if T <= 0:
        return proba
    if abs(T - 1.0) < 1e-6:
        return proba
    scaled = np.power(np.clip(proba, 1e-12, 1.0), 1.0 / T)
    denom = np.sum(scaled)
    if denom <= 0:
        return proba
    return scaled / denom


# -------- Model Loading ---------

_model_bundle: Optional[Dict] = None


def load_model_bundle() -> Optional[Dict]:
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle
    if not os.path.isfile(MODEL_PATH):
        app.logger.warning("MFCC model bundle not found at %s", MODEL_PATH)
        return None
    try:
        _model_bundle = load(MODEL_PATH)
        if not {"scaler", "clf"}.issubset(_model_bundle.keys()):
            app.logger.error("Model bundle missing required keys: %s", _model_bundle.keys())
            _model_bundle = None
        else:
            app.logger.info("Loaded MFCC model bundle from %s", MODEL_PATH)
    except Exception as e:
        app.logger.exception("Failed to load model bundle: %s", e)
        _model_bundle = None
    return _model_bundle


# -------- Food Suggestions ---------

# Normalize label -> canonical key
_alias_map = {
    # languages to states
    "malayalam": "kerala",
    "kannada": "karnataka",
    "tamil": "tamil",
    "telugu": "andhra_pradesh",
    "hindi": "north_india",
    # common variants
    "gujarat": "gujarat",
    "gujrat": "gujarat",
    "andhra": "andhra_pradesh",
    "andhra pradesh": "andhra_pradesh",
    "tamil nadu": "tamil",
}

_food_map: Dict[str, List[str]] = {
    "andhra_pradesh": [
        "Hyderabadi Biryani",
        "Gongura Pachadi",
        "Pesarattu",
        "Pulihora",
    ],
    "karnataka": [
        "Bisi Bele Bath",
        "Mysore Masala Dosa",
        "Ragi Mudde",
        "Maddur Vada",
    ],
    "kerala": [
        "Appam & Stew",
        "Puttu Kadala",
        "Kerala Parotta",
        "Fish Moilee",
    ],
    "tamil": [
        "Idli, Dosa, Vada",
        "Sambar Rice",
        "Pongal",
        "Chettinad Chicken",
    ],
    "gujarat": [
        "Khaman Dhokla",
        "Thepla",
        "Undhiyu",
        "Fafda-Jalebi",
    ],
    "jharkhand": [
        "Dhuska",
        "Thekua",
        "Rugra Curry",
        "Pitha",
    ],
    "north_india": [
        "Chole Bhature",
        "Rajma Chawal",
        "Paratha",
        "Paneer Tikka",
    ],
    # fallback for international accents
    "american": ["Burgers", "BBQ", "Mac & Cheese"],
    "australian": ["Meat Pie", "Lamingtons", "Sausage Sizzle"],
}


def normalize_label(lbl: str) -> str:
    s = (lbl or "").strip().lower().replace("_", " ")
    return _alias_map.get(s, s.replace(" ", "_"))


def get_food_suggestions(accent: str) -> List[str]:
    key = normalize_label(accent)
    # try exact
    if key in _food_map:
        return _food_map[key]
    # try stripping trailing words
    parts = key.split("_")
    while parts:
        k = "_".join(parts)
        if k in _food_map:
            return _food_map[k]
        parts = parts[:-1]
    # default fallback
    return ["Biryani", "Masala Dosa", "Chole Bhature"]


# -------- Routes ---------

@app.route("/", methods=["GET"])  # simple upload form
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])  # multipart/form-data with 'file'
@limiter.limit("10 per minute")
def predict():
    bundle = load_model_bundle()
    if bundle is None:
        return jsonify({"error": "model_not_loaded", "detail": f"Expected at {MODEL_PATH}"}), 503

    if "file" not in request.files:
        return jsonify({"error": "missing_file", "detail": "Upload field name must be 'file'"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty_filename"}), 400

    # Persist to a temp file to ensure librosa can open it reliably
    suffix = os.path.splitext(f.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix, dir=UPLOAD_DIR) as tmp:
        f.save(tmp.name)
        # Load once and run chunked feature extraction for improved sensitivity
        try:
            y, sr = load_audio(tmp.name, sr=TARGET_SR, max_sec=8.0)
        except Exception as e:
            app.logger.exception("Audio load failed: %s", e)
            return jsonify({"error": "audio_load_failed", "detail": str(e)}), 400

        # Input validation: duration & energy
        duration = len(y) / float(sr) if sr else 0.0
        rms = float(np.sqrt(np.mean(y ** 2) + 1e-9))
        if duration < MIN_DURATION_SEC:
            return jsonify({"error": "too_short", "detail": f"Duration {duration:.2f}s < {MIN_DURATION_SEC}s"}), 400
        if rms < MIN_RMS:
            return jsonify({"error": "too_quiet", "detail": f"RMS {rms:.4f} < {MIN_RMS}"}), 400

        # Full-sample features
        full_52 = None
        full_40 = None
        try:
            full_52 = _mfcc_prosody_features_from_signal(y, sr)
        except Exception as e:
            app.logger.warning("Full MFCC+prosody failed: %s", e)
        try:
            full_40 = _mfcc_features_from_signal(y, sr)
        except Exception as e:
            if full_52 is None:
                app.logger.exception("Full MFCC failed and no 52-dim fallback: %s", e)
                return jsonify({"error": "feature_extraction_failed", "detail": str(e)}), 400

        # Chunked features
        feats_52_list = []
        feats_40_list = []
        try:
            if y.size >= int(0.5 * sr) and CHUNK_SEC > 0:
                win = int(CHUNK_SEC * sr)
                hop = int(CHUNK_HOP * sr)
                if hop <= 0:
                    hop = win
                for start in range(0, max(1, y.size - win + 1), hop):
                    seg = y[start:start + win]
                    if seg.size < int(0.4 * win):
                        continue
                    # Skip near-silent chunks
                    if np.sqrt(np.mean(seg ** 2) + 1e-9) < 1e-3:
                        continue
                    try:
                        feats_52_list.append(_mfcc_prosody_features_from_signal(seg, sr))
                    except Exception:
                        pass
                    try:
                        feats_40_list.append(_mfcc_features_from_signal(seg, sr))
                    except Exception:
                        pass
        except Exception as e:
            app.logger.warning("Chunked feature extraction issue: %s", e)

        extracted = {
            "full_52": full_52,
            "full_40": full_40,
            "chunks_52": np.vstack(feats_52_list) if len(feats_52_list) else None,
            "chunks_40": np.vstack(feats_40_list) if len(feats_40_list) else None,
            "raw_audio": y,
            "sr": sr,
        }

    scaler = bundle["scaler"]
    clf = bundle["clf"]
    # Choose features based on scaler expected dimension, then transform (with fallback)
    # Prefer chunked features when enough chunks are present
    chunks_52 = extracted.get("chunks_52")
    chunks_40 = extracted.get("chunks_40")
    feat_52 = extracted.get("full_52")
    feat_40 = extracted.get("full_40")
    # infer expected feature count from scaler
    try:
        expected = getattr(scaler, 'n_features_in_', None)
        if expected is None and hasattr(scaler, 'mean_'):
            expected = len(getattr(scaler, 'mean_'))
    except Exception:
        expected = None

    use_matrix = None
    if expected == 52 and chunks_52 is not None and chunks_52.shape[0] >= MIN_CHUNKS:
        use_matrix = chunks_52
    elif expected == 40 and chunks_40 is not None and chunks_40.shape[0] >= MIN_CHUNKS:
        use_matrix = chunks_40
    elif expected == 52 and feat_52 is not None:
        use_matrix = feat_52.reshape(1, -1)
    elif expected == 40 and feat_40 is not None:
        use_matrix = feat_40.reshape(1, -1)
    else:
        # fallback: pick whichever full feature exists
        if feat_52 is not None:
            use_matrix = feat_52.reshape(1, -1)
        elif feat_40 is not None:
            use_matrix = feat_40.reshape(1, -1)
        elif chunks_52 is not None:
            use_matrix = chunks_52
        else:
            use_matrix = chunks_40

    # Transform base features
    try:
        X_base = scaler.transform(use_matrix)
    except Exception as e:
        # Fallback attempts
        try:
            if use_matrix is chunks_52 and chunks_40 is not None:
                X_base = scaler.transform(chunks_40)
                use_matrix = chunks_40
            elif use_matrix is chunks_40 and chunks_52 is not None:
                X_base = scaler.transform(chunks_52)
                use_matrix = chunks_52
            elif use_matrix.shape[1] == 52 and feat_40 is not None:
                X_base = scaler.transform(feat_40.reshape(1, -1))
                use_matrix = feat_40.reshape(1, -1)
            elif use_matrix.shape[1] == 40 and feat_52 is not None:
                X_base = scaler.transform(feat_52.reshape(1, -1))
                use_matrix = feat_52.reshape(1, -1)
            else:
                raise e
        except Exception as e2:
            app.logger.exception("Scaler transform failed for available features: %s / %s", e, e2)
            return jsonify({"error": "scaler_transform_failed", "detail": str(e2)}), 500

    # Base probabilities (chunk averaging if multiple rows)
    def _predict_proba(matrix: np.ndarray) -> np.ndarray:
        pm = clf.predict_proba(matrix)
        if pm.ndim == 2 and pm.shape[0] > 1:
            return pm.mean(axis=0)
        return pm[0]

    base_proba = _predict_proba(X_base)
    classes = clf.classes_.tolist()
    base_max = float(np.max(base_proba))

    # Decide if we apply TTA
    apply_tta = bool(TTA_ENABLED) and base_max < TTA_PROB_THRESH
    tta_prob_list = []
    if apply_tta:
        y_raw = extracted.get("raw_audio")
        sr = extracted.get("sr", TARGET_SR)
        if y_raw is not None and y_raw.size > 0:
            variants = _generate_tta_variants(y_raw, sr)
            # Feature extraction per variant
            for v in variants:
                try:
                    if expected == 52:
                        fv = _mfcc_prosody_features_from_signal(v, sr).reshape(1, -1)
                    elif expected == 40:
                        fv = _mfcc_features_from_signal(v, sr).reshape(1, -1)
                    else:
                        # try 52 then 40
                        try:
                            fv = _mfcc_prosody_features_from_signal(v, sr).reshape(1, -1)
                        except Exception:
                            fv = _mfcc_features_from_signal(v, sr).reshape(1, -1)
                    Xt = scaler.transform(fv)
                    tta_prob_list.append(_predict_proba(Xt))
                except Exception:
                    continue
    if tta_prob_list:
        # include base probability in averaging
        all_probs = np.vstack([base_proba] + tta_prob_list)
        proba = all_probs.mean(axis=0)
        proba_source = "tta"
    else:
        proba = base_proba
        proba_source = "base"

    # Temperature scaling (post TTA/base probability aggregation)
    if CALIB_TEMPERATURE and CALIB_TEMPERATURE > 0 and abs(CALIB_TEMPERATURE - 1.0) > 1e-6:
        proba = _apply_temperature(proba, CALIB_TEMPERATURE)

    try:
        idx = int(np.argmax(proba))
        pred = classes[idx]
        order = np.argsort(proba)[::-1]
        top3 = [{"label": classes[i], "prob": float(proba[i])} for i in order[:3]]
    except Exception:
        # Some solvers may not expose predict_proba if not enabled
        pred = clf.predict(X_base)[0]
        top3 = None

    foods = get_food_suggestions(str(pred))
    response = {
        "predicted_accent": str(pred),
        "top3": top3,
        "suggested_foods": foods,
        "prob_source": proba_source,
        "max_prob": float(np.max(proba)),
        "tta_applied": bool(tta_prob_list),
        "temperature_used": CALIB_TEMPERATURE,
        "duration_sec": duration,
        "rms_energy": rms,
    }

    # Debug logging of feature/probabilities if enabled or if prediction uncertain
    if DEBUG_LOG or response["max_prob"] < 0.80:
        try:
            ts = f"{int(np.floor(np.random.rand()*1e9))}"  # pseudo unique
            log_path = os.path.join(DEBUG_DIR, f"pred_{ts}.npz")
            np.savez_compressed(
                log_path,
                proba=proba,
                classes=np.array(classes),
                top3=np.array(top3 if top3 else []),
                pred=str(pred),
                source=proba_source,
                base_proba=base_proba,
                tta_count=len(tta_prob_list),
            )
        except Exception:
            pass
    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    """Simple health/status endpoint to verify server readiness."""
    bundle_loaded = os.path.isfile(MODEL_PATH)
    return jsonify({
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_present": bundle_loaded
    })

@app.route("/version", methods=["GET"])
def version():
    bundle = load_model_bundle()
    meta = {}
    if bundle:
        try:
            classes = bundle.get('classes') or getattr(bundle.get('clf'), 'classes_', [])
            meta['classes'] = list(map(str, classes))
            meta['n_classes'] = len(classes)
        except Exception:
            meta['classes'] = None
        meta['keys'] = list(bundle.keys())
        meta['feature_dim'] = bundle.get('feature_dim') or getattr(bundle.get('scaler'), 'n_features_in_', None)
        meta['cv_acc_mean'] = bundle.get('cv_acc_mean')
        meta['cv_f1_macro_mean'] = bundle.get('cv_f1_macro_mean')
    return jsonify({
        'status': 'ok',
        'model_path': MODEL_PATH,
        'model_loaded': bool(bundle),
        'metadata': meta,
        'chunk_sec': CHUNK_SEC,
        'chunk_hop': CHUNK_HOP,
        'min_chunks': MIN_CHUNKS,
        'temperature': CALIB_TEMPERATURE
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Run without the reloader/debugger when started from automation to avoid restart races
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
