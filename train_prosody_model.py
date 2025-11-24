import os
import pathlib
import time
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

TARGET_SR = 16000
N_MFCC = 20
MAX_SEC = 8.0
# Allow limiting per class for faster experimentation
BASE_DIR = pathlib.Path(__file__).resolve().parent
PER_CLASS_CAP = int(os.environ.get("PROSODY_TRAIN_CAP", 1800))
OUT_PATH = os.environ.get(
    "PROSODY_MODEL_OUT",
    str(BASE_DIR / "results" / "presentation_imbalance" / "mfcc_prosody_model.joblib")
)
DATA_DIR = str(BASE_DIR / "IndicAccentDB")
SEED = 13
np.random.seed(SEED)

ACCENT_DIRS = [d for d in os.listdir(DATA_DIR) if (pathlib.Path(DATA_DIR) / d).is_dir()]
print(f"Classes: {ACCENT_DIRS}")


def load_audio(path: str, sr: int = TARGET_SR, max_sec: float = MAX_SEC):
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    try:
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception:
        pass
    try:
        intervals = librosa.effects.split(y, top_db=35)
        if len(intervals) > 0:
            parts = [y[s:e] for s, e in intervals]
            y = np.concatenate(parts) if parts else y
    except Exception:
        pass
    # cap length
    if max_sec and len(y) > int(max_sec * orig_sr):
        y = y[: int(max_sec * orig_sr)]
    if orig_sr != sr:
        import scipy.signal as sig
        y = sig.resample_poly(y, sr, orig_sr)
    return y, sr


def mfcc_prosody_features(y: np.ndarray, sr: int) -> np.ndarray:
    if y.size < 1024:
        y = np.pad(y, (0, 1024 - y.size))
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mf_feat = np.concatenate([mf.mean(axis=1), mf.std(axis=1)])  # 40
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
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
    spec_stats = [float(np.mean(zcr)), float(np.mean(rms)), float(np.std(rms)),
                  float(np.mean(cent)), float(np.std(cent)), float(np.mean(roll)), float(np.std(roll)),
                  float(np.mean(bw)), float(np.std(bw)), f0_med, f0_mean, f0_std]
    return np.concatenate([mf_feat, np.array(spec_stats, dtype=np.float32)])  # 52

features = []
labels = []
start = time.time()
for accent in ACCENT_DIRS:
    accent_path = os.path.join(DATA_DIR, accent)
    files = [f for f in os.listdir(accent_path) if f.lower().endswith('.wav')]
    if PER_CLASS_CAP:
        files = files[:PER_CLASS_CAP]
    print(f"Processing {accent}: {len(files)} files")
    for f in files:
        fp = os.path.join(accent_path, f)
        try:
            y, sr = load_audio(fp)
            feats = mfcc_prosody_features(y, sr)
            if feats.shape[0] != 52:
                continue
            features.append(feats)
            labels.append(accent)
        except Exception:
            continue

X = np.vstack(features)
Y = np.array(labels)
print("Feature matrix", X.shape, "Labels", Y.shape, f"Time {time.time()-start:.1f}s")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Base classifier (Logistic Regression); could switch to MLP if needed
base_clf = LogisticRegression(max_iter=1500, random_state=SEED, n_jobs=None)
# Calibrated for better probabilities
clf = CalibratedClassifierCV(base_clf, cv=3)
clf.fit(X_scaled, Y)

# Quick CV metrics (StratifiedKFold) with existing features (not rigorous, but indicative)
cv_f1 = []
cv_acc = []
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
for tr, va in skf.split(X, Y):
    Xt, Xv = X[tr], X[va]
    Yt, Yv = Y[tr], Y[va]
    Xt_s = scaler.transform(Xt)
    Xv_s = scaler.transform(Xv)
    clf_inner = CalibratedClassifierCV(LogisticRegression(max_iter=1500, random_state=SEED), cv=2)
    clf_inner.fit(Xt_s, Yt)
    pv = clf_inner.predict(Xv_s)
    cv_acc.append(accuracy_score(Yv, pv))
    cv_f1.append(f1_score(Yv, pv, average='macro'))
print(f"CV Accuracy mean={np.mean(cv_acc):.4f} F1_macro mean={np.mean(cv_f1):.4f}")

bundle = {
    'scaler': scaler,
    'clf': clf,
    'feature_dim': 52,
    'classes': clf.classes_,
    'cv_acc_mean': float(np.mean(cv_acc)),
    'cv_f1_macro_mean': float(np.mean(cv_f1)),
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
joblib.dump(bundle, OUT_PATH)
print("Saved 52-dim prosody model bundle ->", OUT_PATH)
