from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoFeatureExtractor, AutoModel

app = Flask(__name__)

ROOT = os.getcwd()
OUTPUT_DIR = os.path.join(ROOT, 'results', 'presentation_imbalance')
MFCC_MODEL_PATH = os.path.join(OUTPUT_DIR, 'mfcc_model.joblib')
HUBERT_MODEL_NAME = 'facebook/hubert-base-ls960'

# Load MFCC model bundle if present
mfcc_bundle = None
if os.path.isfile(MFCC_MODEL_PATH):
    try:
        mfcc_bundle = joblib.load(MFCC_MODEL_PATH)
        print('Loaded MFCC model bundle.')
    except Exception as e:
        print('Failed to load MFCC model bundle:', e)

# Prepare HuBERT for on-the-fly embeddings (optional)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
    hubert_model = AutoModel.from_pretrained(HUBERT_MODEL_NAME).to(DEVICE)
    hubert_model.eval()
except Exception as e:
    feature_extractor = None
    hubert_model = None
    print('HuBERT unavailable:', e)


def load_audio(path, sr=16000, max_sec=8.0):
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if max_sec is not None and orig_sr and len(y) > int(max_sec*orig_sr):
        y = y[:int(max_sec*orig_sr)]
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    return y, sr


def extract_mfcc_vec(path, sr=16000, n_mfcc=20):
    y, sr = load_audio(path, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    return np.concatenate([mean, std])


def extract_hubert_vec(path, target_sr=16000):
    if feature_extractor is None or hubert_model is None:
        return None
    y, sr = load_audio(path, target_sr)
    y = np.asarray(y, dtype=np.float32)
    inputs = feature_extractor(y, sampling_rate=target_sr, return_tensors='pt')
    with torch.no_grad():
        hidden = hubert_model(**{k: v.to(DEVICE) for k, v in inputs.items()}).last_hidden_state
    mean = hidden.mean(dim=1).squeeze().cpu().numpy()
    std = hidden.std(dim=1).squeeze().cpu().numpy()
    return np.concatenate([mean, std])


@app.route('/predict_mfcc', methods=['POST'])
def predict_mfcc():
    if mfcc_bundle is None:
        return jsonify({'error': 'MFCC model not available'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    tmp_path = os.path.join('/tmp', f'upload_{os.getpid()}')
    f.save(tmp_path)
    try:
        vec = extract_mfcc_vec(tmp_path)
        sc = mfcc_bundle['scaler']
        clf = mfcc_bundle['clf']
        X = sc.transform([vec])
        proba = clf.predict_proba(X)[0]
        pred = clf.classes_[np.argmax(proba)]
        return jsonify({'prediction': str(pred), 'probabilities': {str(c): float(p) for c, p in zip(clf.classes_, proba)}})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.route('/predict_hubert', methods=['POST'])
def predict_hubert():
    if feature_extractor is None or hubert_model is None:
        return jsonify({'error': 'HuBERT model not available'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    tmp_path = os.path.join('/tmp', f'upload_{os.getpid()}')
    f.save(tmp_path)
    try:
        vec = extract_hubert_vec(tmp_path)
        if vec is None or mfcc_bundle is None:
            return jsonify({'error': 'Required model not ready'}), 503
        # Reuse MFCC classifier on HuBERT is not valid; in practice you would train a separate classifier.
        # Here we return the embedding length as a health check.
        return jsonify({'hubert_embedding_len': int(len(vec))})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
