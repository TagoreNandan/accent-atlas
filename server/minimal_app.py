import os
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.environ.get("MFCC_MODEL_PATH", "/app/results/presentation_imbalance/mfcc_prosody_model.joblib")
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("COMMIT_SHA") or "unknown"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "build_commit": BUILD_COMMIT,
        "model_path": MODEL_PATH,
        "model_present": os.path.isfile(MODEL_PATH)
    })

@app.route("/predict", methods=["POST"])
def predict():
    # Minimal placeholder: just returns duration (fake) to confirm request path works.
    if "file" not in request.files:
        return jsonify({"error": "missing_file"}), 400
    f = request.files["file"]
    data = f.read()
    size = len(data)
    # Fake feature -> deterministic but simple output
    rng = np.random.default_rng(12345)
    probs = rng.random(3)
    probs = probs / probs.sum()
    labels = ["andhra_pradesh", "tamil", "kerala"]
    top3 = [{"label": labels[i], "prob": float(probs[i])} for i in range(3)]
    return jsonify({
        "status": "ok",
        "bytes_received": size,
        "top3_dummy": top3,
        "build_commit": BUILD_COMMIT
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
