"""
Live Demo - Lightweight Adaptive Fusion UIE
Flask web app for real-time underwater image enhancement demo.
"""

import io
import sys
import os
import base64
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

from method.lightweight_adaptive_fusion import LightweightAdaptiveFusion
from evaluation_metrics import compute_uiqm, compute_uciqe, compute_niqe_approx

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

enhancer = LightweightAdaptiveFusion()


def _encode_img(bgr: np.ndarray) -> str:
    """Encode BGR image to base64 JPEG data-URI."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _process_image(img: np.ndarray) -> dict:
    """Core enhancement and metric calculation logic."""
    t0 = time.perf_counter()
    result = enhancer.enhance(img)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    fps = 1000.0 / elapsed_ms

    uiqm_orig = compute_uiqm(img)["uiqm"]
    uciqe_orig = compute_uciqe(img)["uciqe"]
    niqe_orig = compute_niqe_approx(img)["niqe_approx"]

    uiqm_enh = compute_uiqm(result)["uiqm"]
    uciqe_enh = compute_uciqe(result)["uciqe"]
    niqe_enh = compute_niqe_approx(result)["niqe_approx"]

    return dict(
        original=_encode_img(img),
        enhanced=_encode_img(result),
        width=img.shape[1],
        height=img.shape[0],
        elapsed_ms=round(elapsed_ms, 2),
        fps=round(fps, 1),
        metrics=dict(
            uiqm_orig=round(uiqm_orig, 3),
            uciqe_orig=round(uciqe_orig, 3),
            niqe_orig=round(niqe_orig, 4),
            uiqm_enh=round(uiqm_enh, 3),
            uciqe_enh=round(uciqe_enh, 3),
            niqe_enh=round(niqe_enh, 4),
        ),
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sample/<filename>")
def get_sample(filename):
    """Serve raw sample images from the dataset directory."""
    return send_from_directory(os.path.join("dataset", "raw"), filename)


@app.route("/enhance_sample/<filename>", methods=["POST"])
def enhance_sample(filename):
    """Process a pre-existing sample image."""
    path = os.path.join("dataset", "raw", filename)
    if not os.path.exists(path):
        return jsonify(error="Sample image not found"), 404

    img = cv2.imread(path)
    if img is None:
        return jsonify(error="Invalid sample image"), 400

    return jsonify(**_process_image(img))


@app.route("/enhance", methods=["POST"])
def enhance():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image uploaded"), 400

    raw = file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Invalid image"), 400

    return jsonify(**_process_image(img))


if __name__ == "__main__":
    print("\n  Lightweight Adaptive Fusion - Live Demo")
    print("  http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
