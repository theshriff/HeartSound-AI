
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
from scripts.predict import predict_audio
import tempfile
import os

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

SITE_ORIGIN = "https://theshriff.github.io"


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = SITE_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
def home():
    return jsonify({"status": "HeartSound-AI API is running"})


@app.route("/health")
def health():
    return jsonify({"status": "online"})


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["file"]

    if not audio_file.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only WAV files are supported"}), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:
            temp_path = temp_file.name
            audio_file.save(temp_path)

        result = predict_audio(temp_path)

        return jsonify(result)

    except Exception as error:
        return jsonify({"error": str(error)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run()
