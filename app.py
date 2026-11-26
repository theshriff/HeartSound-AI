
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
from scripts.predict import predict_audio
import tempfile
import os

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__)


# Serve frontend
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# Analyze heart sound
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(file.filename)

    if not filename.lower().endswith(".wav"):
        return jsonify({"error": "Only WAV files are supported."}), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            temp_path = temp_file.name
            file.save(temp_path)

        result = predict_audio(temp_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
