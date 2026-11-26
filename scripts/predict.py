
import numpy as np
import pandas as pd
import librosa
import joblib

from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


# 1. Load trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_sound_combined_model.pkl"

model = joblib.load(MODEL_PATH)


# 2. Extract features from audio
def extract_features(audio_path):

    sample_rate, audio = wavfile.read(audio_path)

    if sample_rate <= 800:
        raise ValueError("Sample rate must be greater than 800 Hz.")

    audio = audio.astype(np.float64)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) < 1024:
        raise ValueError("Audio file is too short.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values.")

    # Bandpass filter
    sos = butter(
        4,
        [25, 400],
        btype="bandpass",
        fs=sample_rate,
        output="sos"
    )

    filtered_audio = sosfiltfilt(sos, audio)

    # --------------------------------
    # A. Original 5 audio features
    # --------------------------------

    rms = np.sqrt(np.mean(filtered_audio ** 2))

    zcr = np.sum(
        np.diff(np.signbit(filtered_audio))
    ) / len(filtered_audio)

    fft = np.fft.rfft(filtered_audio)

    frequencies = np.fft.rfftfreq(
        len(filtered_audio),
        d=1 / sample_rate
    )

    power = np.abs(fft) ** 2
    power[0] = 0

    total_power = np.sum(power)

    if total_power == 0:
        raise ValueError("Audio contains no usable signal.")

    spectral_centroid = np.sum(
        frequencies * power
    ) / total_power

    spectral_bandwidth = np.sqrt(
        np.sum(
            ((frequencies - spectral_centroid) ** 2) * power
        ) / total_power
    )

    dominant_frequency = frequencies[np.argmax(power)]

    features = {
        "RMS": rms,
        "ZCR": zcr,
        "Spectral_Centroid": spectral_centroid,
        "Spectral_Bandwidth": spectral_bandwidth,
        "Dominant_Frequency": dominant_frequency
    }

    # --------------------------------
    # B. Extract 26 MFCC features
    # --------------------------------

    mfcc_audio = audio.astype(np.float32)

    max_value = np.max(np.abs(mfcc_audio))

    if max_value > 0:
        mfcc_audio = mfcc_audio / max_value

    mfcc_filtered = sosfiltfilt(
        sos,
        mfcc_audio
    )

    mfcc = librosa.feature.mfcc(
        y=mfcc_filtered.astype(np.float32),
        sr=sample_rate,
        n_mfcc=13,
        n_fft=512,
        hop_length=128
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    for i in range(13):
        features[f"MFCC_{i+1}_mean"] = mfcc_mean[i]
        features[f"MFCC_{i+1}_std"] = mfcc_std[i]

    # --------------------------------
    # C. Prepare model input
    # --------------------------------

    feature_df = pd.DataFrame([features])
    feature_df = feature_df[model.feature_names_in_]

    return feature_df


# 3. Predict heart sound
def predict_audio(audio_path):

    features = extract_features(audio_path)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    result = "Normal" if prediction == 1 else "Abnormal"

    confidence = float(class_probabilities[prediction])

    return {
        "prediction": result,
        "confidence": confidence
    }
