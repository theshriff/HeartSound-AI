
import numpy as np
import pandas as pd
import librosa

from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
from tkinter import Tk, filedialog

# 1. Select REFERENCE.csv
root = Tk()
root.withdraw()

csv_path = filedialog.askopenfilename(
    title="Select REFERENCE.csv",
    filetypes=[("CSV files", "*.csv")]
)

if not csv_path:
    print("No file selected!")
    exit()

csv_path = Path(csv_path)
dataset_folder = csv_path.parent

# 2. Read labels
labels = pd.read_csv(
    csv_path,
    header=None,
    names=["record_id", "label"]
)

results = []

print("\nStarting MFCC extraction...\n")

# 3. Process audio files
for _, row in labels.iterrows():

    record_id = row["record_id"]
    label = row["label"]

    audio_path = dataset_folder / f"{record_id}.wav"

    try:
        sample_rate, audio = wavfile.read(audio_path)

        audio = audio.astype(np.float32)

        # Convert stereo to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Normalize audio amplitude
        max_value = np.max(np.abs(audio))

        if max_value > 0:
            audio = audio / max_value

        # Band-pass filter
        sos = butter(
            4,
            [25, 400],
            btype="bandpass",
            fs=sample_rate,
            output="sos"
        )

        filtered_audio = sosfiltfilt(sos, audio)

        # Extract 13 MFCC coefficients
        mfcc = librosa.feature.mfcc(
            y=filtered_audio.astype(np.float32),
            sr=sample_rate,
            n_mfcc=13,
            n_fft=512,
            hop_length=128
        )

        # Calculate mean and standard deviation
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        features = {
            "record_id": record_id,
            "label": label
        }

        for i in range(13):
            features[f"MFCC_{i+1}_mean"] = mfcc_mean[i]
            features[f"MFCC_{i+1}_std"] = mfcc_std[i]

        results.append(features)

        print("Processed:", record_id)

    except Exception as error:
        print("Error:", record_id, error)

# 4. Create dataset
feature_data = pd.DataFrame(results)

feature_data["class"] = feature_data["label"].map({
    1: "Normal",
    -1: "Abnormal"
})

# 5. Save results
output_path = dataset_folder / "mfcc_dataset_training_a.csv"

feature_data.to_csv(
    output_path,
    index=False
)

print("\n--- MFCC Extraction Summary ---")
print("Total processed:", len(feature_data))
print("Total features:", 26)
print("\nClass distribution:")
print(feature_data["class"].value_counts())

print("\nSaved to:", output_path)
