
import numpy as np
import pandas as pd

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
    root.destroy()
    raise SystemExit("No file selected!")

csv_path = Path(csv_path)
dataset_folder = csv_path.parent


# 2. Read labels
labels = pd.read_csv(
    csv_path,
    header=None,
    names=["record_id", "label"]
)

results = []

print("\nStarting feature extraction...\n")


# 3. Process every audio file
for _, row in labels.iterrows():

    record_id = str(row["record_id"]).strip()
    label = int(row["label"])

    audio_path = dataset_folder / f"{record_id}.wav"

    if not audio_path.exists():
        print("Missing:", record_id)
        continue

    try:
        sample_rate, audio = wavfile.read(audio_path)

        audio = audio.astype(np.float64)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if len(audio) == 0:
            raise ValueError("Empty audio")

        sos = butter(
            4,
            [25, 400],
            btype="bandpass",
            fs=sample_rate,
            output="sos"
        )

        filtered_audio = sosfiltfilt(sos, audio)

        # RMS
        rms = np.sqrt(np.mean(filtered_audio ** 2))

        # Zero Crossing Rate
        zcr = np.mean(
            np.diff(np.signbit(filtered_audio))
        )

        # Frequency analysis
        frequencies = np.fft.rfftfreq(
            len(filtered_audio),
            d=1 / sample_rate
        )

        fft_values = np.abs(
            np.fft.rfft(filtered_audio)
        )

        power = fft_values ** 2
        power[0] = 0

        total_power = np.sum(power)

        if total_power == 0:
            raise ValueError("Signal has zero power")

        # Spectral Centroid
        centroid = np.sum(
            frequencies * power
        ) / total_power

        # Spectral Bandwidth
        bandwidth = np.sqrt(
            np.sum(
                ((frequencies - centroid) ** 2) * power
            ) / total_power
        )

        # Dominant Frequency
        dominant_frequency = frequencies[
            np.argmax(power)
        ]

        results.append({
            "record_id": record_id,
            "RMS": rms,
            "ZCR": zcr,
            "Spectral_Centroid": centroid,
            "Spectral_Bandwidth": bandwidth,
            "Dominant_Frequency": dominant_frequency,
            "label": label
        })

        print("Processed:", record_id)

    except Exception as error:
        print("Error:", record_id, error)


# 4. Create dataset
feature_data = pd.DataFrame(results)

if feature_data.empty:
    root.destroy()
    raise SystemExit("No audio files were successfully processed.")


# 5. Add class names
feature_data["class"] = feature_data["label"].map({
    1: "Normal",
    -1: "Abnormal"
})


# 6. Save CSV
output_path = dataset_folder / "feature_dataset_training_a.csv"

feature_data.to_csv(
    output_path,
    index=False
)


# 7. Summary
print("\n--- Extraction Summary ---")

print("Total processed:", len(feature_data))

print("\nClass distribution:")
print(feature_data["class"].value_counts())

print("\nColumns:")
print(feature_data.columns.tolist())

print("\nSaved to:", output_path)

root.destroy()
