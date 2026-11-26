
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
from tkinter import Tk, filedialog


# 1. Select audio file
root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select a Heart Sound",
    filetypes=[("WAV files", "*.wav")]
)

if not file_path:
    print("No file selected!")
    root.destroy()
    exit()

file_path = Path(file_path)


# 2. Read audio
sample_rate, audio = wavfile.read(file_path)

audio = audio.astype(np.float64)

# Convert stereo to mono
if audio.ndim == 2:
    audio = audio.mean(axis=1)

print("\nAudio loaded successfully!")
print("Record ID:", file_path.stem)


# 3. Apply Band-pass Filter
sos = butter(
    4,
    [25, 400],
    btype="bandpass",
    fs=sample_rate,
    output="sos"
)

filtered_audio = sosfiltfilt(sos, audio)


# 4. RMS
rms = np.sqrt(np.mean(filtered_audio ** 2))


# 5. Zero Crossing Rate
zero_crossings = np.sum(
    np.diff(np.signbit(filtered_audio))
)

zcr = zero_crossings / len(filtered_audio)


# 6. Frequency Analysis
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
    raise ValueError("Audio contains no usable signal.")


# 7. Spectral Centroid
spectral_centroid = np.sum(
    frequencies * power
) / total_power


# 8. Spectral Bandwidth
spectral_bandwidth = np.sqrt(
    np.sum(
        ((frequencies - spectral_centroid) ** 2) * power
    ) / total_power
)


# 9. Dominant Frequency
dominant_frequency = frequencies[np.argmax(power)]


# 10. Display Features
features = {
    "record_id": file_path.stem,
    "RMS": rms,
    "Zero Crossing Rate": zcr,
    "Spectral Centroid": spectral_centroid,
    "Spectral Bandwidth": spectral_bandwidth,
    "Dominant Frequency": dominant_frequency
}

print("\n--- Extracted Audio Features ---")

for name, value in features.items():
    print(f"{name}: {value:.4f}" if name != "record_id" else f"{name}: {value}")


# 11. Save features
results = pd.DataFrame([features])

output_path = file_path.parent / "extracted_features.csv"

results.to_csv(
    output_path,
    index=False
)

print("\nFeatures saved to:", output_path)

root.destroy()
