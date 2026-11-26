
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, spectrogram
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
    exit()

# 2. Read audio
sample_rate, audio = wavfile.read(file_path)
audio = audio.astype(np.float64)

duration = len(audio) / sample_rate

print("\n--- Audio Information ---")
print("Sample Rate:", sample_rate, "Hz")
print("Duration:", duration, "seconds")

# 3. Apply Band-pass Filter
sos = butter(
    4,
    [25, 400],
    btype="bandpass",
    fs=sample_rate,
    output="sos"
)

filtered_audio = sosfiltfilt(sos, audio)

# 4. Calculate Spectrogram
frequencies, times, power = spectrogram(
    filtered_audio,
    fs=sample_rate,
    nperseg=512,
    noverlap=384
)

# Convert power to decibels
power_db = 10 * np.log10(power + 1e-12)

# 5. Plot results
plt.figure(figsize=(14, 9))

# Original signal
plt.subplot(3, 1, 1)
time_axis = np.arange(len(audio)) / sample_rate
samples = 5 * sample_rate

plt.plot(time_axis[:samples], audio[:samples])
plt.title("Original Heart Sound")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

# Filtered signal
plt.subplot(3, 1, 2)
plt.plot(time_axis[:samples], filtered_audio[:samples])
plt.title("Filtered Heart Sound")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

# Spectrogram
plt.subplot(3, 1, 3)
plt.pcolormesh(
    times,
    frequencies,
    power_db,
    shading="auto",
    cmap="viridis"
)

plt.ylim(0, 500)
plt.title("Heart Sound Spectrogram")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")

plt.tight_layout()
plt.show()
