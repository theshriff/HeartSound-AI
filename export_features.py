from tkinter import Tk, filedialog
from scripts.predict import extract_features

root = Tk()
root.withdraw()

audio_path = filedialog.askopenfilename(
    title="Select a WAV file",
    filetypes=[("WAV files", "*.wav")]
)

root.destroy()

if not audio_path:
    print("No file selected.")
    exit()

features = extract_features(audio_path)

features.to_json(
    "sample_features.json",
    orient="records",
    indent=4
)

print("\nExtracted features:")
print(features.to_string(index=False))

print("\nSaved to sample_features.json")