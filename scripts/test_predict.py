
from tkinter import Tk, filedialog
from predict import predict_audio

root = Tk()
root.withdraw()

audio_path = filedialog.askopenfilename(
    title="Select Heart Sound Audio",
    filetypes=[("WAV files", "*.wav")]
)

if audio_path:
    result = predict_audio(audio_path)

    print("\nPrediction:", result["prediction"])
    print("Confidence:", f'{result["confidence"] * 100:.2f}%')
else:
    print("No file selected.")
