import librosa
import json

mel_filters = librosa.filters.mel(
    sr=2000,
    n_fft=512,
    n_mels=128,
    fmin=0.0,
    fmax=1000.0,
    htk=False,
    norm="slaney",
    dtype="float32"
)

with open("frontend/mel_filters.json", "w") as f:
    json.dump(mel_filters.tolist(), f)

print("Mel filters exported:", mel_filters.shape)