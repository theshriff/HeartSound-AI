import joblib
import numpy as np
import onnxruntime as ort

model = joblib.load("scripts/heart_sound_combined_model.pkl")
session = ort.InferenceSession("scripts/heart_sound_model.onnx")

X = np.random.rand(1, 31).astype(np.float32)

original_prediction = model.predict(X)
original_probability = model.predict_proba(X)

input_name = session.get_inputs()[0].name
onnx_outputs = session.run(None, {input_name: X})

print("Original prediction:", original_prediction)
print("Original probabilities:", original_probability)
print("ONNX outputs:")
for output in onnx_outputs:
    print(output)
