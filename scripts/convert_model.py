import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load("scripts/heart_sound_combined_model.pkl")

onnx_model = convert_sklearn(
    model,
    initial_types=[
        ("float_input", FloatTensorType([None, 31]))
    ],
    options={id(model): {"zipmap": False}},
    target_opset=17
)

with open("scripts/heart_sound_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved successfully!")
