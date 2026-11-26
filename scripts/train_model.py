
import pandas as pd
import joblib

from pathlib import Path
from tkinter import Tk, filedialog

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# 1. Select feature dataset
root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select Feature Dataset",
    filetypes=[("CSV files", "*.csv")]
)

if not file_path:
    print("No file selected!")
    exit()

# 2. Load dataset
data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# 3. Select features
features = [
    "RMS",
    "ZCR",
    "Spectral_Centroid",
    "Spectral_Bandwidth",
    "Dominant_Frequency"
]

X = data[features]
y = data["label"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Create model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

# 6. Train
model.fit(X_train, y_train)

print("\nTraining completed!")

# 7. Predict
predictions = model.predict(X_test)

# 8. Evaluate
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=[-1, 1],
    target_names=["Abnormal", "Normal"],
    zero_division=0
))

# 9. Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[-1, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Abnormal", "Normal"]
)

display.plot(cmap="Blues")
plt.title("Heart Sound Classification")
plt.tight_layout()
plt.show()

# 10. Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# 11. Save model
model_path = Path(file_path).parent / "heart_sound_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved at:", model_path)
