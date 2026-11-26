
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

# 1. Select MFCC dataset
root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select MFCC Dataset",
    filetypes=[("CSV files", "*.csv")]
)

if not file_path:
    print("No file selected!")
    exit()

# 2. Load dataset
data = pd.read_csv(file_path)

print("Dataset loaded!")
print("Dataset shape:", data.shape)

# 3. Select MFCC features
features = [
    column for column in data.columns
    if column.startswith("MFCC_")
]

X = data[features]
y = data["label"]

print("Number of MFCC features:", len(features))

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Create Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

# 6. Train model
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
plt.title("MFCC Heart Sound Classification")
plt.tight_layout()
plt.show()

# 10. Save trained model
model_path = Path(file_path).parent / "heart_sound_mfcc_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved at:", model_path)
