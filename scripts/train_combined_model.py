
import pandas as pd
import numpy as np

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
import joblib


# =====================================
# 1. Select CSV files
# =====================================

root = Tk()
root.withdraw()

print("Select Original Features CSV")

original_path = filedialog.askopenfilename(
    title="Select Original Features CSV",
    filetypes=[("CSV files", "*.csv")]
)

if not original_path:
    raise SystemExit("Original CSV was not selected.")

print("Select MFCC Features CSV")

mfcc_path = filedialog.askopenfilename(
    title="Select MFCC Features CSV",
    filetypes=[("CSV files", "*.csv")]
)

if not mfcc_path:
    raise SystemExit("MFCC CSV was not selected.")

original_data = pd.read_csv(original_path)
mfcc_data = pd.read_csv(mfcc_path)

root.destroy()


# =====================================
# 2. Normalize column names
# =====================================

original_data.columns = original_data.columns.str.strip()
mfcc_data.columns = mfcc_data.columns.str.strip()

def find_id_column(data):

    possible_names = [
        "record_id",
        "record",
        "filename",
        "file_name",
        "audio_file",
        "audio",
        "id"
    ]

    for name in possible_names:
        if name in data.columns:
            return name

    raise ValueError(
        "Could not find record ID column.\n"
        f"Available columns: {data.columns.tolist()}"
    )


original_id = find_id_column(original_data)
mfcc_id = find_id_column(mfcc_data)

original_data = original_data.rename(
    columns={original_id: "record_id"}
)

mfcc_data = mfcc_data.rename(
    columns={mfcc_id: "record_id"}
)


# =====================================
# 3. Check labels
# =====================================

if "label" not in original_data.columns:
    raise ValueError(
        "Original CSV does not contain a label column."
    )

if "label" not in mfcc_data.columns:
    raise ValueError(
        "MFCC CSV does not contain a label column."
    )

original_data["record_id"] = (
    original_data["record_id"].astype(str).str.strip()
)

mfcc_data["record_id"] = (
    mfcc_data["record_id"].astype(str).str.strip()
)

# Keep labels from original dataset
mfcc_features = mfcc_data.drop(
    columns=["label", "class"],
    errors="ignore"
)


# =====================================
# 4. Merge datasets
# =====================================

print("\nMerging datasets...")

combined_data = pd.merge(
    original_data,
    mfcc_features,
    on="record_id",
    how="inner",
    validate="one_to_one"
)

if combined_data.empty:
    raise ValueError(
        "No matching record IDs found between the two CSV files."
    )

print("Merged records:", len(combined_data))


# =====================================
# 5. Prepare features and labels
# =====================================

excluded_columns = [
    "record_id",
    "label",
    "class"
]

feature_columns = [
    col for col in combined_data.columns
    if col not in excluded_columns
]

if len(feature_columns) != 31:
    raise ValueError(
        f"Expected 31 features, but found {len(feature_columns)}.\n"
        f"Features: {feature_columns}"
    )

X = combined_data[feature_columns]
y = combined_data["label"]

if X.isnull().values.any():
    raise ValueError("Feature dataset contains missing values.")

if not set(y.unique()).issubset({1, -1}):
    raise ValueError(
        f"Unexpected label values: {y.unique().tolist()}"
    )

print("\nTotal features:", len(feature_columns))
print("Class distribution:")
print(y.value_counts())


# =====================================
# 6. Split dataset
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================
# 7. Train Random Forest
# =====================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# =====================================
# 8. Evaluate model
# =====================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== RESULTS ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[-1, 1],
        target_names=["Abnormal", "Normal"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[-1, 1]
)

print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Abnormal", "Normal"]
)

display.plot(cmap="Blues")
plt.title("Heart Sound Classification")
plt.show()


# =====================================
# 9. Feature importance
# =====================================

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"].head(10)[::-1],
    importance["Importance"].head(10)[::-1]
)

plt.title("Top 10 Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# =====================================
# 10. Save outputs
# =====================================

output_folder = Path(original_path).parent

model_path = output_folder / "heart_sound_combined_model.pkl"

dataset_path = output_folder / "combined_feature_dataset.csv"

joblib.dump(model, model_path)

combined_data.to_csv(
    dataset_path,
    index=False
)

print("\n========== SAVED ==========")
print("Model:", model_path)
print("Dataset:", dataset_path)
print("\nTraining completed successfully!")
