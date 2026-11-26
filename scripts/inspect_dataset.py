
import pandas as pd
from pathlib import Path
from tkinter import Tk, filedialog

# انتخاب فایل برچسب ها
root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select REFERENCE.csv",
    filetypes=[("CSV files", "*.csv")]
)

if not file_path:
    print("No file selected!")
    exit()

# خواندن فایل CSV
data = pd.read_csv(
    file_path,
    header=None,
    names=["record_id", "label"]
)

# نمایش اطلاعات دیتاست
print("\n--- Dataset Information ---")
print(data.head())

# شمارش نمونه های هر کلاس
print("\n--- Label Counts ---")
print(data["label"].value_counts())

# تبدیل برچسب ها به نام قابل فهم
data["class"] = data["label"].map({
    1: "Normal",
    -1: "Abnormal"
})

print("\n--- Class Distribution ---")
print(data["class"].value_counts())

# بررسی وجود فایل های صوتی
dataset_folder = Path(file_path).parent

existing_files = 0
missing_files = 0

for record_id in data["record_id"]:
    audio_file = dataset_folder / f"{record_id}.wav"

    if audio_file.exists():
        existing_files += 1
    else:
        missing_files += 1

print("\n--- Audio File Check ---")
print("Existing WAV files:", existing_files)
print("Missing WAV files:", missing_files)
