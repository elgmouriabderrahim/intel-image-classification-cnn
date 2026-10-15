import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "raw" / "seg_train"

data = []

for folder in TRAIN_DIR.iterdir():
    if folder.is_dir():
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() == ".jpg":
                data.append({"path": str(file), "label": folder.name})

df = pd.DataFrame(data)

train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))