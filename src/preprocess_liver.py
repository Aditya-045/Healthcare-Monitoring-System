"""
Loading + cleaning for the Indian Liver Patient Dataset (ILPD).

Known characteristics of this dataset:
  - target column is 'Dataset': 1 = liver patient (disease present),
    2 = not a liver patient -> remapped here to 1 / 0
  - 'Gender' is a string column (Male/Female) -> treated as categorical
  - 'Albumin_and_Globulin_Ratio' has a small number of missing values
    (a handful of rows) -> handled by the median imputer in the shared
    preprocessing pipeline, not here
  - otherwise numeric and comparatively clean

As with preprocess_kidney.py, this was written from the well-documented
public version of this dataset, not from inspecting your actual file
(it wasn't available in this environment). Run the sanity check at the
bottom first.
"""
import pandas as pd

TARGET_COL = "Dataset"

NUMERIC_COLS = [
    "Age",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]

CATEGORICAL_COLS = ["Gender"]


def load_and_clean_liver(path):
    """
    Returns (X, y):
      X - DataFrame with NUMERIC_COLS + CATEGORICAL_COLS, cleaned
      y - Series of 0/1 (1 = liver disease present, 0 = not a patient)
    """
    df = pd.read_csv(path)

    # Normalize Gender text just in case of stray whitespace/casing
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.title()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found in {path}")

    df = df.dropna(subset=[TARGET_COL])
    # Source encoding: 1 = liver patient, 2 = not a patient -> map to 1 / 0
    y = df[TARGET_COL].map({1: 1, 2: 0})
    if y.isna().any():
        raise ValueError(
            "Unexpected values in 'Dataset' column beyond {1, 2}; "
            "inspect the raw CSV before proceeding."
        )
    y = y.astype(int)

    feature_cols = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS) if c in df.columns]
    X = df[feature_cols].copy()

    return X, y


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/indian_liver_patient.csv"
    X, y = load_and_clean_liver(data_path)
    print("Shape:", X.shape)
    print("\nClass balance (1=liver disease, 0=no liver disease):")
    print(y.value_counts())
    print("\nMissing values per column:")
    print(X.isna().sum())
    print("\ndtypes:")
    print(X.dtypes)
