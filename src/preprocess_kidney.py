"""
Loading + cleaning for the Chronic Kidney Disease (CKD) dataset.

The commonly-distributed kidney_disease.csv (UCI CKD data, Kaggle mirror) is
known to be messy in a specific, consistent way:
  - id column is a row index, not a feature -> dropped
  - several numeric columns (pcv, wc, rc) are read as text because a few
    cells contain stray whitespace/tabs (e.g. "\t43", " 6.2")
  - categorical columns (rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane)
    sometimes have leading/trailing whitespace or a stray tab
    (e.g. "\tno", " yes", "\tno\t")
  - the target column 'classification' has values like "ckd", "notckd",
    but sometimes with a trailing tab ("ckd\t")

IMPORTANT: this cleaning was written from well-documented, known issues with
this specific public dataset, not from inspecting your actual file (it wasn't
available in this environment). Run the sanity checks at the bottom of this
file first and adjust if your CSV differs.
"""
import pandas as pd

TARGET_COL = "classification"
TARGET_POSITIVE_LABEL = "ckd"  # maps to 1 (disease present)

# Columns that SHOULD be numeric but are sometimes stored as text due to stray characters
NUMERIC_BUT_MESSY = ["pcv", "wc", "rc"]

# Columns that are numeric from the start
NUMERIC_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo",
    "pcv", "wc", "rc",
]

CATEGORICAL_COLS = [
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
]


def _strip_strings(df):
    """Strip whitespace/tabs from every object (string) column and value."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": pd.NA, "?": pd.NA, "": pd.NA})
    return df


def load_and_clean_kidney(path):
    """
    Returns (X, y):
      X - DataFrame with NUMERIC_COLS + CATEGORICAL_COLS, cleaned
      y - Series of 0/1 (1 = ckd, 0 = notckd)
    """
    df = pd.read_csv(path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = _strip_strings(df)

    # A couple of known label inconsistencies in this dataset ("\tno" vs "no",
    # "\tyes" vs "yes") are already fixed by _strip_strings. Some releases
    # also have a stray typo "ckd\t" -> already stripped above.

    # Coerce the numeric-but-messy columns explicitly
    for col in NUMERIC_BUT_MESSY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce all intended-numeric columns defensively (no-op if already numeric)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found in {path}")

    df = df.dropna(subset=[TARGET_COL])
    y = (df[TARGET_COL].str.lower() == TARGET_POSITIVE_LABEL).astype(int)

    feature_cols = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS) if c in df.columns]
    X = df[feature_cols].copy()

    return X, y


if __name__ == "__main__":
    # Quick manual sanity check: run `python src/preprocess_kidney.py` after
    # placing kidney_disease.csv in data/, and eyeball the printed summary
    # before trusting the automated cleaning above.
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/kidney_disease.csv"
    X, y = load_and_clean_kidney(data_path)
    print("Shape:", X.shape)
    print("\nClass balance (1=ckd, 0=notckd):")
    print(y.value_counts())
    print("\nMissing values per column:")
    print(X.isna().sum())
    print("\ndtypes:")
    print(X.dtypes)
