"""
Baseline training for the Chronic Kidney Disease (CKD) classifier.

Run locally (NOT executed here):
    python src/train_kidney.py --data data/kidney_disease.csv --out models

What it does:
  1. Loads + cleans the CKD data (preprocess_kidney.py)
  2. Stratified train/test split
  3. Builds a ColumnTransformer preprocessing pipeline
  4. Compares 3 ensemble baselines (RandomForest, ExtraTrees, GradientBoosting)
     via 5-fold stratified cross-validation, scored on F1
  5. Refits the best one on the full training set
  6. Evaluates once on the held-out test set
  7. Saves the fitted pipeline (preprocessing + model) as one .joblib file,
     plus a metadata JSON, ready for the Streamlit app to load

This is Week 1 scope: establish a baseline. Hyperparameter tuning
(GridSearchCV/RandomizedSearchCV) is Week 2.
"""
import argparse
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import sys
sys.path.append(str(Path(__file__).parent))

from common import build_preprocessor, baseline_candidates, compare_models, best_model_name, save_pipeline
from preprocess_kidney import load_and_clean_kidney, NUMERIC_COLS, CATEGORICAL_COLS


def main(args):
    print(f"Loading data from {args.data} ...")
    X, y = load_and_clean_kidney(args.data)
    print(f"Loaded {len(X)} rows after cleaning. Class balance:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    numeric_cols = [c for c in NUMERIC_COLS if c in X.columns]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    print("\nComparing baseline ensemble models (5-fold CV, F1 scoring)...")
    models = baseline_candidates(random_state=args.random_state)
    cv_results = compare_models(preprocessor, models, X_train, y_train, random_state=args.random_state)

    winner = best_model_name(cv_results)
    print(f"\nBest baseline by mean CV F1: {winner}")

    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", models[winner]),
    ])
    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    print(f"\nHeld-out test F1: {test_f1:.4f}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=["notckd", "ckd"]))
    print("Confusion matrix (rows=actual, cols=predicted, order=[notckd, ckd]):")
    print(confusion_matrix(y_test, y_pred))

    metadata = {
        "disease": "chronic_kidney_disease",
        "chosen_model": winner,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target_positive_label": "ckd",
        "cv_results": cv_results,
        "test_f1": test_f1,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    save_pipeline(final_pipeline, metadata, args.out, "kidney_pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline CKD classifier")
    parser.add_argument("--data", default="data/kidney_disease.csv", help="Path to kidney_disease.csv")
    parser.add_argument("--out", default="models", help="Output directory for the saved pipeline")
    parser.add_argument("--test-size", dest="test_size", type=float, default=0.2)
    parser.add_argument("--random-state", dest="random_state", type=int, default=42)
    main(parser.parse_args())
