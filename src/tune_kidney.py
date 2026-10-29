"""
Week 2 — hyperparameter tuning for the CKD classifier.

Run locally (NOT executed here):
    python src/tune_kidney.py --data data/kidney_disease.csv --models models

What it does:
  1. Reads models/kidney_pipeline_metadata.json from Week 1 to find out
     which baseline model won (RandomForest / ExtraTrees / GradientBoosting)
     and which train/test split was used.
  2. Re-creates the exact same stratified split (same random_state, same
     test_size) so tuned results are directly comparable to the Week 1
     baseline F1.
  3. Runs RandomizedSearchCV (5-fold, scored on F1) over that model's
     hyperparameter space.
  4. For GradientBoosting (no class_weight param), applies balanced
     sample weights to correct for class imbalance during the search.
  5. Evaluates the tuned pipeline on the same held-out test set.
  6. Backs up the Week 1 baseline joblib, then saves the tuned pipeline
     under the original filename (so app.py / later steps don't need to
     change what they load) and updates the metadata JSON with a "tuning"
     section (best_params, tuned_test_f1, baseline_test_f1).
"""
import argparse
import json
import shutil
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

import sys
sys.path.append(str(Path(__file__).parent))

from common import build_preprocessor, baseline_candidates, get_param_distributions, balanced_sample_weight
from preprocess_kidney import load_and_clean_kidney


def main(args):
    models_dir = Path(args.models)
    metadata_path = models_dir / "kidney_pipeline_metadata.json"
    pipeline_path = models_dir / "kidney_pipeline.joblib"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Run train_kidney.py (Week 1) before tuning."
        )
    metadata = json.loads(metadata_path.read_text())

    chosen_model = metadata["chosen_model"]
    numeric_cols = metadata["numeric_features"]
    categorical_cols = metadata["categorical_features"]
    test_size = metadata["test_size"]
    random_state = metadata["random_state"]
    baseline_test_f1 = metadata["test_f1"]

    print(f"Baseline model from Week 1: {chosen_model} (test F1 = {baseline_test_f1:.4f})")

    print(f"Loading data from {args.data} ...")
    X, y = load_and_clean_kidney(args.data)

    # Same split as Week 1, so the comparison is apples-to-apples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    fresh_estimator = baseline_candidates(random_state=random_state)[chosen_model]
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", fresh_estimator)])

    param_distributions = get_param_distributions(chosen_model)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring="f1",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    fit_kwargs = {}
    if chosen_model == "GradientBoosting":
        fit_kwargs["classifier__sample_weight"] = balanced_sample_weight(y_train)

    print(f"\nRunning RandomizedSearchCV ({args.n_iter} candidates, 5-fold CV, scoring=F1)...")
    search.fit(X_train, y_train, **fit_kwargs)

    print(f"\nBest CV F1: {search.best_score_:.4f}")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    tuned_pipeline = search.best_estimator_
    y_pred = tuned_pipeline.predict(X_test)
    tuned_test_f1 = f1_score(y_test, y_pred)

    print(f"\nTuned held-out test F1: {tuned_test_f1:.4f}  (baseline was {baseline_test_f1:.4f})")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=["notckd", "ckd"]))
    print("Confusion matrix (rows=actual, cols=predicted, order=[notckd, ckd]):")
    print(confusion_matrix(y_test, y_pred))

    # Back up the Week 1 baseline before overwriting
    if pipeline_path.exists():
        backup_path = models_dir / "kidney_pipeline_baseline.joblib"
        shutil.copy(pipeline_path, backup_path)
        print(f"\nBacked up Week 1 baseline -> {backup_path}")

    import joblib
    joblib.dump(tuned_pipeline, pipeline_path)
    print(f"Saved tuned pipeline -> {pipeline_path}")

    metadata["tuning"] = {
        "method": "RandomizedSearchCV",
        "n_iter": args.n_iter,
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "tuned_test_f1": float(tuned_test_f1),
        "baseline_test_f1": float(baseline_test_f1),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Updated metadata -> {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the CKD classifier (Week 2)")
    parser.add_argument("--data", default="data/kidney_disease.csv")
    parser.add_argument("--models", default="models")
    parser.add_argument("--n-iter", dest="n_iter", type=int, default=40)
    main(parser.parse_args())
