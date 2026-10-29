"""
Shared utilities for the disease-classification pipelines (CKD, Liver).

Kept deliberately small and dependency-light: one preprocessing helper,
one baseline-model comparison helper, one save helper. Both train_kidney.py
and train_liver.py import from here so the two scripts stay consistent.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def build_preprocessor(numeric_cols, categorical_cols):
    """
    ColumnTransformer:
      - numeric:     median-impute -> standard-scale
      - categorical: most-frequent-impute -> one-hot (unknowns ignored at inference time)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])


def baseline_candidates(random_state=42):
    """
    Ensemble baseline candidates (Week 1 scope: compare defaults, no tuning yet).
    class_weight='balanced' compensates for the class imbalance in both datasets
    without needing SMOTE at this stage.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state
        ),
        # GradientBoosting has no class_weight param; kept for comparison anyway.
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }


def compare_models(preprocessor, models, X_train, y_train, cv_folds=5, random_state=42):
    """
    5-fold stratified CV, scored on F1 (binary, positive class = 1 = disease present).
    Returns dict {model_name: {"mean_f1": ..., "std_f1": ..., "fold_scores": [...]}}
    and prints a small comparison table.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}
    print(f"\n{'Model':<18} {'Mean F1':>10} {'Std F1':>10}")
    print("-" * 40)
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1)
        results[name] = {
            "mean_f1": float(np.mean(scores)),
            "std_f1": float(np.std(scores)),
            "fold_scores": scores.tolist(),
        }
        print(f"{name:<18} {np.mean(scores):>10.4f} {np.std(scores):>10.4f}")
    return results


def best_model_name(results):
    return max(results, key=lambda k: results[k]["mean_f1"])


def get_param_distributions(model_name):
    """
    RandomizedSearchCV search spaces, prefixed with 'classifier__' so they
    plug straight into a Pipeline(steps=[('preprocessor', ...), ('classifier', ...)]).
    """
    grids = {
        "RandomForest": {
            "classifier__n_estimators": [100, 200, 300, 400, 500],
            "classifier__max_depth": [None, 5, 10, 15, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "ExtraTrees": {
            "classifier__n_estimators": [100, 200, 300, 400, 500],
            "classifier__max_depth": [None, 5, 10, 15, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "GradientBoosting": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth": [2, 3, 4, 5],
            "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
    }
    if model_name not in grids:
        raise ValueError(f"No hyperparameter grid defined for '{model_name}'")
    return grids[model_name]


def balanced_sample_weight(y):
    """
    GradientBoostingClassifier has no class_weight param, unlike RandomForest /
    ExtraTrees. This gives it the same class-imbalance correction via
    per-sample weights, passed as fit_params={'classifier__sample_weight': ...}.
    """
    return compute_sample_weight(class_weight="balanced", y=y)


def extract_feature_importance(pipeline, top_n=15):
    """
    Pulls .feature_importances_ off the fitted classifier step and maps them
    back to human-readable, post-one-hot-encoding feature names using the
    preprocessor's get_feature_names_out(). Returns a list of
    {"feature": ..., "importance": ...} dicts, sorted descending, capped at top_n.
    Raises AttributeError if the classifier isn't a tree-based ensemble.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        raise AttributeError(
            f"{type(classifier).__name__} has no feature_importances_ "
            "(expected a tree-based ensemble model)."
        )

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    pairs = sorted(zip(feature_names, importances), key=lambda p: p[1], reverse=True)
    return [{"feature": str(f), "importance": float(i)} for f, i in pairs[:top_n]]


def save_pipeline(pipeline, metadata, out_dir, filename_stem):
    """
    Saves the fitted pipeline as <stem>.joblib and a sidecar
    <stem>_metadata.json (feature lists, chosen model, test metrics, etc.)
    so the Streamlit app can load both without re-deriving anything.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{filename_stem}.joblib"
    meta_path = out_dir / f"{filename_stem}_metadata.json"

    joblib.dump(pipeline, model_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved pipeline  -> {model_path}")
    print(f"Saved metadata  -> {meta_path}")
    return model_path, meta_path
