"""
Week 2, Day 8 — feature importance extraction.

Run locally (NOT executed here), once per disease, after training/tuning:
    python src/feature_importance.py --disease kidney --models models
    python src/feature_importance.py --disease liver  --models models

Loads the saved pipeline, pulls .feature_importances_ off the tree-based
classifier, maps importances back to human-readable feature names (post
one-hot encoding), and saves the top 15 to
models/<disease>_feature_importance.json — which app.py reads to show the
"top contributing features" panel.
"""
import argparse
import json
from pathlib import Path

import joblib

import sys
sys.path.append(str(Path(__file__).parent))

from common import extract_feature_importance


def main(args):
    models_dir = Path(args.models)
    pipeline_path = models_dir / f"{args.disease}_pipeline.joblib"
    out_path = models_dir / f"{args.disease}_feature_importance.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"{pipeline_path} not found. Train that disease's model first.")

    pipeline = joblib.load(pipeline_path)
    importances = extract_feature_importance(pipeline, top_n=args.top_n)

    print(f"\nTop {len(importances)} features for '{args.disease}':")
    for row in importances:
        bar = "#" * int(row["importance"] * 100)
        print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")

    out_path.write_text(json.dumps(importances, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract feature importance from a trained pipeline")
    parser.add_argument("--disease", required=True, choices=["kidney", "liver"])
    parser.add_argument("--models", default="models")
    parser.add_argument("--top-n", dest="top_n", type=int, default=15)
    main(parser.parse_args())
