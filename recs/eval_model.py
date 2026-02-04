#!/usr/bin/env python3
"""
Evaluate a saved clustering model (scaler + centroids) on a test set.

Computes: silhouette_score, inertia, cluster_sizes.
Outputs: recs/model/eval_metrics.json

Usage:
  python -m recs.eval_model
  python -m recs.eval_model --model-dir recs/model --test-csv recs/model/test_set.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]
MODEL_DIR = Path(__file__).resolve().parent / "model"


def eval_model(
    model_dir: Path,
    test_csv: Path | None = None,
) -> dict:
    """Compute eval metrics for a saved model on test data."""
    model_dir = Path(model_dir)
    test_path = test_csv or model_dir / "test_set.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test set not found: {test_path}")

    try:
        import joblib
    except ImportError:
        import pickle
        joblib = None

    scaler_path = model_dir / "scaler.joblib"
    if not scaler_path.exists():
        scaler_path = model_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found in {model_dir}")

    centroids_path = model_dir / "centroids.npy"
    if not centroids_path.exists():
        raise FileNotFoundError(f"Centroids not found: {centroids_path}")

    scaler = joblib.load(scaler_path) if joblib else pickle.load(open(scaler_path, "rb"))
    centroids = np.load(centroids_path)

    df = pd.read_csv(test_path)
    if "id" in df.columns and "track_id" not in df.columns:
        df = df.rename(columns={"id": "track_id"})
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Test set missing columns: {missing}")

    df = df.dropna(subset=FEATURE_COLS)
    df = df[df["tempo"] > 0]
    if len(df) < 2:
        return {"error": "Too few valid rows for evaluation", "n_rows": len(df)}

    X = df[FEATURE_COLS].astype(float).values
    X_scaled = scaler.transform(X)
    labels, dists = pairwise_distances_argmin_min(X_scaled, centroids, metric="euclidean")
    labels = labels.astype(int)

    n_clusters = len(np.unique(labels))
    silhouette = float(silhouette_score(X_scaled, labels)) if n_clusters > 1 else 0.0
    inertia = float(np.sum(dists**2))
    cluster_sizes = dict(zip(*np.unique(labels, return_counts=True)))
    cluster_sizes = {int(k): int(v) for k, v in cluster_sizes.items()}

    metrics = {
        "n_test_tracks": int(len(df)),
        "n_clusters": int(n_clusters),
        "silhouette_score": round(silhouette, 4),
        "inertia": round(inertia, 2),
        "cluster_sizes": cluster_sizes,
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved clustering model")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Model directory")
    parser.add_argument("--test-csv", default=None, help="Test CSV path")
    args = parser.parse_args()

    try:
        metrics = eval_model(Path(args.model_dir), Path(args.test_csv) if args.test_csv else None)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    if "error" in metrics:
        print(metrics["error"])
        return 1

    out_path = Path(args.model_dir) / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Eval metrics saved to {out_path}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
