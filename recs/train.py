#!/usr/bin/env python3
"""
Train baseline model for music recommendation.

Loads Kaggle data (CSV), fits StandardScaler + KMeans on tempo, energy, danceability,
valence, loudness. Saves scaler + centroids to recs/model/ for use at inference.

Usage:
  # Train on Kaggle CSV (required):
  python -m recs.train --csv path/to/kaggle_data.csv [--sample 200000] [--clusters 6]

  # Or from project root:
  python3 -m recs.train --csv data/kaggle_spotify_features.csv --sample 200000 --clusters 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Feature columns (must match inference)
FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]

# Default paths (project root = parent of recs/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent / "model"


def load_data(
    csv_path: Path | str,
    sample_size: int | None = 200_000,
) -> pd.DataFrame:
    """
    Load feature rows from Kaggle CSV with missing value handling:
    - If tempo is missing: drop the track (tempo is critical for BPM matching)
    - If other features missing: impute with median (energy, danceability, valence, loudness)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Kaggle CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}. Found: {list(df.columns)}")

    if "track_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "track_id"})
        else:
            df["track_id"] = [f"T{i:03d}" for i in range(len(df))]

    initial_count = len(df)

    # Convert feature columns to numeric (handles strings/errors)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # CRITICAL: Drop tracks with missing or invalid tempo (tempo is required for BPM matching)
    before_tempo_drop = len(df)
    df = df.dropna(subset=["tempo"])
    # Also drop invalid tempo values (tempo must be > 0)
    invalid_tempo = df[df["tempo"] <= 0]
    df = df[df["tempo"] > 0]
    tempo_dropped = before_tempo_drop - len(df)
    if tempo_dropped > 0:
        print(f"  Dropped {tempo_dropped} tracks with missing or invalid tempo (tempo <= 0)")

    # Impute missing values for other features (energy, danceability, valence, loudness)
    # Use median to avoid skewing from outliers
    other_features = [f for f in FEATURE_COLS if f != "tempo"]
    imputed_counts = {}
    for col in other_features:
        missing = df[col].isna().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputed_counts[col] = missing
            print(f"  Imputed {missing} missing values in '{col}' with median ({median_val:.3f})")

    if sample_size is not None and len(df) > sample_size:
        df = df.head(sample_size)

    final_count = len(df)
    print(f"Loaded {final_count} rows from Kaggle CSV: {csv_path}")
    if initial_count != final_count:
        print(f"  Started with {initial_count} rows, kept {final_count} after cleaning")

    return df


def train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by rows (tracks) into train/test. Same track never in both."""
    n = len(df)
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def train(
    df: pd.DataFrame,
    n_clusters: int = 6,
    random_state: int = 42,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Fit scaler and KMeans on df; return scaler, centroids, labels."""
    X = df[FEATURE_COLS].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_
    return scaler, centroids, labels


def save_model(
    scaler: StandardScaler,
    centroids: np.ndarray,
    n_clusters: int,
    out_dir: Path | str,
) -> None:
    """Save scaler (joblib), centroids (npy), config (json) to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
    except ImportError:
        import pickle
        joblib = None

    if joblib is not None:
        joblib.dump(scaler, out_dir / "scaler.joblib")
    else:
        with open(out_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    np.save(out_dir / "centroids.npy", centroids)

    config = {
        "n_clusters": int(n_clusters),
        "feature_names": FEATURE_COLS,
        "version": "1",
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved model to {out_dir}: scaler.joblib, centroids.npy, config.json")


def build_track_lookup(
    df: pd.DataFrame,
    scaler: StandardScaler,
    centroids: np.ndarray,
    out_path: Path | str,
) -> int:
    """
    Assign cluster to every track using fitted scaler + centroids.
    Save (track_id, cluster, f0..f4) to SQLite for inference without audio features.
    Returns number of rows written.
    """
    import sqlite3

    out_path = Path(out_path)
    X = df[FEATURE_COLS].astype(float).values
    X_scaled = scaler.transform(X)
    # Assign to nearest centroid
    from sklearn.metrics.pairwise import pairwise_distances_argmin_min
    labels = pairwise_distances_argmin_min(X_scaled, centroids, metric="euclidean")[0]

    rows = [
        (str(df.iloc[i]["track_id"]), int(labels[i]), *X_scaled[i].tolist())
        for i in range(len(df))
    ]
    conn = sqlite3.connect(str(out_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS track_lookup (
                track_id TEXT PRIMARY KEY,
                cluster INTEGER NOT NULL,
                f0 REAL, f1 REAL, f2 REAL, f3 REAL, f4 REAL
            )
            """
        )
        conn.execute("DELETE FROM track_lookup")
        conn.executemany(
            "INSERT OR REPLACE INTO track_lookup (track_id, cluster, f0, f1, f2, f3, f4) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM track_lookup").fetchone()[0]
    finally:
        conn.close()
    print(f"Saved track lookup to {out_path} ({n} tracks). Inference can use track_id only (no audio features).")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train baseline (scaler + KMeans) for recs on Kaggle data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on Kaggle CSV (required):
  python -m recs.train --csv data/kaggle_spotify_features.csv

  # With options:
  python -m recs.train --csv kaggle_data.csv --sample 500000 --clusters 8

  # Use all rows (no sampling):
  python -m recs.train --csv kaggle_data.csv --sample 0
        """
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to Kaggle CSV (required)")
    parser.add_argument("--sample", type=int, default=200_000, help="Max rows to use (default 200000, 0 = all)")
    parser.add_argument("--clusters", type=int, default=6, help="KMeans n_clusters (default 6)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output dir (default recs/model/)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction (default 0.8), rest = test")
    parser.add_argument("--seed", type=int, default=42, help="Random state for KMeans and train/test split")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else MODEL_DIR

    print("Loading Kaggle data...")
    df = load_data(
        csv_path=args.csv,
        sample_size=args.sample if args.sample > 0 else None,
    )
    if len(df) < 100:
        print("Warning: very few rows; model may be poor. Need at least hundreds of tracks.")
    if len(df) < 10:
        print("Too few rows. Exiting.")
        return 1

    print("Train/test split (by track)...")
    train_df, test_df = train_test_split(df, train_ratio=args.train_ratio, random_state=args.seed)
    print(f"  Train: {len(train_df)} tracks, Test: {len(test_df)} tracks")

    print("Fitting scaler + KMeans on train set...")
    scaler, centroids, labels = train(train_df, n_clusters=args.clusters, random_state=args.seed)

    save_model(scaler, centroids, args.clusters, out_dir)
    # Save test set for later evaluation (optional)
    test_df.to_csv(out_dir / "test_set.csv", index=False)
    print(f"  Saved test set to {out_dir / 'test_set.csv'} for later evaluation.")

    # Build track lookup: track_id -> cluster + scaled embedding (so inference needs no audio features)
    print("Building track lookup (track_id -> cluster + embedding)...")
    build_track_lookup(df, scaler, centroids, out_dir / "track_lookup.db")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
