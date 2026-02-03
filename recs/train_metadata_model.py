#!/usr/bin/env python3
"""
Train metadata → audio features prediction model.

Uses enriched Kaggle dataset (with Spotify metadata) to train a model that predicts
audio features (tempo, energy, danceability, valence, loudness) from metadata.

Usage:
  python -m recs.train_metadata_model --enriched-csv enriched_kaggle.csv --output-dir recs/model/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]
MODEL_DIR = Path(__file__).resolve().parent / "model"


def extract_metadata_features(df: pd.DataFrame, audio_features_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Extract and encode metadata features for ML model.

    Features:
    - Popularity (normalized 0-1)
    - Release year (normalized)
    - Duration (normalized)
    - Explicit flag (binary)
    - Artist encoding (label encoded, normalized)
    - Genres (multi-hot encoding, top 20 genres)
    - Number of genres
    - Related artists features (average audio features from related artists' songs in training set)

    Args:
        df: DataFrame with metadata columns
        audio_features_df: Optional DataFrame with audio features (for computing related artists averages)
    """
    features = pd.DataFrame(index=df.index)

    # Popularity (normalize 0-100)
    features["popularity"] = df["popularity"].fillna(50) / 100.0

    # Release year (extract from release_date)
    def extract_year(date_str):
        if pd.isna(date_str) or not date_str:
            return 2020  # Default
        try:
            return int(str(date_str).split("-")[0])
        except:
            return 2020

    features["release_year"] = df["release_date"].apply(extract_year)
    features["release_year"] = (features["release_year"] - 1950) / 70.0  # Normalize (1950-2020 range)

    # Duration (normalize, typical range 60k-300k ms = 1-5 min)
    if "duration_ms" in df.columns:
        features["duration"] = df["duration_ms"].fillna(180000) / 300000.0  # Normalize to 0-1
    else:
        features["duration"] = 0.6  # Default ~3 min

    # Explicit flag (binary)
    if "explicit" in df.columns:
        features["explicit"] = df["explicit"].astype(int).fillna(0)
    else:
        features["explicit"] = 0

    # Artist encoding (label encode main artist)
    if "main_artist_id" in df.columns:
        le_artist = LabelEncoder()
        artist_ids = df["main_artist_id"].fillna("unknown").astype(str)
        features["artist_encoded"] = le_artist.fit_transform(artist_ids)
        features["artist_encoded"] = features["artist_encoded"] / max(len(le_artist.classes_), 1)  # Normalize
    else:
        features["artist_encoded"] = 0.5

    # Genres (multi-hot encoding for top genres)
    # Handle both formats: single genre string (Kaggle) or comma-separated/list
    all_genres = set()
    for genres_str in df["genres"].fillna(""):
        if isinstance(genres_str, str):
            # Try comma-separated first, else treat as single genre
            if "," in genres_str:
                genres = [g.strip() for g in genres_str.split(",") if g.strip()]
            else:
                genres = [genres_str.strip()] if genres_str.strip() else []
            all_genres.update(genres)
        elif isinstance(genres_str, list):
            all_genres.update(genres_str)

    # Top 20 genres
    genre_counts = {}
    for genres_str in df["genres"].fillna(""):
        if isinstance(genres_str, str):
            if "," in genres_str:
                genres = [g.strip() for g in genres_str.split(",") if g.strip()]
            else:
                genres = [genres_str.strip()] if genres_str.strip() else []
        elif isinstance(genres_str, list):
            genres = genres_str
        else:
            genres = []
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1

    top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:20]
    top_genre_names = [g[0] for g in top_genres]

    for genre in top_genre_names:
        features[f"genre_{genre}"] = df["genres"].apply(
            lambda x: (
                1 if (
                    (isinstance(x, str) and (genre in x.split(",") if "," in x else genre == x.strip())) or
                    (isinstance(x, list) and genre in x)
                ) else 0
            )
        )

    # Number of genres
    features["num_genres"] = df["genres"].apply(
        lambda x: (
            len(x.split(",")) if isinstance(x, str) and "," in x else
            (1 if isinstance(x, str) and x.strip() else 0) if isinstance(x, str) else
            (len(x) if isinstance(x, list) else 0)
        )
    )
    features["num_genres"] = features["num_genres"] / 10.0  # Normalize

    # Related artists features (average audio features from related artists' songs in training set)
    if audio_features_df is not None and "related_artist_ids" in df.columns:
        # Build artist -> average features mapping from training set
        artist_features = {}
        if "main_artist_id" in audio_features_df.columns:
            for artist_id in audio_features_df["main_artist_id"].dropna().unique():
                artist_tracks = audio_features_df[audio_features_df["main_artist_id"] == artist_id]
                if len(artist_tracks) > 0:
                    artist_features[str(artist_id)] = {
                        "tempo": artist_tracks["tempo"].mean(),
                        "energy": artist_tracks["energy"].mean(),
                        "danceability": artist_tracks["danceability"].mean(),
                        "valence": artist_tracks["valence"].mean(),
                        "loudness": artist_tracks["loudness"].mean(),
                    }

        # For each track, compute average features from related artists
        def get_related_artist_features(related_ids_str):
            if pd.isna(related_ids_str):
                return [0.0] * 5
            if isinstance(related_ids_str, str):
                related_ids = [id.strip() for id in related_ids_str.split(",") if id.strip()]
            elif isinstance(related_ids_str, list):
                related_ids = related_ids_str
            else:
                return [0.0] * 5

            related_feats = []
            for feat_name in ["tempo", "energy", "danceability", "valence", "loudness"]:
                values = []
                for rid in related_ids[:5]:  # Top 5 related artists
                    if rid in artist_features:
                        values.append(artist_features[rid][feat_name])
                avg = sum(values) / len(values) if values else 0.0
                # Normalize based on typical ranges
                if feat_name == "tempo":
                    avg = avg / 200.0  # Normalize tempo (0-200 BPM)
                elif feat_name == "loudness":
                    avg = (avg + 50) / 50.0  # Normalize loudness (-50 to 0 dB)
                # energy, danceability, valence already 0-1
                related_feats.append(avg)

            return related_feats if related_feats else [0.0] * 5

        related_feats = df["related_artist_ids"].apply(get_related_artist_features)
        features["related_tempo"] = related_feats.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0.0)
        features["related_energy"] = related_feats.apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else 0.0)
        features["related_danceability"] = related_feats.apply(lambda x: x[2] if isinstance(x, list) and len(x) > 2 else 0.0)
        features["related_valence"] = related_feats.apply(lambda x: x[3] if isinstance(x, list) and len(x) > 3 else 0.0)
        features["related_loudness"] = related_feats.apply(lambda x: x[4] if isinstance(x, list) and len(x) > 4 else 0.0)
    else:
        # Defaults if no related artists data
        features["related_tempo"] = 0.0
        features["related_energy"] = 0.0
        features["related_danceability"] = 0.0
        features["related_valence"] = 0.0
        features["related_loudness"] = 0.0

    return features


def train_metadata_model(
    enriched_df: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    """
    Train metadata → audio features model.

    Returns dict with model info and metrics.
    """
    print("Extracting metadata features...")
    X_meta = extract_metadata_features(enriched_df, audio_features_df=enriched_df)

    # Target: audio features
    y = enriched_df[FEATURE_COLS].values

    # Filter out rows with missing metadata
    valid_mask = X_meta.notna().all(axis=1) & enriched_df[FEATURE_COLS].notna().all(axis=1)
    X_meta = X_meta[valid_mask]
    y = y[valid_mask]

    print(f"Training samples: {len(X_meta)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_meta, y, test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train separate models for each audio feature
    models = {}
    scalers = {}
    metrics = {}

    for idx, feature_name in enumerate(FEATURE_COLS):
        print(f"\nTraining model for {feature_name}...")
        y_train_feat = y_train[:, idx]
        y_test_feat = y_test[:, idx]

        # Train Gradient Boosting Regressor
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train_feat)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        train_mae = mean_absolute_error(y_train_feat, y_pred_train)
        test_mae = mean_absolute_error(y_test_feat, y_pred_test)
        test_r2 = r2_score(y_test_feat, y_pred_test)

        metrics[feature_name] = {
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
        }

        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

        models[feature_name] = model

    # Save models
    output_dir.mkdir(parents=True, exist_ok=True)

    if joblib is not None:
        for feature_name, model in models.items():
            joblib.dump(model, output_dir / f"metadata_model_{feature_name}.joblib")
    else:
        import pickle
        for feature_name, model in models.items():
            with open(output_dir / f"metadata_model_{feature_name}.pkl", "wb") as f:
                pickle.dump(model, f)

    # Save feature names (for inference)
    feature_names = list(X_meta.columns)
    with open(output_dir / "metadata_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save metrics
    with open(output_dir / "metadata_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Models saved to {output_dir}")
    print(f"   Feature names: {len(feature_names)}")

    return {
        "models": models,
        "feature_names": feature_names,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train metadata → audio features prediction model"
    )
    parser.add_argument("--enriched-csv", type=str, required=True, help="Path to enriched Kaggle CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: recs/model/)")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    args = parser.parse_args()

    enriched_path = Path(args.enriched_csv)
    if not enriched_path.exists():
        print(f"ERROR: Enriched CSV not found: {enriched_path}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR

    print("Loading enriched dataset...")
    df = pd.read_csv(enriched_path)

    # Check required columns
    required = FEATURE_COLS + ["track_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return 1

    # Check metadata columns
    if "popularity" not in df.columns:
        print("ERROR: Dataset not enriched. Run enrich_kaggle_metadata.py first.")
        return 1

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} samples (testing mode)")

    print(f"Loaded {len(df)} tracks")

    # Filter valid rows
    df = df.dropna(subset=FEATURE_COLS)
    df = df[df["tempo"] > 0]
    print(f"Valid tracks: {len(df)}")

    if len(df) < 100:
        print("ERROR: Need at least 100 valid tracks to train")
        return 1

    # Train model
    result = train_metadata_model(df, output_dir)

    print("\n✅ Training complete!")
    print("\nModel metrics:")
    for feature, mets in result["metrics"].items():
        print(f"  {feature}: Test MAE={mets['test_mae']:.4f}, R²={mets['test_r2']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
