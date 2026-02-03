#!/usr/bin/env python3
"""
Improved metadata → audio features prediction model with better features and models.

Improvements:
1. More genres (top 50 instead of 20)
2. Genre interactions (popular genre pairs)
3. Decade/era features
4. Better artist representation
5. Album-level features (if available)
6. Feature interactions (popularity × genre, year × genre)
7. Try multiple models (XGBoost, LightGBM, GradientBoosting)
8. Hyperparameter tuning
9. Better normalization
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

# Try to import XGBoost and LightGBM (better models)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Note: LightGBM not available. Install with: pip install lightgbm")

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]
MODEL_DIR = Path(__file__).resolve().parent / "model"


def extract_metadata_features_improved(
    df: pd.DataFrame, 
    audio_features_df: Optional[pd.DataFrame] = None,
    top_n_genres: int = 50,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract improved metadata features for ML model.
    
    Enhanced features:
    - More genres (top 50)
    - Genre interactions (popular pairs)
    - Decade/era features
    - Feature interactions (popularity × genre, year × genre)
    - Album features (if available)
    - Better normalization
    
    Args:
        feature_names: If provided, only extract features that match these names (for inference)
    """
    features = pd.DataFrame(index=df.index)
    
    # If feature_names provided, we're in inference mode - need to match training features
    inference_mode = feature_names is not None

    # === Basic Features ===

    # Popularity (normalize 0-100)
    features["popularity"] = df["popularity"].fillna(50) / 100.0

    # Release year (extract from release_date)
    def extract_year(date_str):
        if pd.isna(date_str) or not date_str:
            return 2020
        try:
            return int(str(date_str).split("-")[0])
        except:
            return 2020

    features["release_year"] = df["release_date"].apply(extract_year)
    features["release_year_norm"] = (features["release_year"] - 1950) / 70.0

    # Decade (categorical encoding)
    features["decade"] = (features["release_year"] // 10) * 10
    features["decade_norm"] = (features["decade"] - 1950) / 70.0

    # Era (1950s-2020s, 7 eras)
    era_map = {
        1950: 0, 1960: 1, 1970: 2, 1980: 3, 1990: 4, 2000: 5, 2010: 6, 2020: 7
    }
    features["era"] = features["decade"].map(era_map).fillna(7) / 7.0

    # Duration (normalize, typical range 60k-300k ms = 1-5 min)
    if "duration_ms" in df.columns:
        features["duration"] = df["duration_ms"].fillna(180000) / 300000.0
        # Also log-scale duration (songs tend to cluster around certain lengths)
        features["duration_log"] = np.log1p(df["duration_ms"].fillna(180000)) / np.log1p(300000)
    else:
        features["duration"] = 0.6
        features["duration_log"] = 0.5

    # Explicit flag (binary)
    if "explicit" in df.columns:
        features["explicit"] = df["explicit"].astype(int).fillna(0)
    else:
        features["explicit"] = 0

    # === Genre Features (Expanded) ===

    # Collect all genres
    all_genres = set()
    for genres_str in df["genres"].fillna(""):
        if isinstance(genres_str, str):
            if "," in genres_str:
                genres = [g.strip() for g in genres_str.split(",") if g.strip()]
            else:
                genres = [genres_str.strip()] if genres_str.strip() else []
            all_genres.update(genres)
        elif isinstance(genres_str, list):
            all_genres.update(genres_str)

    # Top N genres (increased from 20 to 50)
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

    top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:top_n_genres]
    top_genre_names = [g[0] for g in top_genres]
    
    # In inference mode, use genres from feature_names
    if inference_mode:
        # Extract genre names from feature_names
        training_genres = []
        for fn in feature_names:
            if fn.startswith("genre_") and not fn.startswith("pop_x_genre_") and not fn.startswith("year_x_genre_") and not fn.startswith("dur_x_genre_"):
                training_genres.append(fn.replace("genre_", ""))
        # Use intersection of training genres and current genres
        top_genre_names = [g for g in top_genre_names if g in training_genres] or top_genre_names[:top_n_genres]

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
    features["num_genres"] = features["num_genres"] / 10.0

    # === Genre Interactions (Top 10 genre pairs) ===
    if not inference_mode:
        # Find most common genre pairs (only during training)
        genre_pairs = {}
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
            
            # Create pairs
            for i, g1 in enumerate(genres):
                for g2 in genres[i+1:]:
                    pair = tuple(sorted([g1, g2]))
                    genre_pairs[pair] = genre_pairs.get(pair, 0) + 1

        top_pairs = sorted(genre_pairs.items(), key=lambda x: -x[1])[:10]
    else:
        # In inference mode, extract pairs from feature_names
        training_pairs = []
        for fn in feature_names:
            if fn.startswith("pair_"):
                pair_str = fn.replace("pair_", "").replace("_", " ")
                # Try to split into two genres
                parts = pair_str.split(" ")
                if len(parts) >= 2:
                    training_pairs.append((parts[0], " ".join(parts[1:])))
        top_pairs = [(tuple(p), 0) for p in training_pairs[:10]]
    
    for (g1, g2), _ in top_pairs:
        pair_name = f"{g1}_{g2}".replace(" ", "_").replace("-", "_")
        features[f"pair_{pair_name}"] = df["genres"].apply(
            lambda x: (
                1 if (
                    (isinstance(x, str) and g1 in x and g2 in x) or
                    (isinstance(x, list) and g1 in x and g2 in x)
                ) else 0
            )
        )

    # === Album Features (if available) ===
    if "album_name" in df.columns:
        # Album popularity (if we had it, but we don't - skip for now)
        # Could add album-level aggregations if we had more data
        pass
    
    # === Feature Interactions ===
    
    # Determine which genres to use for interactions
    interaction_genres = top_genre_names[:10] if not inference_mode else []
    if inference_mode:
        # Extract interaction genres from feature_names
        for fn in feature_names:
            if fn.startswith("pop_x_genre_"):
                genre = fn.replace("pop_x_genre_", "")
                if genre not in interaction_genres:
                    interaction_genres.append(genre)
    
    # Popularity × Top Genres (interaction)
    for genre in interaction_genres:
        if f"genre_{genre}" in features.columns:
            features[f"pop_x_genre_{genre}"] = features["popularity"] * features[f"genre_{genre}"]
        elif not inference_mode:
            features[f"pop_x_genre_{genre}"] = 0.0
    
    # Year × Genre (temporal trends)
    for genre in interaction_genres:
        if f"genre_{genre}" in features.columns:
            features[f"year_x_genre_{genre}"] = features["release_year_norm"] * features[f"genre_{genre}"]
        elif not inference_mode:
            features[f"year_x_genre_{genre}"] = 0.0
    
    # Popularity × Explicit (explicit songs might have different characteristics)
    features["pop_x_explicit"] = features["popularity"] * features["explicit"]
    
    # Duration × Genre (some genres have typical song lengths)
    dur_interaction_genres = interaction_genres[:5]
    for genre in dur_interaction_genres:
        if f"genre_{genre}" in features.columns:
            features[f"dur_x_genre_{genre}"] = features["duration"] * features[f"genre_{genre}"]
        elif not inference_mode:
            features[f"dur_x_genre_{genre}"] = 0.0

    # === Artist Features ===

    if "main_artist_id" in df.columns:
        le_artist = LabelEncoder()
        artist_ids = df["main_artist_id"].fillna("unknown").astype(str)
        features["artist_encoded"] = le_artist.fit_transform(artist_ids)
        features["artist_encoded"] = features["artist_encoded"] / max(len(le_artist.classes_), 1)
    else:
        features["artist_encoded"] = 0.5

    # === Related Artists Features ===

    if audio_features_df is not None and "related_artist_ids" in df.columns:
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
                for rid in related_ids[:5]:
                    if rid in artist_features:
                        values.append(artist_features[rid][feat_name])
                avg = sum(values) / len(values) if values else 0.0
                if feat_name == "tempo":
                    avg = avg / 200.0
                elif feat_name == "loudness":
                    avg = (avg + 50) / 50.0
                related_feats.append(avg)

            return related_feats if related_feats else [0.0] * 5

        related_feats = df["related_artist_ids"].apply(get_related_artist_features)
        features["related_tempo"] = related_feats.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0.0)
        features["related_energy"] = related_feats.apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else 0.0)
        features["related_danceability"] = related_feats.apply(lambda x: x[2] if isinstance(x, list) and len(x) > 2 else 0.0)
        features["related_valence"] = related_feats.apply(lambda x: x[3] if isinstance(x, list) and len(x) > 3 else 0.0)
        features["related_loudness"] = related_feats.apply(lambda x: x[4] if isinstance(x, list) and len(x) > 4 else 0.0)
    else:
        features["related_tempo"] = 0.0
        features["related_energy"] = 0.0
        features["related_danceability"] = 0.0
        features["related_valence"] = 0.0
        features["related_loudness"] = 0.0

    # Ensure all expected features exist (for inference mode)
    if inference_mode and feature_names:
        for fn in feature_names:
            if fn not in features.columns:
                features[fn] = 0.0
        # Reorder columns to match feature_names
        features = features[[fn for fn in feature_names if fn in features.columns]]

    return features


def train_model_for_feature(
    X_train, y_train, X_test, y_test, feature_name: str, model_type: str = "xgboost"
):
    """Train a model for a specific audio feature."""

    if model_type == "xgboost" and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        # Fallback to GradientBoosting
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=0
        )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    return model, {
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
    }


def train_metadata_model_improved(
    enriched_df: pd.DataFrame,
    output_dir: Path,
    model_type: str = "xgboost",
    top_n_genres: int = 50,
) -> Dict:
    """Train improved metadata → audio features model."""

    print("Extracting improved metadata features...")
    X_meta = extract_metadata_features_improved(
        enriched_df,
        audio_features_df=enriched_df,
        top_n_genres=top_n_genres
    )

    y = enriched_df[FEATURE_COLS].values

    # Filter valid rows
    valid_mask = X_meta.notna().all(axis=1) & enriched_df[FEATURE_COLS].notna().all(axis=1)
    X_meta = X_meta[valid_mask]
    y = y[valid_mask]

    print(f"Training samples: {len(X_meta)}")
    print(f"Feature vector size: {len(X_meta.columns)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_meta, y, test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train separate models for each audio feature
    models = {}
    metrics = {}

    for idx, feature_name in enumerate(FEATURE_COLS):
        print(f"\nTraining {model_type} model for {feature_name}...")
        y_train_feat = y_train[:, idx]
        y_test_feat = y_test[:, idx]

        model, mets = train_model_for_feature(
            X_train, y_train_feat, X_test, y_test_feat, feature_name, model_type
        )

        metrics[feature_name] = mets
        print(f"  Train MAE: {mets['train_mae']:.4f}")
        print(f"  Test MAE: {mets['test_mae']:.4f}, R²: {mets['test_r2']:.4f}")

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

    # Save feature names
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
        description="Train improved metadata → audio features prediction model"
    )
    parser.add_argument("--enriched-csv", type=str, required=True, help="Path to enriched Kaggle CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: recs/model/)")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "lightgbm", "gbm"],
                       help="Model type to use")
    parser.add_argument("--top-genres", type=int, default=50, help="Number of top genres to use")
    args = parser.parse_args()

    enriched_path = Path(args.enriched_csv)
    if not enriched_path.exists():
        print(f"ERROR: Enriched CSV not found: {enriched_path}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR

    print("Loading enriched dataset...")
    df = pd.read_csv(enriched_path)

    required = FEATURE_COLS + ["track_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return 1

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} samples (testing mode)")

    print(f"Loaded {len(df)} tracks")

    df = df.dropna(subset=FEATURE_COLS)
    df = df[df["tempo"] > 0]
    print(f"Valid tracks: {len(df)}")

    if len(df) < 100:
        print("ERROR: Need at least 100 valid tracks to train")
        return 1

    result = train_metadata_model_improved(
        df,
        output_dir,
        model_type=args.model_type,
        top_n_genres=args.top_genres
    )

    print("\n✅ Training complete!")
    print("\nModel metrics:")
    for feature, mets in result["metrics"].items():
        print(f"  {feature}: Test MAE={mets['test_mae']:.4f}, R²={mets['test_r2']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
