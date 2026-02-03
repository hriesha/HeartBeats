"""
Inference: classify a track into a cluster and run KNN.

Two modes:
1. Fast lookup: track_id -> cluster + neighbors (for tracks in training set)
2. Prediction: audio features -> cluster (for ANY track, even if not in training set)

Loads the precomputed track lookup (track_id -> cluster + scaled embedding) built at training.
Also loads scaler + centroids to predict clusters for new tracks.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import joblib
except ImportError:
    joblib = None

MODEL_DIR = Path(__file__).resolve().parent / "model"
TRACK_LOOKUP_DB = "track_lookup.db"
SCALER_FILE = "scaler.joblib"
CENTROIDS_FILE = "centroids.npy"
CONFIG_FILE = "config.json"

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]


def load_track_row(conn: sqlite3.Connection, track_id: str) -> Optional[Tuple[int, np.ndarray]]:
    """Return (cluster, embedding) for track_id or None if not in lookup."""
    row = conn.execute(
        "SELECT cluster, f0, f1, f2, f3, f4 FROM track_lookup WHERE track_id = ?",
        (str(track_id),),
    ).fetchone()
    if row is None:
        return None
    cluster = int(row[0])
    emb = np.array([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
    return cluster, emb


def get_cluster_and_neighbors(
    track_id: str,
    top_k: int = 10,
    model_dir: Optional[Path | str] = None,
) -> Optional[Tuple[int, List[str]]]:
    """
    Classify track into cluster and get KNN neighbor track_ids using only track_id (no audio features).

    Returns (cluster_id, list of neighbor track_ids) or None if track_id not in lookup.
    """
    model_dir = Path(model_dir) if model_dir else MODEL_DIR
    db_path = model_dir / TRACK_LOOKUP_DB
    if not db_path.exists():
        raise FileNotFoundError(f"Track lookup not found: {db_path}. Run training first.")

    conn = sqlite3.connect(str(db_path))
    try:
        row = load_track_row(conn, track_id)
        if row is None:
            return None
        cluster, emb = row

        # All tracks in same cluster (excluding self)
        rows = conn.execute(
            "SELECT track_id, f0, f1, f2, f3, f4 FROM track_lookup WHERE cluster = ? AND track_id != ?",
            (cluster, str(track_id)),
        ).fetchall()
        if not rows:
            return cluster, []

        # Euclidean distance to embedding
        ids = [r[0] for r in rows]
        vecs = np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows])
        dists = np.linalg.norm(vecs - emb, axis=1)
        order = np.argsort(dists)[:top_k]
        neighbors = [ids[i] for i in order]
        return cluster, neighbors
    finally:
        conn.close()


def get_cluster_only(
    track_id: str,
    model_dir: Optional[Path | str] = None,
) -> Optional[int]:
    """Return cluster_id for track_id or None if not in lookup."""
    model_dir = Path(model_dir) if model_dir else MODEL_DIR
    db_path = model_dir / TRACK_LOOKUP_DB
    if not db_path.exists():
        raise FileNotFoundError(f"Track lookup not found: {db_path}. Run training first.")
    conn = sqlite3.connect(str(db_path))
    try:
        row = load_track_row(conn, track_id)
        return int(row[0]) if row else None
    finally:
        conn.close()


def predict_cluster_from_features(
    features: dict,
    model_dir: Optional[Path | str] = None,
) -> Optional[int]:
    """
    Predict cluster for a track using audio features (even if track not in training set).

    Args:
        features: dict with keys: tempo, energy, danceability, valence, loudness
        model_dir: path to model directory (default: recs/model/)

    Returns:
        cluster_id (int) or None if model files not found
    """
    model_dir = Path(model_dir) if model_dir else MODEL_DIR

    # Load scaler and centroids
    scaler_path = model_dir / SCALER_FILE
    centroids_path = model_dir / CENTROIDS_FILE

    if not scaler_path.exists() or not centroids_path.exists():
        return None

    try:
        if joblib is not None:
            scaler = joblib.load(str(scaler_path))
        else:
            import pickle
            with open(str(scaler_path), "rb") as f:
                scaler = pickle.load(f)

        centroids = np.load(str(centroids_path))

        # Extract features in correct order
        feature_vector = np.array([
            float(features.get("tempo", 0)),
            float(features.get("energy", 0)),
            float(features.get("danceability", 0)),
            float(features.get("valence", 0)),
            float(features.get("loudness", 0)),
        ]).reshape(1, -1)

        # Scale and predict (assign to nearest centroid)
        scaled = scaler.transform(feature_vector)
        distances = np.linalg.norm(centroids - scaled, axis=1)
        cluster_id = int(np.argmin(distances))

        return cluster_id
    except Exception as e:
        print(f"Error predicting cluster from features: {e}")
        return None


def predict_audio_features_from_metadata(
    metadata: dict,
    model_dir: Optional[Path | str] = None,
    training_df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Predict audio features from Spotify metadata (for tracks not in training set).
    
    Tries improved model first, falls back to basic model.
    """
    # Try improved model first (if available)
    try:
        from recs.inference_improved import predict_audio_features_from_metadata_improved
        result = predict_audio_features_from_metadata_improved(metadata, model_dir, training_df)
        if result:
            return result
    except Exception as e:
        # If improved model fails, fall back to basic implementation
        pass
    
    # Fallback to original implementation
    """
    Predict audio features from Spotify metadata (for tracks not in training set).

    Args:
        metadata: dict with keys like:
            - name, artists, artist_names
            - popularity, release_date, duration_ms, explicit
            - genres (list or comma-separated string)
            - main_artist_id, related_artist_ids (optional)
        model_dir: path to model directory
        training_df: Optional DataFrame with training data (for related artists features)

    Returns:
        dict with predicted audio features (tempo, energy, danceability, valence, loudness)
        or None if model not found
    """
    model_dir = Path(model_dir) if model_dir else MODEL_DIR

    # Load feature names
    feature_names_path = model_dir / "metadata_feature_names.json"
    if not feature_names_path.exists():
        return None

    try:
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
    except Exception:
        return None

    # Build feature vector (same as training)
    features = {}

    # Popularity
    features["popularity"] = float(metadata.get("popularity", 50)) / 100.0

    # Release year
    release_date = metadata.get("release_date", "")
    try:
        year = int(str(release_date).split("-")[0]) if release_date else 2020
    except:
        year = 2020
    features["release_year"] = (year - 1950) / 70.0

    # Duration
    duration_ms = metadata.get("duration_ms", 180000)
    features["duration"] = float(duration_ms) / 300000.0

    # Explicit
    features["explicit"] = 1 if metadata.get("explicit", False) else 0

    # Artist (simplified - use 0.5 if not available, would need saved label encoder)
    features["artist_encoded"] = 0.5

    # Genres (handle both single genre string and comma-separated/list)
    genres_str = metadata.get("genres", "")
    if isinstance(genres_str, list):
        genres = genres_str
    elif isinstance(genres_str, str):
        # Try comma-separated first, else treat as single genre
        if "," in genres_str:
            genres = [g.strip() for g in genres_str.split(",") if g.strip()]
        else:
            genres = [genres_str.strip()] if genres_str.strip() else []
    else:
        genres = []

    # Initialize all genre features to 0
    for feat_name in feature_names:
        if feat_name.startswith("genre_"):
            genre_name = feat_name.replace("genre_", "")
            # Check if genre matches (handle both comma-separated and single)
            matches = (
                (isinstance(genres_str, str) and "," in genres_str and genre_name in genres_str.split(",")) or
                (isinstance(genres_str, str) and genre_name == genres_str.strip()) or
                (isinstance(genres_str, list) and genre_name in genres_str) or
                (genre_name in genres)
            )
            features[feat_name] = 1 if matches else 0

    features["num_genres"] = (
        len(genres_str.split(",")) if isinstance(genres_str, str) and "," in genres_str else
        (1 if isinstance(genres_str, str) and genres_str.strip() else 0) if isinstance(genres_str, str) else
        (len(genres_str) if isinstance(genres_str, list) else 0)
    ) / 10.0

    # Related artists features (if training data available)
    if training_df is not None and "related_artist_ids" in metadata:
        # Build artist -> avg features mapping
        artist_features = {}
        if "main_artist_id" in training_df.columns and FEATURE_COLS[0] in training_df.columns:
            for artist_id in training_df["main_artist_id"].dropna().unique():
                artist_tracks = training_df[training_df["main_artist_id"] == artist_id]
                if len(artist_tracks) > 0:
                    artist_features[str(artist_id)] = {
                        "tempo": artist_tracks[FEATURE_COLS[0]].mean() / 200.0,
                        "energy": artist_tracks[FEATURE_COLS[1]].mean(),
                        "danceability": artist_tracks[FEATURE_COLS[2]].mean(),
                        "valence": artist_tracks[FEATURE_COLS[3]].mean(),
                        "loudness": (artist_tracks[FEATURE_COLS[4]].mean() + 50) / 50.0,
                    }

        # Get related artist IDs
        related_ids = metadata.get("related_artist_ids", [])
        if isinstance(related_ids, str):
            related_ids = [id.strip() for id in related_ids.split(",") if id.strip()]

        # Compute averages
        related_feats = {"tempo": 0.0, "energy": 0.0, "danceability": 0.0, "valence": 0.0, "loudness": 0.0}
        count = 0
        for rid in related_ids[:5]:
            if rid in artist_features:
                for k, v in artist_features[rid].items():
                    related_feats[k] += v
                count += 1

        if count > 0:
            for k in related_feats:
                related_feats[k] /= count

        features["related_tempo"] = related_feats["tempo"]
        features["related_energy"] = related_feats["energy"]
        features["related_danceability"] = related_feats["danceability"]
        features["related_valence"] = related_feats["valence"]
        features["related_loudness"] = related_feats["loudness"]
    else:
        # Defaults
        features["related_tempo"] = 0.0
        features["related_energy"] = 0.0
        features["related_danceability"] = 0.0
        features["related_valence"] = 0.0
        features["related_loudness"] = 0.0

    # Build feature vector in correct order
    X = np.array([[features.get(name, 0) for name in feature_names]]).reshape(1, -1)

    # Load models and predict
    predicted_features = {}

    for feature_name in FEATURE_COLS:
        model_path = model_dir / f"metadata_model_{feature_name}.joblib"
        if not model_path.exists():
            model_path = model_dir / f"metadata_model_{feature_name}.pkl"

        if not model_path.exists():
            return None

        try:
            if joblib is not None:
                model = joblib.load(str(model_path))
            else:
                import pickle
                with open(str(model_path), "rb") as f:
                    model = pickle.load(f)

            pred = model.predict(X)[0]
            predicted_features[feature_name] = float(pred)
        except Exception as e:
            print(f"Error predicting {feature_name}: {e}")
            return None

    return predicted_features


def get_cluster_for_track(
    track_id: str,
    audio_features: Optional[dict] = None,
    metadata: Optional[dict] = None,
    model_dir: Optional[Path | str] = None,
) -> Optional[int]:
    """
    Get cluster for a track: try lookup first, then predict from features/metadata.

    Priority:
    1. Lookup (fast path for tracks in training set)
    2. Predict from audio_features if provided
    3. Predict from metadata if provided

    Args:
        track_id: Spotify track ID
        audio_features: Optional dict with tempo, energy, danceability, valence, loudness
        metadata: Optional dict with Spotify metadata (name, artists, genres, popularity, etc.)
        model_dir: path to model directory

    Returns:
        cluster_id (int) or None if not found and no features/metadata provided
    """
    # Try lookup first (fast path for tracks in training set)
    cluster = get_cluster_only(track_id, model_dir)
    if cluster is not None:
        return cluster

    # If not in lookup, predict from audio features if provided
    if audio_features:
        return predict_cluster_from_features(audio_features, model_dir)

    # Otherwise, predict from metadata
    if metadata:
        predicted_features = predict_audio_features_from_metadata(metadata, model_dir)
        if predicted_features:
            return predict_cluster_from_features(predicted_features, model_dir)

    return None
