#!/usr/bin/env python3
"""
Improved KMeans + KNN Algorithm for HeartBeats

Improvements:
1. Expanded feature vector with more audio features
2. Feature weighting (tempo weighted higher for BPM matching)
3. Better query point construction
4. Improved cluster selection
5. Tempo-aware distance metrics
6. Preparation for dynamic k and better clustering algorithms
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

log = logging.getLogger("improved_kmeans_knn")

# ==================== FEATURE CONFIGURATION ====================

# Features for clustering (all features that define a "vibe")
CLUSTERING_FEATURES = [
    "tempo",
    "energy",
    "danceability",
    "valence",
    "loudness",
    "acousticness",
    "instrumentalness",
    # "speechiness",  # Optional - can add if needed
    # "liveness",     # Optional - can add if needed
]

# Features for KNN matching (subset focused on BPM matching)
KNN_FEATURES = [
    "tempo",
    "energy",
    "danceability",
    "valence",
    "loudness",
]

# Feature weights for KNN (higher = more important)
# Tempo should be weighted highest for BPM matching
FEATURE_WEIGHTS = {
    "tempo": 3.0,           # Most important for BPM matching
    "energy": 1.5,          # Important for workout intensity
    "danceability": 1.2,    # Important for rhythm
    "valence": 1.0,         # Mood
    "loudness": 0.8,        # Less critical
    "acousticness": 1.0,    # For clustering
    "instrumentalness": 1.0, # For clustering
}


# ==================== UTILITY FUNCTIONS ====================

def get_available_features(df: pd.DataFrame) -> List[str]:
    """Get list of features that are actually available in the DataFrame."""
    available = []
    all_features = set(CLUSTERING_FEATURES + KNN_FEATURES)
    for feat in all_features:
        if feat in df.columns and df[feat].notna().any():
            available.append(feat)
    return available


def normalize_tempo(tempo: float, allow_doubling: bool = True) -> List[float]:
    """
    Return tempo and optionally 2x/half tempo for matching.
    This handles cases where tracks might be at half/double speed.
    """
    tempos = [tempo]
    if allow_doubling:
        if tempo < 120:  # Likely to have a 2x version
            tempos.append(tempo * 2)
        if tempo > 60:   # Likely to have a half version
            tempos.append(tempo / 2)
    return tempos


def get_feature_weights(feature_names: List[str]) -> np.ndarray:
    """Get weight vector for given feature names."""
    weights = []
    for feat in feature_names:
        weights.append(FEATURE_WEIGHTS.get(feat, 1.0))
    return np.array(weights)


# ==================== IMPROVED CLUSTERING ====================

def determine_optimal_k(
    df: pd.DataFrame,
    k_range: Tuple[int, int] = (2, 10),
    sample_size: Optional[int] = 1000
) -> int:
    """
    Determine optimal number of clusters using silhouette score.

    Args:
        df: DataFrame with features
        k_range: (min_k, max_k) to try
        sample_size: If df is large, sample this many points for speed

    Returns:
        Optimal k value
    """
    features = [f for f in CLUSTERING_FEATURES if f in df.columns]
    if len(features) == 0:
        return 4  # Default

    X = df[features].values.astype(float)

    # Remove any NaN/inf
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]

    if len(X) == 0:
        return 4

    # Sample if too large
    if sample_size and len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X[indices]

    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_scaled = scaler.fit_transform(X)

    best_k = 4
    best_score = -1

    min_k, max_k = k_range

    for k in range(min_k, min(max_k + 1, len(X))):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)

            # Silhouette score requires at least 2 clusters and 2 samples per cluster
            if len(np.unique(labels)) >= 2 and np.min(np.bincount(labels)) >= 2:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        except Exception:
            continue

    log.info(f"Optimal k determined: {best_k} (silhouette score: {best_score:.3f})")
    return best_k


def run_improved_kmeans(
    df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    auto_k: bool = False,
    random_state: int = 42
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, Dict[str, Any]]:
    """
    Run improved KMeans clustering with better feature selection and scaling.

    Returns:
        - df with cluster assignments
        - fitted KMeans model
        - fitted scaler
        - metadata dict with cluster info
    """
    # Determine features to use
    features = [f for f in CLUSTERING_FEATURES if f in df.columns]
    if len(features) == 0:
        raise ValueError("No clustering features available in DataFrame")

    log.info(f"Using {len(features)} features for clustering: {features}")

    # Extract and clean features
    X = df[features].copy()

    # Handle missing values (fill with median)
    for col in X.columns:
        X[col].fillna(X[col].median(), inplace=True)

    X = X.values.astype(float)

    # Remove any infinite values
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    df_filtered = df[mask].copy()

    if len(X) == 0:
        raise ValueError("No valid feature data after cleaning")

    # Determine k
    if auto_k:
        n_clusters = determine_optimal_k(df_filtered)
    elif n_clusters is None:
        n_clusters = 4  # Default

    log.info(f"Running KMeans with k={n_clusters}")

    # Use RobustScaler - handles outliers better than StandardScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Run KMeans with multiple initializations
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,  # Try 10 different initializations
        max_iter=300
    )
    clusters = km.fit_predict(X_scaled)

    # Add cluster assignments
    df_filtered = df_filtered.copy()
    df_filtered["cluster"] = clusters

    # Calculate cluster statistics
    cluster_info = []
    for cid in range(n_clusters):
        cluster_tracks = df_filtered[df_filtered["cluster"] == cid]
        if len(cluster_tracks) > 0:
            cluster_info.append({
                "id": cid,
                "size": len(cluster_tracks),
                "mean_tempo": float(cluster_tracks["tempo"].mean()),
                "std_tempo": float(cluster_tracks["tempo"].std()),
                "mean_energy": float(cluster_tracks["energy"].mean()) if "energy" in cluster_tracks.columns else 0.0,
                "mean_danceability": float(cluster_tracks["danceability"].mean()) if "danceability" in cluster_tracks.columns else 0.0,
                "mean_valence": float(cluster_tracks["valence"].mean()) if "valence" in cluster_tracks.columns else 0.0,
            })

    metadata = {
        "n_clusters": n_clusters,
        "features_used": features,
        "cluster_info": cluster_info,
        "silhouette_score": silhouette_score(X_scaled, clusters) if len(np.unique(clusters)) > 1 else 0.0,
    }

    log.info(f"Clustering complete: {n_clusters} clusters, silhouette={metadata['silhouette_score']:.3f}")

    return df_filtered, km, scaler, metadata


# ==================== IMPROVED KNN ====================

def build_weighted_query_point(
    target_bpm: float,
    df_cluster: pd.DataFrame,
    feature_names: List[str],
    use_cluster_stats: bool = True
) -> np.ndarray:
    """
    Build a better query point for KNN.

    Instead of just using median, we:
    1. Use cluster mean for features (more representative)
    2. Set tempo to target_bpm (obviously)
    3. Optionally adjust other features based on BPM ranges
    """
    query = []

    for feat in feature_names:
        if feat == "tempo":
            query.append(target_bpm)
        else:
            if use_cluster_stats:
                # Use cluster mean instead of median
                query.append(df_cluster[feat].mean())
            else:
                query.append(df_cluster[feat].median())

    return np.array(query).reshape(1, -1)


def improved_knn_in_cluster(
    df: pd.DataFrame,
    scaler: StandardScaler,
    target_bpm: float,
    cluster_id: int,
    topk: int = 10,
    use_weights: bool = True,
    allow_tempo_variants: bool = True,
) -> pd.DataFrame:
    """
    Improved KNN matching with weighted distances and better query point.

    Improvements:
    - Weighted feature distances (tempo weighted higher)
    - Better query point construction
    - Optional tempo variant matching (half/double time)
    """
    # Filter to cluster
    df_cluster = df[df["cluster"] == cluster_id].copy()

    if df_cluster.empty:
        log.warning(f"Cluster {cluster_id} is empty")
        return pd.DataFrame()

    # Determine features to use
    features = [f for f in KNN_FEATURES if f in df_cluster.columns]
    if len(features) == 0:
        features = [f for f in CLUSTERING_FEATURES if f in df_cluster.columns]

    if len(features) == 0:
        raise ValueError("No KNN features available")

    log.info(f"Using {len(features)} features for KNN: {features}")

    # Extract and clean features
    X = df_cluster[features].copy()
    for col in X.columns:
        X[col].fillna(X[col].median(), inplace=True)
    X = X.values.astype(float)

    # Remove invalid rows
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    df_filtered = df_cluster[mask].copy()

    if len(X) == 0:
        return pd.DataFrame()

    # Scale features
    X_scaled = scaler.transform(X)

    # Build query point
    query_point = build_weighted_query_point(target_bpm, df_filtered, features)
    query_scaled = scaler.transform(query_point)

    # Get feature weights
    if use_weights:
        weights = get_feature_weights(features)
        # Apply weights to scaled features
        X_weighted = X_scaled * weights
        query_weighted = query_scaled * weights
    else:
        X_weighted = X_scaled
        query_weighted = query_scaled

    # Fit KNN
    n_neighbors = min(topk, len(X_weighted))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X_weighted)

    # Query
    distances, indices = knn.kneighbors(query_weighted)
    distances = distances[0]
    indices = indices[0]

    # Build results
    results = []
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        row = df_filtered.iloc[idx]
        result = {
            "track_id": row.get("track_id", ""),
            "name": row.get("name", ""),
            "artists": row.get("artists", ""),
            "cluster": cluster_id,
            "tempo": float(row["tempo"]),
            "energy": float(row.get("energy", 0)),
            "danceability": float(row.get("danceability", 0)),
            "valence": float(row.get("valence", 0)),
            "loudness": float(row.get("loudness", 0)),
            "distance": float(dist),
            "rank": rank,
        }
        results.append(result)

    return pd.DataFrame(results)


# ==================== IMPROVED CLUSTER SELECTION ====================

def select_best_cluster(
    df: pd.DataFrame,
    target_bpm: float,
    method: str = "weighted_tempo"
) -> Tuple[int, Dict[str, Any]]:
    """
    Select best cluster for target BPM using improved heuristics.

    Methods:
    - "weighted_tempo": Consider both mean tempo and tempo spread
    - "closest_mean": Simple closest mean tempo (original)
    - "tempo_range": Cluster whose tempo range contains target
    """
    if "cluster" not in df.columns:
        raise ValueError("DataFrame has no 'cluster' column")

    cluster_stats = df.groupby("cluster").agg({
        "tempo": ["mean", "std", "min", "max", "count"]
    }).reset_index()
    cluster_stats.columns = ["cluster", "mean_tempo", "std_tempo", "min_tempo", "max_tempo", "count"]

    if method == "weighted_tempo":
        # Score clusters by: closeness to target, but also consider if target is within range
        cluster_stats["tempo_delta"] = (cluster_stats["mean_tempo"] - target_bpm).abs()
        cluster_stats["in_range"] = (target_bpm >= cluster_stats["min_tempo"]) & (target_bpm <= cluster_stats["max_tempo"])
        cluster_stats["score"] = (
            -cluster_stats["tempo_delta"] * 0.7 +  # Prefer closer mean
            cluster_stats["in_range"].astype(float) * 50.0 +  # Bonus if in range
            -cluster_stats["std_tempo"] * 0.3  # Prefer tighter clusters
        )
    elif method == "tempo_range":
        # Prefer clusters whose range contains target
        in_range = (target_bpm >= cluster_stats["min_tempo"]) & (target_bpm <= cluster_stats["max_tempo"])
        if in_range.any():
            cluster_stats = cluster_stats[in_range]
            cluster_stats["score"] = -cluster_stats["tempo_delta"]
        else:
            cluster_stats["score"] = -cluster_stats["tempo_delta"]
    else:  # closest_mean
        cluster_stats["score"] = -(cluster_stats["mean_tempo"] - target_bpm).abs()

    best = cluster_stats.sort_values("score", ascending=False).iloc[0]

    return int(best["cluster"]), {
        "mean_tempo": float(best["mean_tempo"]),
        "std_tempo": float(best["std_tempo"]),
        "tempo_delta": float(abs(best["mean_tempo"] - target_bpm)),
        "in_range": bool((target_bpm >= best["min_tempo"]) and (target_bpm <= best["max_tempo"])),
        "method": method
    }


# ==================== MAIN WORKFLOW ====================

def run_full_pipeline(
    df: pd.DataFrame,
    target_bpm: Optional[float] = None,
    n_clusters: Optional[int] = None,
    auto_k: bool = False,
    topk: int = 10,
    cluster_selection_method: str = "weighted_tempo"
) -> Dict[str, Any]:
    """
    Run the full improved pipeline:
    1. Cluster tracks with improved KMeans
    2. Select best cluster for target BPM
    3. Run improved KNN to get top matches

    Returns comprehensive results dict.
    """
    log.info("=" * 60)
    log.info("Running improved KMeans + KNN pipeline")
    log.info("=" * 60)

    # Step 1: Clustering
    df_clustered, km, scaler, cluster_metadata = run_improved_kmeans(
        df,
        n_clusters=n_clusters,
        auto_k=auto_k
    )

    results = {
        "clustered_df": df_clustered,
        "kmeans_model": km,
        "scaler": scaler,
        "cluster_metadata": cluster_metadata,
        "target_bpm": target_bpm,
        "matches": None,
        "selected_cluster": None,
    }

    # Step 2: If target BPM provided, find matches
    if target_bpm is not None:
        log.info(f"\nFinding matches for target BPM: {target_bpm}")

        # Select best cluster
        best_cluster, cluster_info = select_best_cluster(
            df_clustered,
            target_bpm,
            method=cluster_selection_method
        )
        results["selected_cluster"] = best_cluster
        results["cluster_selection_info"] = cluster_info

        log.info(f"Selected cluster {best_cluster}: mean_tempo={cluster_info['mean_tempo']:.1f}, "
                f"delta={cluster_info['tempo_delta']:.1f}")

        # Run KNN
        matches = improved_knn_in_cluster(
            df_clustered,
            scaler,
            target_bpm,
            best_cluster,
            topk=topk,
            use_weights=True
        )

        results["matches"] = matches
        log.info(f"Found {len(matches)} matches")

    return results
