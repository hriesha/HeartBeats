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

# Import cluster naming
try:
    from algorithms.cluster_naming import generate_cluster_names
    CLUSTER_NAMING_AVAILABLE = True
except ImportError:
    try:
        from cluster_naming import generate_cluster_names
        CLUSTER_NAMING_AVAILABLE = True
    except ImportError:
        log.warning("Cluster naming module not available")
        CLUSTER_NAMING_AVAILABLE = False

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
FEATURE_WEIGHTS_KNN = {
    "tempo": 3.0,           # Most important for BPM matching
    "energy": 1.5,          # Important for workout intensity
    "danceability": 1.2,    # Important for rhythm
    "valence": 1.0,         # Mood
    "loudness": 0.8,        # Less critical
}

# Feature weights for CLUSTERING (tempo should dominate)
# These are base weights - tempo weight will be amplified based on BPM
FEATURE_WEIGHTS_CLUSTERING = {
    "tempo": 5.0,           # MUCH higher weight for clustering (BPM is primary)
    "energy": 1.0,
    "danceability": 1.0,
    "valence": 1.0,
    "loudness": 0.8,
    "acousticness": 0.9,
    "instrumentalness": 0.9,
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


def get_feature_weights(feature_names: List[str], for_clustering: bool = False) -> np.ndarray:
    """Get weight vector for given feature names."""
    weight_dict = FEATURE_WEIGHTS_CLUSTERING if for_clustering else FEATURE_WEIGHTS_KNN
    weights = []
    for feat in feature_names:
        weights.append(weight_dict.get(feat, 1.0))
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


def filter_tracks_by_bpm(
    df: pd.DataFrame,
    target_bpm: float,
    bpm_tolerance: float = 30.0,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    STEP 1: Filter tracks by BPM using KNN.

    First, we find tracks with tempo close to the target BPM.
    This narrows down the dataset before clustering.

    Args:
        df: Full track DataFrame
        target_bpm: Target BPM to filter around
        bpm_tolerance: Maximum BPM distance (default ±30 BPM)
        top_n: If provided, return top N closest tracks (None = all within tolerance)

    Returns:
        Filtered DataFrame with tracks near target BPM
    """
    if "tempo" not in df.columns:
        raise ValueError("DataFrame must have 'tempo' column")

    # Calculate BPM distance
    df = df.copy()
    df["bpm_distance"] = np.abs(df["tempo"] - target_bpm)

    # Filter by tolerance
    if top_n is None:
        # Return all tracks within tolerance
        filtered = df[df["bpm_distance"] <= bpm_tolerance].copy()
        log.info(f"Filtered {len(filtered)} tracks within ±{bpm_tolerance} BPM of {target_bpm} (from {len(df)} total)")
    else:
        # Return top N closest tracks
        filtered = df.nsmallest(min(top_n, len(df)), "bpm_distance").copy()
        log.info(f"Selected top {len(filtered)} tracks closest to {target_bpm} BPM (from {len(df)} total)")

    # Drop helper column
    if "bpm_distance" in filtered.columns:
        filtered = filtered.drop(columns=["bpm_distance"])

    return filtered


def run_improved_kmeans(
    df: pd.DataFrame,
    target_bpm: Optional[float] = None,
    n_clusters: Optional[int] = None,
    auto_k: bool = False,
    random_state: Optional[int] = None,
    bpm_filter_first: bool = True,
    bpm_tolerance: float = 30.0
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, Dict[str, Any]]:
    """
    Run improved KMeans clustering.

    Flow:
    1. If target_bpm provided: Filter tracks by BPM first (KNN-style filter)
    2. Cluster filtered tracks by all audio features (creates "vibes")
    3. KNN is run separately within each cluster to find best BPM matches

    Args:
        target_bpm: Target BPM - if provided, filters tracks first, then clusters
        bpm_filter_first: If True and target_bpm provided, filter by BPM before clustering
        bpm_tolerance: BPM range for filtering (default ±30 BPM)

    Returns:
        - df with cluster assignments (filtered if bpm_filter_first=True)
        - fitted KMeans model
        - fitted scaler
        - metadata dict with cluster info
    """
    # STEP 1: Filter by BPM if target provided
    original_size = len(df)
    if target_bpm and bpm_filter_first:
        df = filter_tracks_by_bpm(df, target_bpm, bpm_tolerance=bpm_tolerance)
        log.info(f"BPM filter: {len(df)} tracks (from {original_size}) near {target_bpm} BPM")

        if len(df) < 10:
            log.warning(f"Very few tracks ({len(df)}) after BPM filtering. Consider increasing tolerance.")

    # Determine features to use (all features, not just tempo)
    features = [f for f in CLUSTERING_FEATURES if f in df.columns]
    if len(features) == 0:
        raise ValueError("No clustering features available in DataFrame")

    log.info(f"Clustering {len(df)} tracks using {len(features)} features: {features}")

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

    # No tempo transformation needed - we already filtered by BPM
    # Now we cluster by ALL features equally to create meaningful "vibes"

    # Determine k - default to auto if not specified
    if n_clusters is None:
        auto_k = True  # Always use dynamic k by default
    if auto_k:
        n_clusters = determine_optimal_k(df_filtered)
        log.info(f"Auto-determined optimal k: {n_clusters} clusters")
    elif n_clusters is None:
        n_clusters = 4  # Fallback default

    log.info(f"Running KMeans with k={n_clusters}, BPM-aware: {target_bpm is not None}")

    # Use RobustScaler - handles outliers better than StandardScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # No special weighting needed - we already filtered by BPM
    # Now cluster by all features equally to create different "vibes"
    # Each cluster represents a different musical style/mood within the BPM range
    X_weighted = X_scaled

    # Use consistent random_state for reproducibility
    clustering_random_state = random_state or 42

    log.info(f"Clustering {len(X_weighted)} filtered tracks into {n_clusters} vibes based on all features")

    # Run KMeans with multiple initializations
    km = KMeans(
        n_clusters=n_clusters,
        random_state=clustering_random_state,
        n_init=10,  # Try 10 different initializations
        max_iter=300
    )
    clusters = km.fit_predict(X_weighted)

    # Add cluster assignments
    df_filtered = df_filtered.copy()
    df_filtered["cluster"] = clusters

    # Generate dynamic cluster names and metadata
    if CLUSTER_NAMING_AVAILABLE:
        try:
            cluster_info_list = generate_cluster_names(
                df_filtered,
                user_library_size=len(df_filtered)
            )
            # Convert to simple list format for metadata
            cluster_info = [
                {
                    "id": c["id"],
                    "size": c["track_count"],
                    "name": c["name"],
                    "tags": c["tags"],
                    "color": c["color"],
                    "mean_tempo": c["mean_tempo"],
                    "std_tempo": float(df_filtered[df_filtered["cluster"] == c["id"]]["tempo"].std()),
                    "mean_energy": c["mean_energy"],
                    "mean_danceability": c["mean_danceability"],
                    "mean_valence": c["mean_valence"],
                }
                for c in cluster_info_list
            ]
            log.info(f"Generated dynamic names for {len(cluster_info)} clusters")
        except Exception as e:
            log.warning(f"Dynamic cluster naming failed, using defaults: {e}")
            # Fallback to basic naming
            cluster_info = []
            color_palette = ["#EAE2B7", "#FCBF49", "#F77F00", "#D62828", "#003049"]
            for cid in range(n_clusters):
                cluster_tracks = df_filtered[df_filtered["cluster"] == cid]
                if len(cluster_tracks) > 0:
                    cluster_info.append({
                        "id": cid,
                        "size": len(cluster_tracks),
                        "name": f"Cluster {cid + 1}",
                        "tags": [],
                        "color": color_palette[cid % len(color_palette)],
                        "mean_tempo": float(cluster_tracks["tempo"].mean()),
                        "std_tempo": float(cluster_tracks["tempo"].std()),
                        "mean_energy": float(cluster_tracks["energy"].mean()) if "energy" in cluster_tracks.columns else 0.0,
                        "mean_danceability": float(cluster_tracks["danceability"].mean()) if "danceability" in cluster_tracks.columns else 0.0,
                        "mean_valence": float(cluster_tracks["valence"].mean()) if "valence" in cluster_tracks.columns else 0.0,
                    })
    else:
        # Fallback: basic naming
        cluster_info = []
        color_palette = ["#EAE2B7", "#FCBF49", "#F77F00", "#D62828", "#003049"]
        for cid in range(n_clusters):
            cluster_tracks = df_filtered[df_filtered["cluster"] == cid]
            if len(cluster_tracks) > 0:
                cluster_info.append({
                    "id": cid,
                    "size": len(cluster_tracks),
                    "name": f"Cluster {cid + 1}",
                    "tags": [],
                    "color": color_palette[cid % len(color_palette)],
                    "mean_tempo": float(cluster_tracks["tempo"].mean()),
                    "std_tempo": float(cluster_tracks["tempo"].std()),
                    "mean_energy": float(cluster_tracks["energy"].mean()) if "energy" in cluster_tracks.columns else 0.0,
                    "mean_danceability": float(cluster_tracks["danceability"].mean()) if "danceability" in cluster_tracks.columns else 0.0,
                    "mean_valence": float(cluster_tracks["valence"].mean()) if "valence" in cluster_tracks.columns else 0.0,
                })

    # Calculate silhouette on weighted features if BPM-aware
    score_data = X_weighted if target_bpm else X_scaled
    # Calculate silhouette on scaled features
    silhouette = silhouette_score(X_scaled, clusters) if len(np.unique(clusters)) > 1 else 0.0

    metadata = {
        "n_clusters": n_clusters,
        "features_used": features,
        "cluster_info": cluster_info,
        "silhouette_score": silhouette,
        "target_bpm": target_bpm,
        "bpm_filtered": target_bpm is not None and bpm_filter_first,
        "original_track_count": original_size,
        "filtered_track_count": len(df_filtered),
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
    scaler,
    target_bpm: float,
    cluster_id: int,
    topk: int = 10,
    use_weights: bool = True,
    allow_tempo_variants: bool = True,
    clustering_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    STEP 3: Run KNN within a specific cluster to find best BPM matches.

    This is the final step:
    1. We already filtered by BPM (step 1)
    2. We already clustered into vibes (step 2)
    3. Now we fine-tune within each cluster to find top-K songs closest to target BPM
    """
    """
    Improved KNN matching with weighted distances and better query point.

    Improvements:
    - Weighted feature distances (tempo weighted higher)
    - Better query point construction
    - Optional tempo variant matching (half/double time)

    Args:
        clustering_features: The features that were used for clustering (must match scaler).
                            If None, will try to infer from available features.
    """
    # Filter to cluster
    df_cluster = df[df["cluster"] == cluster_id].copy()

    if df_cluster.empty:
        log.warning(f"Cluster {cluster_id} is empty")
        return pd.DataFrame()

    # Use the same features that were used for clustering (to match scaler)
    if clustering_features:
        features = [f for f in clustering_features if f in df_cluster.columns]
    else:
        # Try to infer from CLUSTERING_FEATURES
        features = [f for f in CLUSTERING_FEATURES if f in df_cluster.columns]

    # Fallback to KNN_FEATURES if needed
    if len(features) == 0:
        features = [f for f in KNN_FEATURES if f in df_cluster.columns]

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

    # Get feature weights (for KNN, not clustering)
    if use_weights:
        weights = get_feature_weights(features, for_clustering=False)
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
