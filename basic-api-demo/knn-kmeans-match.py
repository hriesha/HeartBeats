#!/usr/bin/env python3
# knn-kmeans-match.py
#
# Unified workflow:
#  1. Load audio features CSV
#  2. Run KMeans clustering on 5D feature vector
#  3. Given a heart rate (BPM):
#     a. Pick the cluster with mean tempo closest to HR
#     b. Use KNN to find top-K songs in that cluster (by 5D distance)
#  4. Return ranked songs with distances and metadata

import logging
import argparse
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# ---------------------------
# Setup
# ---------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("heartbeats-knn-kmeans")

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]


# ---------------------------
# Helpers
# ---------------------------

def load_features(csv_path: str) -> pd.DataFrame:
    """
    Load features CSV. Expects at least:
      track_id, name, artists, tempo, energy, danceability, valence, loudness
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def run_kmeans(df: pd.DataFrame, n_clusters: int, random_state: int = 42) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Run KMeans on the 5D feature vector and return:
      - df with a new 'cluster' column
      - the fitted KMeans object
      - the StandardScaler (for consistent scaling)
    """
    X = df[FEATURE_COLS].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = km.fit_predict(X_scaled)

    df = df.copy()
    df["cluster"] = clusters
    return df, km, scaler


def pick_cluster_by_tempo(df: pd.DataFrame, target_bpm: float) -> int:
    """
    Simple heuristic: choose the cluster whose mean tempo is closest to target_bpm.
    """
    if "cluster" not in df.columns:
        raise ValueError("DataFrame has no 'cluster' column")

    cluster_stats = (
        df.groupby("cluster")["tempo"]
        .mean()
        .reset_index()
        .rename(columns={"tempo": "mean_tempo"})
    )

    cluster_stats["tempo_delta"] = (cluster_stats["mean_tempo"] - target_bpm).abs()
    row = cluster_stats.sort_values("tempo_delta").iloc[0]
    best_cluster = int(row["cluster"])
    best_tempo = float(row["mean_tempo"])
    tempo_delta = float(row["tempo_delta"])
    
    return best_cluster, best_tempo, tempo_delta


def knn_in_cluster(
    df: pd.DataFrame,
    scaler: StandardScaler,
    target_bpm: float,
    topk: int = 10,
    cluster_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Use KNN to find the top-K songs closest to the target heart rate in feature space,
    optionally restricted to a specific cluster.

    Steps:
      1. Filter df to cluster (if specified), or use all songs
      2. Scale features using the provided scaler
      3. Build KNN model on scaled features
      4. Find the "query point" as the scaled representation of target HR
         (construct a pseudo-song with HR as tempo and average other features)
      5. Return top-K with distances and metadata

    Returns DataFrame with columns:
      track_id, name, artists, cluster, tempo, energy, danceability, valence, loudness,
      distance, rank
    """
    if "cluster" not in df.columns:
        raise ValueError("DataFrame has no 'cluster' column")

    # Filter to cluster if specified
    if cluster_id is not None:
        df_filtered = df[df["cluster"] == cluster_id].copy()
        if df_filtered.empty:
            log.warning(f"Cluster {cluster_id} is empty; returning empty result.")
            return pd.DataFrame()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        return pd.DataFrame()

    # Extract and scale features
    X = df_filtered[FEATURE_COLS].values.astype(float)
    X_scaled = scaler.transform(X)

    # Fit KNN on cluster
    n_neighbors = min(topk, len(df_filtered))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X_scaled)

    # Create a "query point" representing the target HR
    # Heuristic: set tempo=target_bpm, use median values for other features
    query_point = np.array([
        target_bpm,
        df_filtered["energy"].median(),
        df_filtered["danceability"].median(),
        df_filtered["valence"].median(),
        df_filtered["loudness"].median()
    ]).reshape(1, -1)
    query_point_scaled = scaler.transform(query_point)

    # Query KNN
    distances, indices = knn.kneighbors(query_point_scaled)
    distances = distances[0]  # shape (n_neighbors,)
    indices = indices[0]      # shape (n_neighbors,)

    # Build result dataframe
    result_rows = []
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        row = df_filtered.iloc[idx]
        result_rows.append({
            "track_id": row.get("track_id"),
            "name": row.get("name"),
            "artists": row.get("artists"),
            "cluster": row.get("cluster"),
            "tempo": float(row["tempo"]),
            "energy": float(row["energy"]),
            "danceability": float(row["danceability"]),
            "valence": float(row["valence"]),
            "loudness": float(row["loudness"]),
            "distance": float(dist),
            "rank": rank,
        })

    result = pd.DataFrame(result_rows)
    return result


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KMeans + KNN song matcher: cluster songs, then find best matches by heart rate."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="audio_features_sample.csv",
        help="Path to the features CSV file",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of KMeans clusters to create",
    )
    parser.add_argument(
        "--hr",
        type=float,
        help="Target heart rate in BPM (for KNN matching)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="How many songs to return (KNN neighbors)",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="If set, restrict KNN search to this cluster ID",
    )
    args, _ = parser.parse_known_args()

    # 1) Load features
    df = load_features(args.csv_path)
    log.info(f"Loaded {len(df)} rows from {args.csv_path}")

    # 2) Run KMeans
    df_clustered, km, scaler = run_kmeans(df, n_clusters=args.clusters)
    log.info(f"Assigned clusters 0..{args.clusters - 1}")

    # Print cluster summary
    cluster_counts = df_clustered["cluster"].value_counts().sort_index()
    log.info("Cluster sizes:")
    for cid, count in cluster_counts.items():
        mean_tempo = df_clustered[df_clustered["cluster"] == cid]["tempo"].mean()
        log.info(f"  cluster {cid}: {count} tracks (mean tempo: {mean_tempo:.1f} BPM)")

    # 3) Write clustered features CSV
    out_path = "clustered_features.csv"
    df_clustered.to_csv(out_path, index=False)
    log.info(f"Wrote clustered features → {out_path}")

    # 4) If HR provided, use KNN to find best songs
    target_hr = args.hr
    if target_hr is None:
        try:
            s = input("Enter heart rate BPM (or press Enter to skip): ").strip()
            target_hr = float(s) if s else None
        except Exception:
            target_hr = None

    if target_hr is not None:
        log.info(f"\n{'='*60}")
        log.info(f"Finding songs for target HR: {target_hr} BPM")
        log.info(f"{'='*60}")

        # Pick best cluster if not specified
        chosen_cluster = args.cluster
        if chosen_cluster is None:
            chosen_cluster, mean_tempo, tempo_delta = pick_cluster_by_tempo(df_clustered, target_hr)
            log.info(
                f"Auto-selected cluster {chosen_cluster} "
                f"(mean tempo: {mean_tempo:.1f} BPM, delta: {tempo_delta:.1f} BPM)"
            )
        else:
            log.info(f"Using specified cluster {chosen_cluster}")

        # Run KNN on the cluster
        picks = knn_in_cluster(
            df_clustered,
            scaler=scaler,
            target_bpm=target_hr,
            topk=args.k,
            cluster_id=chosen_cluster,
        )

        if picks.empty:
            log.warning("No candidates found for the given heart rate/cluster.")
        else:
            # Write output CSV
            picks_out = "knn_picks.csv"
            picks.to_csv(picks_out, index=False)
            log.info(f"Top {len(picks)} KNN matches → {picks_out}")

            # Pretty print
            cols_print = ["rank", "name", "artists", "tempo", "energy", "danceability", "distance"]
            print("\n" + "="*80)
            print(f"Top {len(picks)} songs (cluster {chosen_cluster}, HR={target_hr} BPM):")
            print("="*80)
            print(picks[cols_print].to_string(index=False, justify="left"))
            print("="*80 + "\n")


if __name__ == "__main__":
    main()
