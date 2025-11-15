#!/usr/bin/env python3
# clustering-match.py
#
# Offline version:
#  - Loads audio_features_sample.csv (synthetic or precomputed features)
#  - Runs K-Means on [tempo, energy, danceability, valence, loudness]
#  - Writes clustered_features.csv
#  - Optionally, given a heart rate, picks closest songs by tempo
#    (optionally restricted to a specific cluster)

import logging
import argparse
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------
# Setup
# ---------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("heartbeats-offline")

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


def run_kmeans(df: pd.DataFrame, n_clusters: int, random_state: int = 42) -> Tuple[pd.DataFrame, KMeans]:
    """
    Run KMeans on the 5D feature vector and return:
      - df with a new 'cluster' column
      - the fitted KMeans object
    """
    X = df[FEATURE_COLS].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = km.fit_predict(X_scaled)

    df = df.copy()
    df["cluster"] = clusters
    return df, km


def best_tempo_match(song_bpm: float, target: float, consider_multiples: bool = True) -> Tuple[float, float, float]:
    """
    Return (delta, matched_tempo, multiplier) where matched_tempo is the
    candidate among {bpm, bpm/2, bpm*2} that’s closest to target.
    """
    candidates = [(song_bpm, 1.0)]
    if consider_multiples:
        candidates += [(song_bpm / 2.0, 0.5), (song_bpm * 2.0, 2.0)]

    # ignore absurd values
    filtered = [(b, m) for (b, m) in candidates if 30 <= b <= 240]
    if not filtered:
        filtered = [(song_bpm, 1.0)]

    deltas = [abs(target - b) for (b, _m) in filtered]
    i = min(range(len(filtered)), key=lambda j: deltas[j])
    return deltas[i], filtered[i][0], filtered[i][1]


def closest_songs(
    df: pd.DataFrame,
    target_bpm: float,
    topk: int = 10,
    consider_multiples: bool = True,
    cluster: Optional[int] = None,
) -> pd.DataFrame:
    """
    df columns expected: track_id, name, artists, tempo, (optional) cluster
    Returns a sorted DF of the topk closest to target_bpm (ascending delta).

    If cluster is not None, restrict to that cluster.
    """
    if "tempo" not in df.columns:
        raise ValueError("DataFrame must have 'tempo' column")

    if cluster is not None:
        if "cluster" not in df.columns:
            raise ValueError("DataFrame has no 'cluster' column but cluster was specified")
        df = df[df["cluster"] == cluster]

    if df.empty:
        return df

    rows = []
    for _, r in df.iterrows():
        tempo = r.get("tempo")
        if pd.isna(tempo):
            continue
        delta, matched, mult = best_tempo_match(float(tempo), float(target_bpm), consider_multiples)
        rows.append({
            "track_id": r.get("track_id"),
            "name": r.get("name"),
            "artists": r.get("artists"),
            "cluster": r.get("cluster") if "cluster" in r else None,
            "tempo": float(tempo),
            "matched_tempo": matched,
            "multiplier": mult,
            "delta": delta,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["delta"]).head(topk).reset_index(drop=True)
    return out


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
    return int(row["cluster"])


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
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
        help="Number of KMeans clusters (vibes) to create",
    )
    parser.add_argument(
        "--hr",
        type=float,
        help="Target heart rate in BPM (for picking closest songs)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="How many songs to return for the given HR",
    )
    parser.add_argument(
        "--no-multiples",
        action="store_true",
        help="Disable half/double-time matching for tempo",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="If set, restrict HR-based picks to this cluster ID",
    )
    args, _ = parser.parse_known_args()

    # 1) Load features
    df = load_features(args.csv_path)
    log.info(f"Loaded {len(df)} rows from {args.csv_path}")

    # 2) Run KMeans clustering
    df_clustered, km = run_kmeans(df, n_clusters=args.clusters)
    log.info(f"Assigned clusters 0..{args.clusters - 1}")

    # Print quick cluster summary
    cluster_counts = df_clustered["cluster"].value_counts().sort_index()
    log.info("Cluster sizes:")
    for cid, count in cluster_counts.items():
        log.info(f"  cluster {cid}: {count} tracks")

    # 3) Write clustered features CSV
    out_path = "clustered_features.csv"
    df_clustered.to_csv(out_path, index=False)
    log.info(f"Wrote clustered features → {out_path}")

    # 4) If HR provided, pick closest songs
    target_hr = args.hr
    if target_hr is not None:
        chosen_cluster = args.cluster
        if chosen_cluster is None:
            # auto-pick cluster by mean tempo
            chosen_cluster = pick_cluster_by_tempo(df_clustered, target_hr)
            log.info(f"No cluster specified; auto-selected cluster {chosen_cluster} by tempo")

        picks = closest_songs(
            df_clustered,
            target_bpm=target_hr,
            topk=args.k,
            consider_multiples=not args.no_multiples,
            cluster=chosen_cluster,
        )
        if picks.empty:
            log.warning("No candidates found for the given heart rate/cluster.")
        else:
            picks_out = "closest_songs.csv"
            picks.to_csv(picks_out, index=False)
            log.info(f"Top {len(picks)} closest → {picks_out}")
            cols_print = ["name", "artists", "cluster", "tempo", "matched_tempo", "multiplier", "delta"]
            print("\nClosest songs:")
            print(picks[cols_print].to_string(index=False, justify="left"))


if __name__ == "__main__":
    main()