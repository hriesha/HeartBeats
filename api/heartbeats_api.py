#!/usr/bin/env python3
"""
HeartBeats API Backend
Flask/FastAPI endpoint to handle:
1. KMeans clustering of user's library
2. KNN matching based on BPM
3. Spotify track retrieval
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import numpy/pandas with better error handling
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
except ImportError as e:
    print("=" * 60)
    print("ERROR: Failed to import required packages")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nThis is likely due to a numpy architecture mismatch.")
    print("Please run the fix script:")
    print("  ./fix_numpy.sh")
    print("\nOr manually fix with:")
    print("  pip3 install --upgrade --force-reinstall numpy pandas scikit-learn")
    print("=" * 60)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger("heartbeats_api")

# Add project root to path for spotify_api_integration (optional)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from spotify_api_integration import SpotifyIntegration
    SPOTIFY_AVAILABLE = True
except ImportError:
    log.warning("Spotify integration not available (python-dotenv/spotipy not installed)")
    SPOTIFY_AVAILABLE = False
    SpotifyIntegration = None

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]

# Global state for clustering (cache to avoid recomputing)
_clustered_df: Optional[pd.DataFrame] = None
_kmeans_model: Optional[KMeans] = None
_scaler: Optional[StandardScaler] = None
_spotify: Optional[SpotifyIntegration] = None


def load_features(csv_path: str = "basic-api-demo/audio_features_sample.csv") -> pd.DataFrame:
    """Load features CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Features CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def run_kmeans(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> tuple:
    """Run KMeans clustering."""
    X = df[FEATURE_COLS].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = km.fit_predict(X_scaled)
    df_clustered = df.copy()
    df_clustered["cluster"] = clusters
    return df_clustered, km, scaler


def pick_cluster_by_tempo(df: pd.DataFrame, target_bpm: float) -> tuple:
    """Pick cluster with mean tempo closest to target BPM."""
    cluster_stats = (
        df.groupby("cluster")["tempo"]
        .mean()
        .reset_index()
        .rename(columns={"tempo": "mean_tempo"})
    )
    cluster_stats["tempo_delta"] = (cluster_stats["mean_tempo"] - target_bpm).abs()
    row = cluster_stats.sort_values("tempo_delta").iloc[0]
    return int(row["cluster"]), float(row["mean_tempo"]), float(row["tempo_delta"])


def knn_in_cluster(
    df: pd.DataFrame,
    scaler: StandardScaler,
    target_bpm: float,
    topk: int = 10,
    cluster_id: Optional[int] = None,
) -> pd.DataFrame:
    """Use KNN to find top-K songs closest to target BPM in feature space."""
    if cluster_id is not None:
        df_filtered = df[df["cluster"] == cluster_id].copy()
        if df_filtered.empty:
            return pd.DataFrame()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        return pd.DataFrame()

    X = df_filtered[FEATURE_COLS].values.astype(float)
    X_scaled = scaler.transform(X)

    n_neighbors = min(topk, len(df_filtered))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X_scaled)

    query_point = np.array([
        target_bpm,
        df_filtered["energy"].median(),
        df_filtered["danceability"].median(),
        df_filtered["valence"].median(),
        df_filtered["loudness"].median()
    ]).reshape(1, -1)
    query_point_scaled = scaler.transform(query_point)

    distances, indices = knn.kneighbors(query_point_scaled)
    distances = distances[0]
    indices = indices[0]

    result_rows = []
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        row = df_filtered.iloc[idx]
        result_rows.append({
            "track_id": row.get("track_id"),
            "name": row.get("name"),
            "artists": row.get("artists"),
            "cluster": int(row.get("cluster")),
            "tempo": float(row["tempo"]),
            "energy": float(row["energy"]),
            "danceability": float(row["danceability"]),
            "valence": float(row["valence"]),
            "loudness": float(row["loudness"]),
            "distance": float(dist),
            "rank": rank,
        })

    return pd.DataFrame(result_rows)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/api/clusters', methods=['POST'])
def get_clusters():
    """
    Run KMeans clustering on user's library.
    Returns cluster information.
    """
    global _clustered_df, _kmeans_model, _scaler

    try:
        data = request.get_json() or {}
        csv_path = data.get("csv_path", "basic-api-demo/audio_features_sample.csv")
        n_clusters = data.get("n_clusters", 4)

        # Load and cluster
        df = load_features(csv_path)
        df_clustered, km, scaler = run_kmeans(df, n_clusters=n_clusters)

        # Cache for later use
        _clustered_df = df_clustered
        _kmeans_model = km
        _scaler = scaler

        # Get cluster stats
        cluster_stats = []
        for cluster_id in sorted(df_clustered["cluster"].unique()):
            cluster_data = df_clustered[df_clustered["cluster"] == cluster_id]
            cluster_stats.append({
                "cluster_id": int(cluster_id),
                "count": len(cluster_data),
                "mean_tempo": float(cluster_data["tempo"].mean()),
                "mean_energy": float(cluster_data["energy"].mean()),
                "mean_danceability": float(cluster_data["danceability"].mean()),
            })

        return jsonify({
            "success": True,
            "clusters": cluster_stats,
            "total_tracks": len(df_clustered)
        })

    except Exception as e:
        log.error(f"Error in get_clusters: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/tracks', methods=['POST'])
def get_tracks():
    """
    Get tracks for a given BPM and cluster using KNN.
    Returns list of track IDs and metadata.
    """
    global _clustered_df, _scaler

    try:
        data = request.get_json() or {}
        target_bpm = data.get("bpm")
        cluster_id = data.get("cluster_id")  # Optional
        topk = data.get("topk", 10)

        if target_bpm is None:
            return jsonify({"success": False, "error": "BPM is required"}), 400

        if _clustered_df is None or _scaler is None:
            # Run clustering first if not done
            df = load_features()
            _clustered_df, _, _scaler = run_kmeans(df, n_clusters=4)

        # Pick cluster if not specified
        if cluster_id is None:
            cluster_id, mean_tempo, tempo_delta = pick_cluster_by_tempo(_clustered_df, target_bpm)

        # Run KNN
        picks = knn_in_cluster(
            _clustered_df,
            _scaler,
            target_bpm=target_bpm,
            topk=topk,
            cluster_id=cluster_id,
        )

        if picks.empty:
            return jsonify({"success": False, "error": "No tracks found"}), 404

        # Convert to list of dicts
        tracks = picks.to_dict("records")

        return jsonify({
            "success": True,
            "cluster_id": cluster_id,
            "tracks": tracks,
            "count": len(tracks)
        })

    except Exception as e:
        log.error(f"Error in get_tracks: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/tracks/details', methods=['POST'])
def get_track_details():
    """
    Get full track details from Spotify API for a list of track IDs.
    """
    global _spotify

    try:
        data = request.get_json() or {}
        track_ids = data.get("track_ids", [])

        if not track_ids:
            return jsonify({"success": False, "error": "track_ids is required"}), 400

        # Initialize Spotify if needed
        if _spotify is None:
            _spotify = SpotifyIntegration()
            if not _spotify.connect():
                return jsonify({"success": False, "error": "Failed to connect to Spotify"}), 500

        # Get track details
        tracks = _spotify.get_tracks(track_ids)
        formatted_tracks = [_spotify.format_track_for_display(t) for t in tracks if t]

        return jsonify({
            "success": True,
            "tracks": formatted_tracks,
            "count": len(formatted_tracks)
        })

    except Exception as e:
        log.error(f"Error in get_track_details: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/cluster/tracks', methods=['POST'])
def get_cluster_tracks():
    """
    Combined endpoint: Get tracks for BPM + cluster, then fetch full details.
    """
    global _clustered_df, _scaler, _spotify

    try:
        data = request.get_json() or {}
        target_bpm = data.get("bpm")
        cluster_id = data.get("cluster_id")
        topk = data.get("topk", 10)

        if target_bpm is None:
            return jsonify({"success": False, "error": "BPM is required"}), 400

        if _clustered_df is None or _scaler is None:
            df = load_features()
            _clustered_df, _, _scaler = run_kmeans(df, n_clusters=4)

        # Pick cluster if not specified
        if cluster_id is None:
            cluster_id, mean_tempo, tempo_delta = pick_cluster_by_tempo(_clustered_df, target_bpm)

        # Run KNN to get track IDs
        picks = knn_in_cluster(
            _clustered_df,
            _scaler,
            target_bpm=target_bpm,
            topk=topk,
            cluster_id=cluster_id,
        )

        if picks.empty:
            return jsonify({"success": False, "error": "No tracks found"}), 404

        track_ids = picks["track_id"].tolist()

        # Get full track details from Spotify
        if _spotify is None:
            try:
                _spotify = SpotifyIntegration()
                _spotify.connect()
            except Exception as e:
                log.warning(f"Spotify connection failed: {e}, returning basic track info")

        if _spotify and _spotify.is_connected():
            tracks_full = _spotify.get_tracks(track_ids)
            formatted_tracks = [_spotify.format_track_for_display(t) for t in tracks_full if t]
            
            # Merge with KNN metadata
            track_map = {t["track_id"]: t for t in picks.to_dict("records")}
            for track_detail in formatted_tracks:
                track_id = track_detail.get("id")
                if track_id in track_map:
                    track_detail.update(track_map[track_id])
            
            return jsonify({
                "success": True,
                "cluster_id": cluster_id,
                "tracks": formatted_tracks,
                "count": len(formatted_tracks)
            })
        else:
            # Return basic info if Spotify fails
            tracks = picks.to_dict("records")
            return jsonify({
                "success": True,
                "cluster_id": cluster_id,
                "tracks": tracks,
                "count": len(tracks)
            })

    except Exception as e:
        log.error(f"Error in get_cluster_tracks: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": f"Route not found: {request.path}",
        "available_routes": [
            "/api/health",
            "/api/clusters",
            "/api/tracks",
            "/api/tracks/details",
            "/api/cluster/tracks"
        ]
    }), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # Changed from 5000 to 5001 (5000 is used by AirPlay)
    print(f"\n🚀 HeartBeats API Server")
    print(f"📍 Running on http://0.0.0.0:{port}")
    print(f"🔗 Health check: http://localhost:{port}/api/health")
    print(f"\nAvailable endpoints:")
    print(f"  POST /api/clusters")
    print(f"  POST /api/tracks")
    print(f"  POST /api/tracks/details")
    print(f"  POST /api/cluster/tracks")
    print(f"\n⚠️  Note: Frontend should be accessed via Vite dev server (usually http://localhost:5173)\n")
    app.run(host="0.0.0.0", port=port, debug=True)
