"""
Flask API server for HeartBeats algorithm integration

This server handles:
1. Running KMeans clustering on user's Spotify library
2. KNN matching to find tracks for a given cluster and BPM
3. Integration with spotify_api_integration.py for track retrieval
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import sys

# Setup logging first
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger("heartbeats_api")

# Import our Spotify integration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spotify_api_integration import SpotifyIntegration
try:
    from annas_archive_helper import get_audio_features, is_archive_available
    ANNAS_ARCHIVE_AVAILABLE = True
except ImportError:
    log.warning("annas_archive_helper not available, will use Spotify API only")
    ANNAS_ARCHIVE_AVAILABLE = False
    def get_audio_features(track_ids): return pd.DataFrame()
    def is_archive_available(): return False

# Import improved algorithm
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'algorithms'))
    from improved_kmeans_knn import (
        run_improved_kmeans,
        improved_knn_in_cluster,
        select_best_cluster,
        CLUSTERING_FEATURES,
        KNN_FEATURES
    )
    USE_IMPROVED_ALGORITHM = True
except ImportError:
    log.warning("Improved algorithm not available, using basic version")
    USE_IMPROVED_ALGORITHM = False

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables to cache results
spotify_client: Optional[SpotifyIntegration] = None
cached_clustered_df: Optional[pd.DataFrame] = None
cached_scaler: Optional[StandardScaler] = None
cached_bpm: Optional[float] = None

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]


def initialize_spotify():
    """Initialize Spotify client if not already done."""
    global spotify_client
    if spotify_client is None:
        spotify_client = SpotifyIntegration()
        if not spotify_client.connect():
            raise RuntimeError("Failed to connect to Spotify")
    return spotify_client


def get_user_tracks_features() -> pd.DataFrame:
    """
    Get user's saved tracks and their audio features.
    Uses Anna's Archive if available, otherwise falls back to Spotify API.
    Returns DataFrame with track_id, name, artists, and audio features.
    """
    sp = initialize_spotify()

    # Get saved track IDs
    track_ids = []
    track_metadata = {}  # Store name, artists for later
    offset = 0
    limit = 50

    log.info("Fetching user's saved tracks...")
    while True:
        try:
            results = sp.sp.current_user_saved_tracks(limit=limit, offset=offset) if sp.sp else None
            if not results:
                break
            items = results.get("items", [])
            if not items:
                break

            for item in items:
                track = item.get("track", {})
                if track and track.get("type") == "track" and not track.get("is_local"):
                    track_id = track.get("id")
                    if track_id:
                        track_ids.append(track_id)
                        # Store metadata
                        track_metadata[track_id] = {
                            "name": track.get("name", ""),
                            "artists": ", ".join([a.get("name", "") for a in track.get("artists", [])])
                        }

            if len(items) < limit:
                break
            offset += limit
        except Exception as e:
            log.error("Error fetching saved tracks: %r", e)
            break

    log.info("Found %d saved tracks", len(track_ids))

    if not track_ids:
        return pd.DataFrame()

    # Try Anna's Archive first if available
    if ANNAS_ARCHIVE_AVAILABLE and is_archive_available():
        log.info("Attempting to load audio features from Anna's Archive...")
        features_df = get_audio_features(track_ids)

        if not features_df.empty:
            # Merge with metadata
            for idx, row in features_df.iterrows():
                track_id = row.get("track_id")
                if track_id in track_metadata:
                    features_df.at[idx, "name"] = track_metadata[track_id]["name"]
                    features_df.at[idx, "artists"] = track_metadata[track_id]["artists"]

            log.info("Successfully loaded %d tracks from Anna's Archive", len(features_df))

            # Fill missing metadata if needed
            if "name" not in features_df.columns:
                features_df["name"] = features_df["track_id"].map(lambda tid: track_metadata.get(tid, {}).get("name", ""))
            if "artists" not in features_df.columns:
                features_df["artists"] = features_df["track_id"].map(lambda tid: track_metadata.get(tid, {}).get("artists", ""))

            return features_df
        else:
            log.info("Anna's Archive data not available or incomplete, falling back to Spotify API")

    # Fallback to Spotify API
    log.info("Fetching audio features from Spotify API...")
    features_list = []
    batch_size = 100  # Spotify allows up to 100 at a time

    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        try:
            if not sp.sp:
                log.error("Spotify client not initialized")
                break
            audio_features = sp.sp.audio_features(batch)
            tracks_info = sp.sp.tracks(batch)

            if not audio_features:
                log.warning("No audio features returned for batch starting at index %d", i)
                continue

            tracks_list = tracks_info.get("tracks", []) if tracks_info else []

            for feat, track_info in zip(audio_features, tracks_list):
                if feat and feat.get("tempo") and feat.get("tempo") > 0:
                    track_id = feat.get("id")
                    track_name = track_info.get("name", "") if track_info else track_metadata.get(track_id, {}).get("name", "")
                    artists = track_info.get("artists", []) if track_info else []
                    artist_names = ", ".join([a.get("name", "") for a in artists]) if artists else track_metadata.get(track_id, {}).get("artists", "")

                    # Include all available audio features
                    features_list.append({
                        "track_id": track_id,
                        "name": track_name,
                        "artists": artist_names,
                        "tempo": feat.get("tempo", 0),
                        "energy": feat.get("energy", 0),
                        "danceability": feat.get("danceability", 0),
                        "valence": feat.get("valence", 0),
                        "loudness": feat.get("loudness", 0),
                        "acousticness": feat.get("acousticness", 0),
                        "instrumentalness": feat.get("instrumentalness", 0),
                        "speechiness": feat.get("speechiness", 0),
                        "liveness": feat.get("liveness", 0),
                        "key": feat.get("key", 0),
                        "mode": feat.get("mode", 0),
                        "time_signature": feat.get("time_signature", 4),
                        "duration_ms": feat.get("duration_ms", 0),
                    })
        except Exception as e:
            log.error("Error fetching audio features batch starting at index %d: %r", i, e)
            import traceback
            log.error(traceback.format_exc())

    df = pd.DataFrame(features_list)
    log.info("Retrieved audio features for %d tracks (out of %d track IDs)", len(df), len(track_ids))

    if df.empty:
        log.warning("No tracks with valid audio features found!")
        log.warning("This could mean:")
        log.warning("  1. Tracks don't have tempo data")
        log.warning("  2. API rate limiting")
        log.warning("  3. Tracks are local files or unavailable")
        log.warning("  4. Consider using Anna's Archive data - see annas_archive_helper.py")

    return df


def run_clustering(
    df: pd.DataFrame,
    n_clusters: int = 4,
    bpm: Optional[float] = None,
    auto_k: bool = False,
    bpm_filter_first: bool = True,
    bpm_tolerance: float = 30.0
) -> Dict[str, Any]:
    """
    Run KMeans clustering on tracks and return cluster information.
    Uses improved algorithm if available, falls back to basic version.

    Returns:
        {
            "clusters": [
                {
                    "id": 0,
                    "name": "Chill Flow",
                    "color": "#EAE2B7",
                    "tags": ["lo-fi", "calm"],
                    "mean_tempo": 90.5,
                    "track_count": 25
                },
                ...
            ],
            "df": DataFrame with cluster assignments (for later KNN)
        }
    """
    global cached_clustered_df, cached_scaler

    if df.empty:
        return {"clusters": [], "df": df}

    # Use improved algorithm if available
    if USE_IMPROVED_ALGORITHM:
        try:
            log.info("Using improved KMeans algorithm")
            # Pass BPM to make clustering BPM-aware
            df_clustered, km, scaler, metadata = run_improved_kmeans(
                df,
                target_bpm=bpm,
                n_clusters=n_clusters,
                auto_k=auto_k,
                bpm_filter_first=bpm_filter_first,
                bpm_tolerance=bpm_tolerance
            )

            # Cache for KNN queries
            cached_clustered_df = df_clustered
            cached_scaler = scaler
            # Store which features were used for clustering and BPM
            cached_clustered_df._clustering_features = metadata["features_used"]
            cached_clustered_df._target_bpm = bpm

            # Use dynamic cluster names from metadata (already generated by improved algorithm)
            cluster_info = []
            for c in metadata["cluster_info"]:
                cluster_info.append({
                    "id": c["id"],
                    "name": c.get("name", f"Cluster {c['id']}"),
                    "color": c.get("color", "#EAE2B7"),
                    "tags": c.get("tags", []),
                    "mean_tempo": c["mean_tempo"],
                    "track_count": c["size"]
                })

            return {
                "clusters": cluster_info,
                "df": df_clustered.to_dict(orient="records")
            }
        except Exception as e:
            log.warning("Improved algorithm failed, falling back to basic: %r", e)
            # Fall through to basic algorithm

    # Basic algorithm (fallback)
    log.info("Using basic KMeans algorithm")
    X = df[FEATURE_COLS].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = km.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster"] = clusters

    # Cache for KNN queries
    cached_clustered_df = df_clustered
    cached_scaler = scaler

    # Calculate cluster stats
    cluster_info = []
    cluster_colors = ["#EAE2B7", "#FCBF49", "#F77F00", "#D62828"]
    cluster_names = ["Chill Flow", "Focus Pulse", "Energy Rush", "Intense Beats"]
    cluster_tags = [
        ["lo-fi", "calm"],
        ["deep work", "concentration"],
        ["upbeat", "motivating"],
        ["powerful", "high-energy"]
    ]

    for cid in range(n_clusters):
        cluster_tracks = df_clustered[df_clustered["cluster"] == cid]
        if len(cluster_tracks) > 0:
            cluster_info.append({
                "id": cid,
                "name": cluster_names[cid] if cid < len(cluster_names) else f"Cluster {cid}",
                "color": cluster_colors[cid] if cid < len(cluster_colors) else "#EAE2B7",
                "tags": cluster_tags[cid] if cid < len(cluster_tags) else [],
                "mean_tempo": float(cluster_tracks["tempo"].mean()),
                "track_count": int(len(cluster_tracks))
            })

    return {
        "clusters": cluster_info,
        "df": df_clustered.to_dict(orient="records")
    }


def get_knn_tracks(cluster_id: int, target_bpm: float, topk: int = 10) -> List[Dict[str, Any]]:
    """
    Use KNN to find top-K tracks in the specified cluster matching the target BPM.
    Uses improved algorithm if available.

    Returns list of track dictionaries with track_id, name, artists, tempo, etc.
    """
    global cached_clustered_df, cached_scaler

    if cached_clustered_df is None or cached_scaler is None:
        raise ValueError("Clustering not performed yet. Call /cluster first.")

    # Use improved KNN if available
    if USE_IMPROVED_ALGORITHM:
        try:
            log.info("Using improved KNN algorithm")
            # Get the features that were used for clustering
            clustering_features_used = getattr(cached_clustered_df, '_clustering_features', None)
            if clustering_features_used is None:
                # Try to infer from available features
                clustering_features_used = [f for f in CLUSTERING_FEATURES if f in cached_clustered_df.columns]

            # Use the same features that were used for clustering
            matches_df = improved_knn_in_cluster(
                cached_clustered_df,
                cached_scaler,
                target_bpm,
                cluster_id,
                topk=topk,
                use_weights=True,
                clustering_features=clustering_features_used
            )

            if matches_df.empty:
                log.warning("Improved KNN returned empty results")
                # Fall through to basic algorithm
                raise ValueError("Empty results from improved KNN")

            # Convert to list of dicts
            results = []
            for _, row in matches_df.iterrows():
                results.append({
                    "track_id": row.get("track_id", ""),
                    "name": row.get("name", ""),
                    "artists": row.get("artists", ""),
                    "cluster": int(row.get("cluster", cluster_id)),
                    "tempo": float(row.get("tempo", 0)),
                    "energy": float(row.get("energy", 0)),
                    "danceability": float(row.get("danceability", 0)),
                    "valence": float(row.get("valence", 0)),
                    "loudness": float(row.get("loudness", 0)),
                    "distance": float(row.get("distance", 0)),
                    "rank": int(row.get("rank", 0)),
                })
            return results
        except Exception as e:
            log.warning("Improved KNN failed, falling back to basic: %r", e)
            import traceback
            log.error(traceback.format_exc())
            # Fall through to basic algorithm

    # Basic algorithm (fallback)
    log.info("Using basic KNN algorithm")
    df_cluster = cached_clustered_df[cached_clustered_df["cluster"] == cluster_id].copy()

    if df_cluster.empty:
        return []

    # Extract and scale features
    X = df_cluster[FEATURE_COLS].values.astype(float)
    X_scaled = cached_scaler.transform(X)

    # Fit KNN on cluster
    n_neighbors = min(topk, len(df_cluster))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X_scaled)

    # Create query point (target BPM with median other features)
    query_point = np.array([
        target_bpm,
        df_cluster["energy"].median(),
        df_cluster["danceability"].median(),
        df_cluster["valence"].median(),
        df_cluster["loudness"].median()
    ]).reshape(1, -1)
    query_point_scaled = cached_scaler.transform(query_point)

    # Query KNN
    distances, indices = knn.kneighbors(query_point_scaled)
    distances = distances[0]
    indices = indices[0]

    # Build result list
    results = []
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        row = df_cluster.iloc[idx]
        results.append({
            "track_id": row.get("track_id"),
            "name": row.get("name"),
            "artists": row.get("artists"),
            "cluster": int(row.get("cluster")),
            "tempo": float(row.get("tempo")),
            "energy": float(row.get("energy")),
            "danceability": float(row.get("danceability")),
            "valence": float(row.get("valence")),
            "loudness": float(row.get("loudness")),
            "distance": float(dist),
            "rank": rank
        })

    return results


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/api/spotify/connect', methods=['POST'])
def connect_spotify():
    """Connect to Spotify (for initial connection)."""
    try:
        sp = initialize_spotify()
        user_info = sp.get_user_info()
        return jsonify({
            "success": True,
            "user": {
                "name": user_info.get("display_name"),
                "id": user_info.get("id")
            }
        })
    except Exception as e:
        log.error("Spotify connection error: %r", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/cluster', methods=['POST'])
def cluster_tracks():
    """
    Run KMeans clustering on user's tracks.

    Request body:
        {
            "bpm": 120.0,  # Target BPM
            "n_clusters": 4  # Optional, defaults to 4
        }

    Returns:
        {
            "clusters": [...],  # Cluster info
            "df": [...]  # Track data with cluster assignments
        }
    """
    global cached_bpm, cached_clustered_df, cached_scaler

    try:
        data = request.get_json()
        bpm = data.get("bpm")
        n_clusters = data.get("n_clusters")  # None = auto-determine

        if bpm is None:
            return jsonify({"error": "BPM is required"}), 400

        new_bpm = float(bpm)

        # CRITICAL: Always invalidate cache when BPM changes
        # Even small BPM changes should trigger re-clustering
        if cached_bpm is not None:
            if abs(cached_bpm - new_bpm) > 0.5:  # Any significant BPM change
                log.info("BPM changed from %s to %s - invalidating cache to re-cluster", cached_bpm, new_bpm)
                cached_clustered_df = None
                cached_scaler = None
        else:
            # First time clustering
            log.info("First clustering request for BPM: %s", new_bpm)

        cached_bpm = new_bpm

        log.info("Running BPM-aware clustering for BPM: %s (auto clusters: %s)", new_bpm, n_clusters is None)

        # Get user's tracks and features
        df = get_user_tracks_features()

        if df.empty:
            return jsonify({"error": "No tracks found. Please save some tracks to your Spotify library first."}), 400

        # Run clustering with BPM filtering first (KNN -> KMeans -> KNN flow)
        result = run_clustering(
            df,
            n_clusters=n_clusters,
            bpm=new_bpm,
            auto_k=(n_clusters is None),
            bpm_filter_first=True,  # Filter by BPM before clustering
            bpm_tolerance=30.0  # ±30 BPM range
        )

        log.info("Clustering complete: %d clusters generated for BPM %s", len(result.get("clusters", [])), new_bpm)

        return jsonify(result)

    except Exception as e:
        log.error("Clustering error: %r", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/cluster/<int:cluster_id>/tracks', methods=['GET'])
def get_cluster_tracks(cluster_id: int):
    """
    Get KNN-matched tracks for a specific cluster.

    Query params:
        bpm: Target BPM (uses cached BPM if not provided)
        topk: Number of tracks to return (default 10)

    Returns:
        {
            "tracks": [...],  # List of track dictionaries
            "cluster_id": 0,
            "bpm": 120.0
        }
    """
    try:
        bpm = request.args.get("bpm", type=float) or cached_bpm
        topk = request.args.get("topk", type=int, default=10)

        if bpm is None:
            return jsonify({"error": "BPM is required"}), 400

        log.info("Getting KNN tracks for cluster %d, BPM: %s, topk: %d", cluster_id, bpm, topk)

        tracks = get_knn_tracks(cluster_id, float(bpm), topk=topk)

        return jsonify({
            "tracks": tracks,
            "cluster_id": cluster_id,
            "bpm": float(bpm)
        })

    except Exception as e:
        log.error("KNN error: %r", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/tracks/details', methods=['POST'])
def get_track_details():
    """
    Get detailed track information from Spotify for a list of track IDs.

    Request body:
        {
            "track_ids": ["track_id1", "track_id2", ...]
        }

    Returns:
        {
            "tracks": [...]  # List of formatted track dictionaries
        }
    """
    try:
        data = request.get_json()
        track_ids = data.get("track_ids", [])

        if not track_ids:
            return jsonify({"error": "track_ids array is required"}), 400

        sp = initialize_spotify()
        tracks = sp.get_tracks(track_ids)

        # Format tracks for UI
        formatted_tracks = []
        for track in tracks:
            if track:
                formatted = sp.format_track_for_display(track)
                formatted_tracks.append(formatted)

        return jsonify({"tracks": formatted_tracks})

    except Exception as e:
        log.error("Track details error: %r", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
