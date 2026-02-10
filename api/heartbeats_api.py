#!/usr/bin/env python3
"""
HeartBeats API Backend
Flask/FastAPI endpoint to handle:
1. KMeans clustering of user's library
2. KNN matching based on BPM
3. Spotify track retrieval
"""

import os
import re
import sys
import logging
import sqlite3
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify, redirect
from flask_cors import CORS

# Load .env (Spotify credentials, etc.) if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Import numpy/pandas with better error handling
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
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
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
try:
    from spotify_api_integration import SpotifyIntegration
    SPOTIFY_AVAILABLE = True
except ImportError:
    log.warning("Spotify integration not available (python-dotenv/spotipy not installed)")
    SPOTIFY_AVAILABLE = False
    SpotifyIntegration = None

try:
    from algorithms.cluster_naming import generate_cluster_names
    CLUSTER_NAMING_AVAILABLE = True
except ImportError:
    CLUSTER_NAMING_AVAILABLE = False
    generate_cluster_names = None

# Recs module: pre-trained cluster + KNN from track_id only (no Anna's Archive at request time)
try:
    import recs
    from recs.inference import (
        get_cluster_only,
        get_cluster_and_neighbors,
        predict_cluster_from_features,
        predict_audio_features_from_metadata,
        get_cluster_for_track,
    )
    from recs.pace_to_step_bpm import pace_to_step_bpm
    RECS_AVAILABLE = True
except ImportError as e:
    log.warning("recs module not available: %s", e)
    RECS_AVAILABLE = False
    get_cluster_only = None
    get_cluster_and_neighbors = None
    predict_cluster_from_features = None
    predict_audio_features_from_metadata = None
    get_cluster_for_track = None
    pace_to_step_bpm = None

# Cache for tempo lookup from training CSV
_tempo_lookup_cache: Optional[Dict[str, float]] = None
# Cache for audio features lookup from training CSV
_audio_features_lookup_cache: Optional[Dict[str, Dict[str, float]]] = None

# Rate limit safety: Max tracks per cluster to avoid hitting Spotify API limits
# Spotify allows ~300 requests/30s, and 50 tracks per request = 15k tracks/30s theoretical max
# We use 25 as a safe, conservative limit per cluster
MAX_TRACKS_PER_CLUSTER = 25
MAX_TRACKS_FOR_KNN = 15  # For queue extension (KNN from track)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]

# Anna's Archive (local) audio features DB (gitignored, but present locally)
ANNAS_DB_PATH = os.environ.get(
    "ANNAS_ARCHIVE_DB",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "annas_archive_data", "spotify_clean_audio_features.sqlite3"),
)
ANNAS_DEFAULT_LIMIT = int(os.environ.get("ANNAS_ARCHIVE_LIMIT", "25000"))
SPOTIFY_LIBRARY_LIMIT = int(os.environ.get("SPOTIFY_LIBRARY_LIMIT", "1000"))  # Reduced default for faster processing
BPM_TOLERANCE = float(os.environ.get("BPM_TOLERANCE", "15"))  # ±15 BPM for first raw filtering
BPM_FILTER_MIN_TRACKS = int(os.environ.get("BPM_FILTER_MIN_TRACKS", "80"))

# Global state for clustering (cache to avoid recomputing)
_last_bpm: Optional[float] = None
_clustered_df: Optional[pd.DataFrame] = None
_kmeans_model: Optional[KMeans] = None
_gmm_model: Optional[Any] = None  # GaussianMixture when cluster_method == "gmm"
_scaler: Optional[StandardScaler] = None
_cluster_method: str = "kmeans"  # "kmeans" | "gmm"
_spotify: Optional[SpotifyIntegration] = None
_spotify_cc = None  # spotipy client-credentials client
_predicted_features_cache: Dict[str, dict] = {}  # track_id -> predicted audio features (for KNN)
_dataset_metadata: Dict[str, Dict[str, Any]] = {}  # track_id -> {track_name, artists, album_name} from CSV


def _load_features_from_annas_archive(limit: int = ANNAS_DEFAULT_LIMIT) -> pd.DataFrame:
    """
    Load audio features from the local Anna's Archive SQLite DB.

    Note: the DB only has audio features + track_id (no names/artists).
    We'll fetch names/artists later via /api/tracks/details using Spotify API.
    """
    if not os.path.exists(ANNAS_DB_PATH):
        raise FileNotFoundError(f"Anna's Archive DB not found: {ANNAS_DB_PATH}")

    conn = sqlite3.connect(ANNAS_DB_PATH)
    try:
        # Keep query simple and fast: take first N rows with required features present.
        query = """
            SELECT
                track_id,
                tempo,
                energy,
                danceability,
                valence,
                loudness
            FROM track_audio_features
            WHERE
                null_response = 0
                AND tempo IS NOT NULL
                AND energy IS NOT NULL
                AND danceability IS NOT NULL
                AND valence IS NOT NULL
                AND loudness IS NOT NULL
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(int(limit),))
    finally:
        conn.close()

    # Add placeholders so the rest of the pipeline can keep working.
    if "name" not in df.columns:
        df["name"] = ""
    if "artists" not in df.columns:
        df["artists"] = ""

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Anna's Archive query missing required columns: {missing}")

    return df


def _get_user_saved_track_ids(limit: int = SPOTIFY_LIBRARY_LIMIT) -> List[str]:
    """
    Fetch the current user's saved-track IDs via Spotify OAuth (user-library-read).
    """
    global _spotify

    if not SPOTIFY_AVAILABLE or SpotifyIntegration is None:
        raise RuntimeError("Spotify integration not available")

    if _spotify is None:
        _spotify = SpotifyIntegration()
        if not _spotify.connect():
            raise RuntimeError("Failed to connect to Spotify")

    if not _spotify.is_connected() or not _spotify.sp:
        raise RuntimeError("Spotify not connected")

    track_ids: List[str] = []
    offset = 0
    page_size = 50

    while len(track_ids) < limit:
        resp = _spotify.sp.current_user_saved_tracks(limit=page_size, offset=offset)
        items = (resp or {}).get("items") or []
        if not items:
            break

        for item in items:
            track = (item or {}).get("track") or {}
            # Skip local tracks (no Spotify ID)
            if track.get("is_local"):
                continue
            tid = track.get("id")
            if tid:
                track_ids.append(tid)
                if len(track_ids) >= limit:
                    break

        if len(items) < page_size:
            break
        offset += page_size

    # Deduplicate but keep order
    seen = set()
    deduped: List[str] = []
    for tid in track_ids:
        if tid not in seen:
            seen.add(tid)
            deduped.append(tid)
    return deduped


def _load_features_from_annas_archive_for_track_ids(track_ids: List[str]) -> pd.DataFrame:
    """
    Load audio features for specific Spotify track IDs from the local Anna's Archive DB.
    SQLite has a variable limit, so we query in chunks.
    """
    if not track_ids:
        return pd.DataFrame()
    if not os.path.exists(ANNAS_DB_PATH):
        raise FileNotFoundError(f"Anna's Archive DB not found: {ANNAS_DB_PATH}")

    conn = sqlite3.connect(ANNAS_DB_PATH)
    try:
        chunks = []
        # Stay well under SQLite's default variable limit (999)
        chunk_size = 900
        base_sql = """
            SELECT
                track_id,
                tempo,
                energy,
                danceability,
                valence,
                loudness
            FROM track_audio_features
            WHERE
                null_response = 0
                AND tempo IS NOT NULL
                AND energy IS NOT NULL
                AND danceability IS NOT NULL
                AND valence IS NOT NULL
                AND loudness IS NOT NULL
                AND track_id IN ({placeholders})
        """

        for i in range(0, len(track_ids), chunk_size):
            batch = track_ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(batch))
            sql = base_sql.format(placeholders=placeholders)
            chunks.append(pd.read_sql_query(sql, conn, params=batch))

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    # Add placeholders so the rest of the pipeline can keep working.
    if "name" not in df.columns:
        df["name"] = ""
    if "artists" not in df.columns:
        df["artists"] = ""

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Anna's Archive query missing required columns: {missing}")

    return df


def _filter_df_by_bpm(df: pd.DataFrame, bpm: float) -> pd.DataFrame:
    """Filter df to tracks with |tempo - bpm| <= BPM_TOLERANCE. Fall back to full df if too few."""
    if df.empty or "tempo" not in df.columns:
        return df
    tol = BPM_TOLERANCE
    filtered = df[np.abs(df["tempo"].astype(float) - float(bpm)) <= tol].copy()
    if len(filtered) >= BPM_FILTER_MIN_TRACKS:
        return filtered
    # Relax tolerance
    filtered2 = df[np.abs(df["tempo"].astype(float) - float(bpm)) <= tol * 2].copy()
    return filtered2 if len(filtered2) >= BPM_FILTER_MIN_TRACKS else df


def load_features(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load audio features.

    Priority:
    - If csv_path provided: load that CSV.
    - Else: load from local Anna's Archive DB if present.
    - Else: fall back to the sample CSV.
    """
    if csv_path:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Features CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        return df

    # Default: use Anna's Archive if available
    if os.path.exists(ANNAS_DB_PATH):
        return _load_features_from_annas_archive()

    # Final fallback: demo CSV
    demo_path = "basic-api-demo/audio_features_sample.csv"
    df = pd.read_csv(demo_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def run_kmeans(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> tuple:
    """Run KMeans clustering. Returns (df with 'cluster'), model, scaler."""
    X = df[FEATURE_COLS].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = km.fit_predict(X_scaled)
    df_clustered = df.copy()
    df_clustered["cluster"] = clusters
    return df_clustered, km, scaler


def run_gmm(
    df: pd.DataFrame,
    n_components: int = 4,
    random_state: int = 42,
) -> tuple:
    """
    Run GMM clustering. Returns (df with 'cluster', 'prob_0'..'prob_{K-1}'), model, scaler.
    'cluster' is hard assignment (argmax of probs) for compatibility with KNN/pick_cluster.
    """
    X = df[FEATURE_COLS].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        covariance_type="full",
        n_init=3,
    )
    labels = gmm.fit_predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    df_clustered = df.copy()
    df_clustered["cluster"] = labels
    for k in range(n_components):
        df_clustered[f"prob_{k}"] = probs[:, k].astype(float)
    return df_clustered, gmm, scaler


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


def _knn_on_df(
    df: pd.DataFrame,
    scaler: StandardScaler,
    target_bpm: float,
    topk: int = 10,
    cluster_id_override: Optional[int] = None,
) -> pd.DataFrame:
    """Run KNN on the given df (no cluster filtering). Returns same schema as knn_in_cluster."""
    global _dataset_metadata
    if df.empty:
        return pd.DataFrame()
    X = df[FEATURE_COLS].values.astype(float)
    X_scaled = scaler.transform(X)
    n_neighbors = min(topk, len(df))
    log.debug(f"_knn_on_df: df has {len(df)} tracks, requesting {topk} neighbors, will return {n_neighbors}")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_scaled)
    query = np.array([
        target_bpm,
        df["energy"].median(),
        df["danceability"].median(),
        df["valence"].median(),
        df["loudness"].median(),
    ]).reshape(1, -1)
    query_scaled = scaler.transform(query)
    dists, inds = nn.kneighbors(query_scaled)
    dists = dists[0]
    inds = inds[0]
    log.debug(f"_knn_on_df: KNN found {len(inds)} neighbors")
    rows = []
    for rank, (idx, d) in enumerate(zip(inds, dists), start=1):
        r = df.iloc[idx]
        tid = str(r.get("track_id"))
        c = int(cluster_id_override) if cluster_id_override is not None else int(r.get("cluster"))
        
        # Get metadata from dataset lookup or dataframe
        metadata = _dataset_metadata.get(tid, {})
        track_name = r.get("name") or metadata.get("track_name", "")
        artists = r.get("artists") or metadata.get("artists", "")
        
        out = {
            "track_id": tid,
            "name": track_name,
            "artists": artists,
            "cluster": c,
            "tempo": float(r["tempo"]),
            "energy": float(r["energy"]),
            "danceability": float(r["danceability"]),
            "valence": float(r["valence"]),
            "loudness": float(r["loudness"]),
            "distance": float(d),
            "rank": rank,
        }
        if f"prob_{c}" in r.index:
            out["cluster_prob"] = float(r[f"prob_{c}"])
        rows.append(out)
    return pd.DataFrame(rows)


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
    return _knn_on_df(df_filtered, scaler, target_bpm, topk, cluster_id_override=cluster_id)


def _knn_from_track(
    df: pd.DataFrame,
    scaler: StandardScaler,
    track_id: str,
    topk: int = 10,
    cluster_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    KNN using a track's feature vector as query (not BPM).
    Excludes the query track from results.
    """
    global _dataset_metadata
    row = df[df["track_id"].astype(str) == str(track_id)]
    if row.empty:
        return pd.DataFrame()
    row = row.iloc[0]
    if cluster_id is not None:
        df_cand = df[df["cluster"] == cluster_id].copy()
        if df_cand.empty:
            return pd.DataFrame()
    else:
        df_cand = df.copy()
    df_cand = df_cand[df_cand["track_id"].astype(str) != str(track_id)]
    if df_cand.empty:
        return pd.DataFrame()
    X = df_cand[FEATURE_COLS].values.astype(float)
    X_scaled = scaler.transform(X)
    query = np.array([
        float(row["tempo"]),
        float(row["energy"]),
        float(row["danceability"]),
        float(row["valence"]),
        float(row["loudness"]),
    ]).reshape(1, -1)
    query_scaled = scaler.transform(query)
    n = min(topk, len(df_cand))
    nn = NearestNeighbors(n_neighbors=n, metric="euclidean")
    nn.fit(X_scaled)
    dists, inds = nn.kneighbors(query_scaled)
    dists = dists[0]
    inds = inds[0]
    rows = []
    for rank, (idx, d) in enumerate(zip(inds, dists), start=1):
        r = df_cand.iloc[idx]
        tid = str(r.get("track_id"))
        c = int(cluster_id) if cluster_id is not None else int(r.get("cluster"))
        
        # Get metadata from dataset lookup or dataframe
        metadata = _dataset_metadata.get(tid, {})
        track_name = r.get("name") or metadata.get("track_name", "")
        artists = r.get("artists") or metadata.get("artists", "")
        
        out = {
            "track_id": tid,
            "name": track_name,
            "artists": artists,
            "cluster": c,
            "tempo": float(r["tempo"]),
            "energy": float(r["energy"]),
            "danceability": float(r["danceability"]),
            "valence": float(r["valence"]),
            "loudness": float(r["loudness"]),
            "distance": float(d),
            "rank": rank,
        }
        if f"prob_{c}" in r.index:
            out["cluster_prob"] = float(r[f"prob_{c}"])
        rows.append(out)
    return pd.DataFrame(rows)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = {
        "status": "ok",
        "recs_available": RECS_AVAILABLE,
        "spotify_available": SPOTIFY_AVAILABLE,
        "spotify_connected": False,
    }
    if SPOTIFY_AVAILABLE and _spotify is not None:
        try:
            status["spotify_connected"] = _spotify.is_connected() if _spotify else False
        except Exception:
            pass
    return jsonify(status)


# =============================================================================
# Spotify OAuth Endpoints
# =============================================================================

@app.route('/api/spotify/status', methods=['GET'])
def spotify_status():
    """
    Check if user is authenticated with Spotify.
    Returns user info if connected, or connected=false if not.

    This is called by the frontend on app load to check if we can skip
    the Spotify login screen (user already has a cached token).
    """
    global _spotify

    if not SPOTIFY_AVAILABLE or SpotifyIntegration is None:
        return jsonify({
            "connected": False,
            "error": "Spotify integration not available. Install spotipy and python-dotenv."
        })

    try:
        if _spotify is None:
            _spotify = SpotifyIntegration()

        # Check if we have a cached token
        token_info = _spotify.auth_manager.get_cached_token()

        if token_info:
            # Refresh token if expired
            if _spotify.auth_manager.is_token_expired(token_info):
                log.info("Cached token expired, refreshing...")
                token_info = _spotify.auth_manager.refresh_access_token(token_info['refresh_token'])

            # Initialize client if needed
            if _spotify.sp is None:
                import spotipy
                _spotify.sp = spotipy.Spotify(auth_manager=_spotify.auth_manager)

            # Verify connection by getting user info
            user = _spotify.sp.current_user()
            _spotify._current_user = user

            return jsonify({
                "connected": True,
                "user": {
                    "id": user.get("id"),
                    "display_name": user.get("display_name"),
                    "email": user.get("email"),
                    "product": user.get("product"),  # "premium" or "free"
                    "images": user.get("images", [])
                }
            })
        else:
            return jsonify({"connected": False})

    except Exception as e:
        log.warning("Spotify status check failed: %r", e)
        return jsonify({"connected": False, "error": str(e)})


@app.route('/api/spotify/auth-url', methods=['GET'])
def spotify_auth_url():
    """
    Generate the Spotify OAuth authorization URL.

    The frontend calls this, then redirects the user to the returned URL.
    The user logs into Spotify, grants permissions, then Spotify redirects
    them back to our /api/spotify/callback endpoint.
    """
    global _spotify

    if not SPOTIFY_AVAILABLE or SpotifyIntegration is None:
        return jsonify({
            "success": False,
            "error": "Spotify integration not available. Install spotipy and python-dotenv."
        }), 503

    try:
        if _spotify is None:
            _spotify = SpotifyIntegration()

        # Get the authorization URL from Spotipy's auth manager
        auth_url = _spotify.auth_manager.get_authorize_url()

        log.info("Generated Spotify auth URL: %s", auth_url[:80] + "...")
        return jsonify({
            "success": True,
            "auth_url": auth_url
        })
    except Exception as e:
        log.error("Failed to generate Spotify auth URL: %r", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/callback', methods=['GET'])
@app.route('/api/spotify/callback', methods=['GET'])
def spotify_callback():
    """
    Handle the OAuth callback from Spotify.

    After the user authorizes on Spotify's website, Spotify redirects them here
    with a ?code=xxx parameter. We exchange that code for an access token,
    then redirect the user back to the frontend with a success/error indicator.
    """
    global _spotify

    code = request.args.get('code')
    error = request.args.get('error')

    # Get frontend URL from environment or default to Vite dev server
    frontend_url = os.environ.get('FRONTEND_URL', 'http://localhost:5173')

    if error:
        log.error("Spotify OAuth error: %s", error)
        return redirect(f"{frontend_url}?spotify_error={error}")

    if not code:
        log.error("Spotify OAuth callback missing code parameter")
        return redirect(f"{frontend_url}?spotify_error=no_code")

    try:
        if _spotify is None:
            _spotify = SpotifyIntegration()

        # Exchange the authorization code for an access token
        # Spotipy handles this and caches the token automatically
        token_info = _spotify.auth_manager.get_access_token(code)

        if token_info:
            # Initialize the Spotify client with the new token
            import spotipy
            _spotify.sp = spotipy.Spotify(auth_manager=_spotify.auth_manager)
            _spotify._current_user = _spotify.sp.current_user()

            user_name = _spotify._current_user.get('display_name', 'Unknown')
            log.info("Spotify OAuth successful for user: %s", user_name)

            # Redirect back to frontend with success flag
            return redirect(f"{frontend_url}?spotify_connected=true")
        else:
            log.error("Spotify OAuth: token exchange returned None")
            return redirect(f"{frontend_url}?spotify_error=token_exchange_failed")

    except Exception as e:
        log.error("Spotify OAuth callback failed: %r", e)
        # URL-encode the error message to avoid breaking the redirect
        import urllib.parse
        error_msg = urllib.parse.quote(str(e))
        return redirect(f"{frontend_url}?spotify_error={error_msg}")


@app.route('/api/spotify/logout', methods=['POST'])
def spotify_logout():
    """
    Log out of Spotify by clearing the cached token.

    After this, the user will need to re-authorize on their next visit.
    """
    global _spotify

    try:
        # Remove the cache file that stores the OAuth token
        cache_path = ".cache-heartbeats"
        if os.path.exists(cache_path):
            os.remove(cache_path)
            log.info("Removed Spotify token cache: %s", cache_path)

        # Reset the global Spotify instance
        _spotify = None

        return jsonify({"success": True, "message": "Logged out of Spotify"})
    except Exception as e:
        log.error("Spotify logout failed: %r", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/spotify/token', methods=['GET'])
def spotify_token():
    """
    Return the current Spotify access token for the Web Playback SDK.

    The SDK needs a fresh access token to initialize the player.
    This endpoint returns the token from the server-side cache,
    refreshing it if expired.
    """
    global _spotify

    if not SPOTIFY_AVAILABLE or SpotifyIntegration is None:
        return jsonify({
            "success": False,
            "error": "Spotify integration not available"
        }), 503

    try:
        if _spotify is None:
            _spotify = SpotifyIntegration()

        token_info = _spotify.auth_manager.get_cached_token()

        if not token_info:
            return jsonify({
                "success": False,
                "error": "No cached token. Please connect Spotify first."
            }), 401

        # Refresh if expired
        if _spotify.auth_manager.is_token_expired(token_info):
            log.info("Token expired, refreshing for SDK...")
            token_info = _spotify.auth_manager.refresh_access_token(
                token_info['refresh_token']
            )

        if not token_info or 'access_token' not in token_info:
            return jsonify({
                "success": False,
                "error": "Failed to get valid token"
            }), 401

        import time as _time
        return jsonify({
            "success": True,
            "access_token": token_info['access_token'],
            "expires_in": token_info.get('expires_at', 0) - int(_time.time()),
        })
    except Exception as e:
        log.error("Token endpoint failed: %r", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/recs/coverage', methods=['GET', 'POST'])
def recs_coverage():
    """
    Test how well the recs model covers the user's Spotify library.
    Returns: total_saved, in_lookup, not_in_lookup, by_cluster counts.
    Use this to see how many of your saved tracks the model can recommend from.
    """
    if not RECS_AVAILABLE or get_cluster_only is None:
        return jsonify({
            "success": False,
            "error": "recs module not available",
        }), 503

    if not SPOTIFY_AVAILABLE or SpotifyIntegration is None:
        return jsonify({
            "success": False,
            "error": "Spotify integration not available",
        }), 503

    try:
        track_ids = _get_user_saved_track_ids()
    except RuntimeError as e:
        if "not connected" in str(e).lower() or "failed to connect" in str(e).lower():
            return jsonify({
                "success": False,
                "error": "Spotify not connected. Please connect Spotify in the app first.",
            }), 401
        raise
    except Exception as e:
        log.error("Failed to get user saved tracks: %r", e)
        return jsonify({
            "success": False,
            "error": f"Failed to fetch Spotify library: {str(e)}",
        }), 500

    try:
        total = len(track_ids)
        by_cluster: Dict[int, int] = {}
        in_lookup_ids: List[str] = []
        not_in_lookup_ids: List[str] = []

        for tid in track_ids:
            c = get_cluster_only(str(tid))
            if c is not None:
                by_cluster[c] = by_cluster.get(c, 0) + 1
                in_lookup_ids.append(tid)
            else:
                not_in_lookup_ids.append(tid)

        return jsonify({
            "success": True,
            "total_saved": total,
            "in_lookup": len(in_lookup_ids),
            "not_in_lookup": len(not_in_lookup_ids),
            "coverage_pct": round(100.0 * len(in_lookup_ids) / total, 1) if total else 0,
            "by_cluster": {str(k): v for k, v in sorted(by_cluster.items())},
            "sample_in_lookup": in_lookup_ids[:10],
            "sample_not_in_lookup": not_in_lookup_ids[:10],
        })
    except Exception as e:
        log.error("recs coverage failed: %r", e)
        return jsonify({"success": False, "error": str(e)}), 500


def _cluster_df_for_knn(
    df: pd.DataFrame,
    cluster_id: Optional[int],
    prob_threshold: Optional[float],
    cluster_method: str,
) -> pd.DataFrame:
    """
    Return subset of clustered df to use for KNN.
    - If cluster_id is None: use full df (pick_cluster will choose by BPM).
    - If cluster_method == "kmeans": filter by cluster == cluster_id only.
    - If cluster_method == "gmm" and prob_threshold is not None: filter by
      prob_{cluster_id} >= prob_threshold; else same as kmeans (hard cluster).
    """
    if cluster_id is None:
        return df.copy()
    if cluster_method != "gmm" or prob_threshold is None:
        out = df[df["cluster"] == cluster_id].copy()
        return out
    prob_col = f"prob_{cluster_id}"
    if prob_col not in df.columns:
        return df[df["cluster"] == cluster_id].copy()
    return df[df[prob_col] >= float(prob_threshold)].copy()


def _parse_artists(artist_names: str) -> List[str]:
    """Parse 'Artist A, Artist B & Artist C' into ['Artist A', 'Artist B', 'Artist C']."""
    if not artist_names or not isinstance(artist_names, str):
        return []
    parts = re.split(r", | & ", artist_names)
    return [p.strip() for p in parts if p.strip()]


def _fetch_spotify_track_details_internal(track_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch track details from Spotify (OAuth or client credentials).
    Returns list of dicts with id, track_id, name, artist_names, etc.
    """
    global _spotify, _spotify_cc
    formatted: List[Dict[str, Any]] = []

    if SPOTIFY_AVAILABLE and SpotifyIntegration is not None:
        try:
            if _spotify is None:
                _spotify = SpotifyIntegration()
                _spotify.connect()
            if _spotify and _spotify.is_connected():
                tracks = _spotify.get_tracks(track_ids)
                formatted = [_spotify.format_track_for_display(t) for t in tracks if t]
        except Exception as e:
            log.warning("Spotify OAuth for track details failed: %r", e)

    if not formatted:
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials

            cid = os.getenv("SPOTIPY_CLIENT_ID")
            secret = os.getenv("SPOTIPY_CLIENT_SECRET")
            if not cid or not secret:
                return [{"id": tid, "track_id": tid, "artist_names": ""} for tid in track_ids]
            if _spotify_cc is None:
                _spotify_cc = spotipy.Spotify(
                    auth_manager=SpotifyClientCredentials(client_id=cid, client_secret=secret),
                )
            for i in range(0, len(track_ids), 50):
                batch = track_ids[i : i + 50]
                resp = _spotify_cc.tracks(batch)
                for t in (resp.get("tracks") or []):
                    if not t:
                        continue
                    formatted.append({
                        "id": t.get("id"),
                        "name": t.get("name"),
                        "artist_names": ", ".join(a.get("name", "") for a in (t.get("artists") or [])),
                        "artists": t.get("artists", []),  # Keep full artist objects
                        "album": (t.get("album") or {}).get("name"),
                        "album_id": (t.get("album") or {}).get("id"),
                        "duration_ms": t.get("duration_ms"),
                        "explicit": t.get("explicit", False),
                        "popularity": t.get("popularity", 50),
                        "preview_url": t.get("preview_url"),
                        "external_urls": (t.get("external_urls") or {}).get("spotify"),
                        "images": (t.get("album") or {}).get("images"),
                        "release_date": (t.get("album") or {}).get("release_date"),
                        "track_id": t.get("id"),
                    })
        except Exception as e:
            log.warning("Spotify client-credentials track details failed: %r", e)
            formatted = [{"id": tid, "track_id": tid, "artist_names": ""} for tid in track_ids]

    return formatted


def _fetch_full_metadata_for_tracks(track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch full metadata for tracks including genres and related artists (for metadata prediction).
    Returns dict[track_id] -> metadata dict.
    """
    global _spotify, _spotify_cc

    metadata_map = {}

    # Get basic track info
    tracks_info = _fetch_spotify_track_details_internal(track_ids)

    # Get Spotify client (OAuth or client credentials)
    sp = None
    if SPOTIFY_AVAILABLE and SpotifyIntegration is not None:
        try:
            if _spotify is None:
                _spotify = SpotifyIntegration()
                _spotify.connect()
            if _spotify and _spotify.is_connected() and _spotify.sp:
                sp = _spotify.sp
        except Exception:
            pass

    if not sp:
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
            cid = os.getenv("SPOTIPY_CLIENT_ID")
            secret = os.getenv("SPOTIPY_CLIENT_SECRET")
            if cid and secret:
                if _spotify_cc is None:
                    _spotify_cc = spotipy.Spotify(
                        auth_manager=SpotifyClientCredentials(client_id=cid, client_secret=secret),
                    )
                sp = _spotify_cc
        except Exception:
            pass

    if not sp:
        # Fallback: return basic metadata without genres/related artists
        for track_info in tracks_info:
            tid = str(track_info.get("track_id") or track_info.get("id") or "")
            if tid:
                artists = track_info.get("artists", [])
                artist_ids = [a.get("id") for a in artists if isinstance(a, dict) and a.get("id")]
                metadata_map[tid] = {
                    "name": track_info.get("name", ""),
                    "artists": [a.get("name") if isinstance(a, dict) else str(a) for a in artists],
                    "artist_names": track_info.get("artist_names", ""),
                    "popularity": track_info.get("popularity", 50),
                    "release_date": track_info.get("release_date", ""),
                    "duration_ms": track_info.get("duration_ms", 180000),
                    "explicit": track_info.get("explicit", False),
                    "genres": [],  # Will be filled below if sp available
                    "main_artist_id": artist_ids[0] if artist_ids else None,
                    "related_artist_ids": [],  # Will be filled below if sp available
                }
        return metadata_map

    # Fetch genres and related artists for each track (with timeout protection)
    import time
    start_time = time.time()
    max_metadata_time = 10.0  # Max 10 seconds for metadata fetching
    
    for i, track_info in enumerate(tracks_info):
        # Timeout protection
        if time.time() - start_time > max_metadata_time:
            log.warning(f"Metadata fetch timeout after {i}/{len(tracks_info)} tracks")
            break
            
        tid = str(track_info.get("track_id") or track_info.get("id") or "")
        if not tid:
            continue

        artists = track_info.get("artists", [])
        artist_ids = [a.get("id") for a in artists if isinstance(a, dict) and a.get("id")]
        main_artist_id = artist_ids[0] if artist_ids else None

        genres = []
        related_artist_ids = []

        if main_artist_id:
            try:
                # Add timeout for individual API calls
                artist_info = sp.artist(main_artist_id)
                genres = artist_info.get("genres", [])

                # Get related artists (top 5) - skip if taking too long
                if time.time() - start_time < max_metadata_time - 1:
                    related = sp.artist_related_artists(main_artist_id)
                    related_artist_ids = [a.get("id") for a in related.get("artists", [])[:5]]
            except Exception as e:
                log.debug(f"Could not fetch artist info for {main_artist_id}: {e}")

        metadata_map[tid] = {
            "name": track_info.get("name", ""),
            "artists": [a.get("name") if isinstance(a, dict) else str(a) for a in artists],
            "artist_names": track_info.get("artist_names", ""),
            "popularity": track_info.get("popularity", 50),
            "release_date": track_info.get("release_date", ""),
            "duration_ms": track_info.get("duration_ms", 180000),
            "explicit": track_info.get("explicit", False),
            "genres": genres,
            "main_artist_id": main_artist_id,
            "related_artist_ids": related_artist_ids,
        }

    return metadata_map


def _top_artists_per_cluster(
    df_clustered: pd.DataFrame,
    max_tracks_per_cluster: int = 200,
    top_n: int = 3,
) -> Dict[int, List[str]]:
    """
    For each cluster, compute cumulative artist counts across tracks,
    then return top 2–3 artists per cluster. Uses Spotify track details.
    """
    result: Dict[int, List[str]] = {}
    cids = sorted(df_clustered["cluster"].unique())

    all_tids: List[str] = []
    cluster_tids: Dict[int, List[str]] = {}
    for cid in cids:
        subset = df_clustered[df_clustered["cluster"] == cid]
        tids = subset["track_id"].astype(str).dropna().unique().tolist()[:max_tracks_per_cluster]
        cluster_tids[cid] = tids
        all_tids.extend(tids)
    all_tids = list(dict.fromkeys(all_tids))
    if not all_tids:
        for cid in cids:
            result[cid] = []
        return result

    details = _fetch_spotify_track_details_internal(all_tids)
    tid_to_artists: Dict[str, List[str]] = {}
    for d in details:
        tid = str(d.get("track_id") or d.get("id") or "")
        artists = _parse_artists(d.get("artist_names") or "")
        tid_to_artists[tid] = artists

    for cid in cids:
        cnt: Counter = Counter()
        for tid in cluster_tids.get(cid, []):
            for a in tid_to_artists.get(tid, []):
                cnt[a] += 1
        total = sum(cnt.values())
        if not total:
            result[cid] = []
            continue
        by_count = sorted(cnt.keys(), key=lambda x: (-cnt[x], x))
        top = by_count[:top_n]
        result[cid] = top

    return result


def _cluster_names_from_backend(df_clustered: pd.DataFrame, cluster_method: str) -> Dict[int, Dict[str, Any]]:
    """Build name/tags/color per cluster via cluster_naming (or unique fallbacks)."""
    out: Dict[int, Dict[str, Any]] = {}
    colors = ["#EAE2B7", "#FCBF49", "#F77F00", "#D62828", "#003049", "#9B59B6", "#2ECC71", "#3498DB"]
    cids = sorted(df_clustered["cluster"].unique())

    if CLUSTER_NAMING_AVAILABLE and generate_cluster_names is not None:
        try:
            infos = generate_cluster_names(df_clustered)
            for info in infos:
                cid = int(info.get("id", -1))
                out[cid] = {
                    "name": info.get("name", f"Vibe {cid}"),
                    "tags": info.get("tags", []),
                    "color": info.get("color", colors[cid % len(colors)]),
                }
        except Exception as e:
            log.warning("cluster_naming failed: %r", e)

    for i, cid in enumerate(cids):
        if cid not in out:
            out[cid] = {
                "name": f"Vibe {cid}",
                "tags": [],
                "color": colors[cid % len(colors)],
            }

    # Ensure unique names (cluster_naming can return duplicates)
    used: Dict[str, int] = {}
    for cid in cids:
        meta = out[cid]
        name = meta.get("name", "")
        count = used.get(name, 0)
        used[name] = count + 1
        if count > 0:
            meta["name"] = f"{name} {count + 1}"
    return out


def _resolve_target_bpm(data: dict) -> Optional[float]:
    """Resolve target BPM from body: bpm, or pace_value + pace_unit via recs.pace_to_step_bpm."""
    bpm = data.get("bpm")
    if bpm is not None:
        return float(bpm)
    if RECS_AVAILABLE and pace_to_step_bpm is not None:
        pace_value = data.get("pace_value")
        pace_unit = data.get("pace_unit")
        if pace_value is not None and pace_unit:
            try:
                result = pace_to_step_bpm(float(pace_value), pace_unit)
                # pace_to_step_bpm returns a dict with 'step_bpm_final' key
                if isinstance(result, dict):
                    return float(result.get("step_bpm_final", result.get("step_bpm_raw", 0)))
                return float(result)
            except Exception as e:
                log.debug(f"Failed to convert pace to BPM: {e}")
                pass
    return None


def _load_tempo_lookup() -> Dict[str, float]:
    """Load tempo lookup map from track_lookup.db (fast SQLite) or CSV fallback."""
    global _tempo_lookup_cache
    if _tempo_lookup_cache is not None:
        return _tempo_lookup_cache

    _tempo_lookup_cache = {}
    try:
        model_dir = Path(__file__).resolve().parent.parent / "recs" / "model"
        
        # Try SQLite first (much faster than CSV)
        db_path = model_dir / "track_lookup.db"
        if db_path.exists():
            log.info(f"Loading tempo lookup from SQLite: {db_path}")
            conn = sqlite3.connect(str(db_path))
            try:
                # Get tempo from track_lookup table if it has tempo column, otherwise use test_set.csv
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='track_lookup'")
                if cursor.fetchone():
                    # Check if tempo column exists
                    cursor = conn.execute("PRAGMA table_info(track_lookup)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if "tempo" in columns:
                        # Use LIMIT to speed up initial load (we can load more on demand)
                        rows = conn.execute("SELECT track_id, tempo FROM track_lookup WHERE tempo > 0 LIMIT 500000").fetchall()
                        for tid, tempo in rows:
                            _tempo_lookup_cache[str(tid)] = float(tempo)
                        log.info(f"Loaded {len(_tempo_lookup_cache)} tempo values from SQLite")
                        return _tempo_lookup_cache
                    else:
                        log.info("track_lookup.db doesn't have tempo column, will use CSV fallback")
            except Exception as e:
                log.warning(f"SQLite tempo lookup failed: {e}, falling back to CSV")
            finally:
                conn.close()
        
        # Load from BOTH datasets for maximum coverage
        merged_training_path = model_dir / "merged_training.csv"  # 2.2M tracks
        test_set_path = model_dir / "test_set.csv"  # 439k tracks
        
        datasets_to_load = []
        if merged_training_path.exists():
            datasets_to_load.append(("merged_training.csv", merged_training_path))
        if test_set_path.exists():
            datasets_to_load.append(("test_set.csv", test_set_path))
        
        if datasets_to_load:
            log.info(f"Loading tempo lookup from {len(datasets_to_load)} dataset(s) (chunked)")
            chunk_size = 50000
            
            for dataset_name, csv_path in datasets_to_load:
                log.info(f"Loading from {dataset_name}...")
                rows_from_this_dataset = 0
                try:
                    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=["track_id", "tempo"], low_memory=False):
                        for _, row in chunk.iterrows():
                            tid = str(row["track_id"])
                            tempo = float(row["tempo"])
                            if tempo > 0:
                                # Only add if not already in cache (merged_training takes priority)
                                if tid not in _tempo_lookup_cache:
                                    _tempo_lookup_cache[tid] = tempo
                                    rows_from_this_dataset += 1
                        # Progress log every 100k rows
                        if len(_tempo_lookup_cache) % 100000 < chunk_size:
                            log.info(f"  Loaded {len(_tempo_lookup_cache):,} tempo values so far...")
                except Exception as e:
                    log.warning(f"Error loading from {dataset_name}: {e}, continuing...")
                    continue
                log.info(f"  Added {rows_from_this_dataset:,} new tempo values from {dataset_name}")
            
            log.info(f"Total loaded: {len(_tempo_lookup_cache):,} unique tempo values from {len(datasets_to_load)} dataset(s)")
    except Exception as e:
        log.warning(f"Could not load tempo lookup: {e}")

    return _tempo_lookup_cache


def _load_audio_features_lookup() -> Dict[str, Dict[str, float]]:
    """Load audio features lookup map from track_lookup.db (fast SQLite) or CSV fallback."""
    global _audio_features_lookup_cache
    if _audio_features_lookup_cache is not None:
        return _audio_features_lookup_cache

    _audio_features_lookup_cache = {}
    try:
        model_dir = Path(__file__).resolve().parent.parent / "recs" / "model"
        
        # Try SQLite first (much faster)
        db_path = model_dir / "track_lookup.db"
        if db_path.exists():
            log.info(f"Loading audio features from SQLite: {db_path}")
            conn = sqlite3.connect(str(db_path))
            try:
                # track_lookup has f0-f4 (scaled features), we need original tempo
                # Use test_set.csv for tempo, but we can get other features from recs inference
                # For now, fallback to CSV but use smaller file
                pass
            finally:
                conn.close()
        
        # Load from BOTH datasets for maximum coverage
        merged_training_path = model_dir / "merged_training.csv"  # 2.2M tracks
        test_set_path = model_dir / "test_set.csv"  # 439k tracks
        
        datasets_to_load = []
        if merged_training_path.exists():
            datasets_to_load.append(("merged_training.csv", merged_training_path))
        if test_set_path.exists():
            datasets_to_load.append(("test_set.csv", test_set_path))
        
        if datasets_to_load:
            log.info(f"Loading audio features from {len(datasets_to_load)} dataset(s) (chunked)")
            required_cols = ["track_id"] + FEATURE_COLS
            chunk_size = 50000
            
            for dataset_name, csv_path in datasets_to_load:
                log.info(f"Loading from {dataset_name}...")
                rows_from_this_dataset = 0
                try:
                    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=required_cols, low_memory=False):
                        for _, row in chunk.iterrows():
                            tid = str(row["track_id"])
                            tempo = float(row["tempo"])
                            if tempo > 0:  # Valid track
                                # Only add if not already in cache (merged_training takes priority)
                                if tid not in _audio_features_lookup_cache:
                                    _audio_features_lookup_cache[tid] = {
                                        "tempo": float(row["tempo"]),
                                        "energy": float(row["energy"]),
                                        "danceability": float(row["danceability"]),
                                        "valence": float(row["valence"]),
                                        "loudness": float(row["loudness"]),
                                    }
                                    rows_from_this_dataset += 1
                        # Progress log every 100k rows
                        if len(_audio_features_lookup_cache) % 100000 < chunk_size:
                            log.info(f"  Loaded {len(_audio_features_lookup_cache):,} audio feature sets so far...")
                except Exception as e:
                    log.warning(f"Error loading from {dataset_name}: {e}, continuing...")
                    continue
                log.info(f"  Added {rows_from_this_dataset:,} new audio feature sets from {dataset_name}")
            
            log.info(f"Total loaded: {len(_audio_features_lookup_cache):,} unique audio feature sets from {len(datasets_to_load)} dataset(s)")
    except Exception as e:
        log.warning(f"Could not load audio features lookup: {e}")

    return _audio_features_lookup_cache


def _get_tempo_for_track(track_id: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """
    Get tempo for a track:
    1. Try lookup from training CSV (fast)
    2. Try predicted tempo from metadata (if metadata provided)
    3. Return None if not available
    """
    # Try lookup first
    tempo_lookup = _load_tempo_lookup()
    tempo = tempo_lookup.get(str(track_id))
    if tempo is not None:
        return tempo

    # Try prediction from metadata
    if metadata and RECS_AVAILABLE and predict_audio_features_from_metadata:
        try:
            pred_features = predict_audio_features_from_metadata(metadata)
            if pred_features and "tempo" in pred_features:
                return float(pred_features["tempo"])
        except Exception as e:
            log.debug(f"Could not predict tempo for {track_id}: {e}")

    return None


@app.route('/api/clusters', methods=['POST'])
def get_clusters():
    """
    Run clustering on user's library.
    Body: bpm (optional), pace_value + pace_unit (optional, recs), cluster_method "kmeans"|"gmm",
    n_clusters, csv_path, use_recs_model (optional). When use_recs_model=True: assign user's
    saved tracks to pre-trained recs clusters (no Anna's Archive needed). Returns cluster info.
    """
    global _clustered_df, _kmeans_model, _gmm_model, _scaler, _cluster_method, _last_bpm

    try:
        data = request.get_json() or {}
        csv_path = data.get("csv_path")
        n_clusters = data.get("n_clusters", 4)
        annas_limit = data.get("annas_limit")
        use_spotify_library = data.get("use_spotify_library", True)
        use_recs_model = data.get("use_recs_model", False)
        cluster_method = (data.get("cluster_method") or "kmeans").lower()
        if cluster_method not in ("kmeans", "gmm"):
            cluster_method = "kmeans"
        bpm = data.get("bpm")
        if bpm is None and use_recs_model:
            bpm = _resolve_target_bpm(data)

        # Recs path: filter by BPM first, then cluster only filtered tracks
        if use_recs_model:
            if not RECS_AVAILABLE:
                return jsonify({
                    "success": False,
                    "error": "recs module not available. Make sure it's installed and models are trained.",
                }), 503
            # No longer require Spotify connection - we cluster the dataset instead
            # (Spotify OAuth still needed for playback, but not for clustering)

        if use_recs_model and RECS_AVAILABLE:
            try:
                # Resolve target BPM from pace or bpm
                target_bpm = _resolve_target_bpm(data)
                if target_bpm is None:
                    return jsonify({
                        "success": False,
                        "error": "Must provide either 'bpm' or 'pace_value' + 'pace_unit'",
                    }), 400

                log.info(f"Target BPM resolved: {target_bpm}")
                
                # Load dataset from training CSV instead of user's library
                model_dir = Path(__file__).resolve().parent.parent / "recs" / "model"
                dataset_path = model_dir / "test_set.csv"  # Use test_set.csv (has all tracks with metadata)
                
                # Fallback to demo_tracks.csv (real Spotify IDs with audio features)
                if not dataset_path.exists():
                    dataset_path = Path(__file__).resolve().parent.parent / "demo_tracks.csv"
                    if not dataset_path.exists():
                        return jsonify({
                            "success": False,
                            "error": "Dataset CSV not found. Please ensure training dataset is available.",
                        }), 404
                
                log.info(f"Loading tracks from dataset: {dataset_path}")
                df_dataset = pd.read_csv(dataset_path, low_memory=False)
                
                # Standardize column names
                if "id" in df_dataset.columns and "track_id" not in df_dataset.columns:
                    df_dataset = df_dataset.rename(columns={"id": "track_id"})
                
                # Ensure required columns exist
                required_cols = ["track_id"] + FEATURE_COLS
                missing_cols = [c for c in required_cols if c not in df_dataset.columns]
                if missing_cols:
                    return jsonify({
                        "success": False,
                        "error": f"Dataset missing required columns: {missing_cols}",
                    }), 400
                
                log.info(f"Loaded {len(df_dataset):,} tracks from dataset")
                
                # Filter by BPM range
                log.info(f"Filtering dataset tracks by BPM {target_bpm} +/- 15...")
                df_filtered = df_dataset[
                    (df_dataset["tempo"] > 0) & 
                    (df_dataset["tempo"].notna()) &
                    (abs(df_dataset["tempo"] - target_bpm) <= 15)
                ].copy()
                
                log.info(f"Filtered to {len(df_filtered):,} tracks in BPM range (from {len(df_dataset):,} total)")
                
                if len(df_filtered) == 0:
                    return jsonify({
                        "success": True,
                        "clusters": [],
                        "total_tracks": len(df_dataset),
                        "filtered_tracks": 0,
                        "cluster_method": "recs",
                        "message": f"No tracks found within {target_bpm} +/- 15 BPM. Try a different pace.",
                    })
                
                # Limit to reasonable number for clustering (max 5000 tracks)
                if len(df_filtered) > 5000:
                    log.info(f"Limiting to 5000 tracks for faster clustering (sampling from {len(df_filtered):,})")
                    df_filtered = df_filtered.sample(n=5000, random_state=42).reset_index(drop=True)
                
                # Prepare filtered tracks for clustering
                filtered_tracks: List[tuple] = []
                for _, row in df_filtered.iterrows():
                    filtered_tracks.append((
                        str(row["track_id"]),
                        float(row["tempo"]),
                        {
                            "tempo": float(row["tempo"]),
                            "energy": float(row["energy"]),
                            "danceability": float(row["danceability"]),
                            "valence": float(row["valence"]),
                            "loudness": float(row["loudness"]),
                        }
                    ))
                
                log.info(f"Prepared {len(df_filtered)} tracks for clustering")

                # Step 2: Cluster only the filtered tracks
                # Keep metadata separate, prepare features-only dataframe for clustering
                df_features = df_filtered[["track_id"] + FEATURE_COLS].copy()
                
                # Ensure all feature columns are numeric and valid
                for col in FEATURE_COLS:
                    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
                
                # Drop any rows with NaN in features
                df_features = df_features.dropna(subset=FEATURE_COLS)
                
                log.info(f"Final dataset for clustering: {len(df_features)} tracks")

                # Cluster using KMeans
                n_clusters_actual = min(n_clusters, len(df_features))  # Don't cluster more than tracks
                if n_clusters_actual < 2:
                    n_clusters_actual = 1

                X = df_features[FEATURE_COLS].values.astype(float)
                scaler_filtered = StandardScaler()
                X_scaled = scaler_filtered.fit_transform(X)

                km = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                df_features["cluster"] = labels

                # Log cluster sizes
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                log.info(f"Cluster sizes: {dict(cluster_counts)}")
                for cid, count in cluster_counts.items():
                    if count < 5:
                        log.warning(f"Cluster {cid} has only {count} tracks - may result in limited recommendations")

                # Store for KNN later - store features + cluster assignments
                global _clustered_df, _scaler, _dataset_metadata
                _clustered_df = df_features.copy()  # Features + cluster for KNN
                _scaler = scaler_filtered
                _cluster_method = "kmeans"
                _last_bpm = target_bpm
                
                # Store dataset metadata lookup for track names/artists (convert track_id to string for lookup)
                _dataset_metadata = {}
                for _, row in df_filtered.iterrows():
                    tid = str(row["track_id"])
                    _dataset_metadata[tid] = {
                        "track_name": str(row.get("track_name", row.get("name", ""))),
                        "artists": str(row.get("artists", row.get("artist_names", ""))),
                        "album_name": str(row.get("album_name", row.get("album", ""))),
                    }
                log.info(f"Stored metadata for {len(_dataset_metadata)} tracks")

                # Merge cluster assignments back with full dataset (includes metadata)
                df_clustered = df_features[["track_id", "cluster"]].merge(
                    df_filtered,
                    on="track_id",
                    how="inner"
                )
                
                # Build cluster stats with track metadata from CSV
                by_cluster: Dict[int, List[str]] = {}
                track_metadata: Dict[str, dict] = {}  # track_id -> {name, artists, etc.}
                
                for _, row in df_clustered.iterrows():
                    cid = int(row["cluster"])
                    tid = str(row["track_id"])
                    by_cluster.setdefault(cid, []).append(tid)
                    
                    # Store metadata from CSV for cluster naming
                    track_metadata[tid] = {
                        "name": str(row.get("track_name", row.get("name", ""))),
                        "artists": str(row.get("artists", row.get("artist_names", ""))),
                        "album": str(row.get("album_name", row.get("album", ""))),
                    }

                # Generate cluster names from top artists in each cluster
                top_artists_map = {}
                colors = ["#EAE2B7", "#FCBF49", "#F77F00", "#D62828", "#003049", "#9B59B6", "#2ECC71", "#3498DB"]
                
                for cid, tids in sorted(by_cluster.items()):
                    cluster_data = df_clustered[df_clustered["cluster"] == cid]
                    mean_energy = cluster_data["energy"].mean()
                    mean_danceability = cluster_data["danceability"].mean()
                    mean_valence = cluster_data["valence"].mean()
                    
                    # Get top artists from cluster tracks
                    artist_counts: Counter = Counter()
                    for tid in tids[:200]:  # Sample first 200 tracks
                        artists_str = track_metadata.get(tid, {}).get("artists", "")
                        if artists_str:
                            # Parse artists (handle comma/ampersand separated)
                            artists = [a.strip() for a in re.split(r",|&", artists_str) if a.strip()]
                            for artist in artists[:2]:  # Max 2 artists per track
                                artist_counts[artist] += 1
                    
                    # Get top 2-3 artists
                    top_artists = [artist for artist, _ in artist_counts.most_common(3)]
                    
                    # Generate name: use top artists if available, otherwise use features
                    if top_artists:
                        name = ", ".join(top_artists[:2])
                    elif mean_energy > 0.7 and mean_danceability > 0.7:
                        name = "High Energy"
                    elif mean_energy < 0.4 and mean_valence < 0.4:
                        name = "Chill Vibes"
                    elif mean_danceability > 0.75:
                        name = "Dance Floor"
                    elif mean_valence > 0.7:
                        name = "Happy Vibes"
                    else:
                        name = f"Vibe {cid}"
                    
                    top_artists_map[cid] = top_artists if top_artists else [name]

                cluster_stats = []
                for cid in sorted(by_cluster.keys()):
                    cluster_data = df_clustered[df_clustered["cluster"] == cid]
                    tids = by_cluster[cid]
                    cluster_name = ", ".join(top_artists_map.get(cid, [])) if top_artists_map.get(cid) else f"Vibe {cid}"
                    cluster_stats.append({
                        "cluster_id": int(cid),
                        "count": len(tids),
                        "mean_tempo": float(cluster_data["tempo"].mean()),
                        "mean_energy": float(cluster_data["energy"].mean()),
                        "mean_danceability": float(cluster_data["danceability"].mean()),
                        "name": cluster_name,
                        "top_artists": top_artists_map.get(cid, []),
                        "tags": [],
                        "color": colors[int(cid) % len(colors)],
                    })

                # No predicted features needed - using only lookup cache
                # (predicted_features_map removed since we skip metadata prediction)

                return jsonify({
                    "success": True,
                    "clusters": cluster_stats,
                    "total_tracks": len(df_dataset),
                    "filtered_tracks": len(filtered_tracks),
                    "cluster_method": "recs",
                    "target_bpm": target_bpm,
                    "bpm_range": f"{target_bpm - 15:.1f} - {target_bpm + 15:.1f}",
                    "note": "Clustering dataset tracks (not user library) - no Spotify API calls needed",
                })
            except Exception as e:
                log.error("use_recs_model path failed: %r", e, exc_info=True)
                import traceback
                error_details = traceback.format_exc()
                log.error(f"Full traceback: {error_details}")
                if use_recs_model and not (RECS_AVAILABLE and get_cluster_for_track):
                    return jsonify({"success": False, "error": "recs model not available"}), 503
                return jsonify({
                    "success": False,
                    "error": f"Clustering failed: {str(e)}",
                    "details": error_details[-500:] if len(error_details) > 500 else error_details,  # Last 500 chars
                }), 500

        # Load data
        if csv_path:
            df = load_features(csv_path)
        else:
            df = pd.DataFrame()
            if use_spotify_library:
                try:
                    ids = _get_user_saved_track_ids()
                    df = _load_features_from_annas_archive_for_track_ids(ids)
                    log.info("Loaded %d / %d saved tracks from Anna's Archive", len(df), len(ids))
                except Exception as e:
                    log.warning("Could not load from Spotify library via Anna's Archive: %r", e)
            if df.empty and os.path.exists(ANNAS_DB_PATH):
                limit = int(annas_limit) if annas_limit is not None else ANNAS_DEFAULT_LIMIT
                df = _load_features_from_annas_archive(limit=limit)
                log.info("Loaded %d tracks from Anna's Archive (global slice, limit=%d)", len(df), limit)
            if df.empty:
                df = load_features(None)

        # BPM-aware filter: subset by tempo so cluster counts change with BPM
        if bpm is not None:
            pre = len(df)
            df = _filter_df_by_bpm(df, float(bpm))
            log.info("BPM filter (bpm=%s): %d -> %d tracks", bpm, pre, len(df))

        # Cluster
        if cluster_method == "gmm":
            df_clustered, gmm, scaler = run_gmm(df, n_components=n_clusters)
            _clustered_df = df_clustered
            _kmeans_model = None
            _gmm_model = gmm
        else:
            df_clustered, km, scaler = run_kmeans(df, n_clusters=n_clusters)
            _clustered_df = df_clustered
            _kmeans_model = km
            _gmm_model = None
        _scaler = scaler
        _cluster_method = cluster_method
        _last_bpm = bpm

        # Top 2–3 artists per cluster (from cumulative % across tracks) for name/description
        top_artists_map = _top_artists_per_cluster(df_clustered, max_tracks_per_cluster=200, top_n=3)
        colors = ["#EAE2B7", "#FCBF49", "#F77F00", "#D62828", "#003049", "#9B59B6", "#2ECC71", "#3498DB"]

        cluster_stats = []
        for cid in sorted(df_clustered["cluster"].unique()):
            cluster_data = df_clustered[df_clustered["cluster"] == cid]
            top_artists = top_artists_map.get(int(cid), [])
            name = ", ".join(top_artists) if top_artists else f"Vibe {cid}"
            s = {
                "cluster_id": int(cid),
                "count": len(cluster_data),
                "mean_tempo": float(cluster_data["tempo"].mean()),
                "mean_energy": float(cluster_data["energy"].mean()),
                "mean_danceability": float(cluster_data["danceability"].mean()),
                "name": name,
                "top_artists": top_artists,
                "tags": [],
                "color": colors[int(cid) % len(colors)],
            }
            if cluster_method == "gmm" and f"prob_{cid}" in df_clustered.columns:
                s["mean_prob"] = float(cluster_data[f"prob_{cid}"].mean())
            cluster_stats.append(s)

        return jsonify({
            "success": True,
            "clusters": cluster_stats,
            "total_tracks": len(df_clustered),
            "cluster_method": cluster_method,
        })

    except Exception as e:
        log.error(f"Error in get_clusters: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/clusters/viz', methods=['GET'])
def get_clusters_viz():
    """
    Return clustering data for visualization.
    Requires clustering to be run first (POST /api/clusters).
    Returns: clusters (stats), tracks (track_id, cluster, x, y, probs?), cluster_method.
    Uses PCA(2) on scaled features for (x, y).
    """
    global _clustered_df, _scaler, _cluster_method

    if _clustered_df is None or _scaler is None:
        return jsonify({
            "success": False,
            "error": "Run POST /api/clusters first",
        }), 400

    X = _clustered_df[FEATURE_COLS].values.astype(float)
    X_scaled = _scaler.transform(X)
    n_components = min(2, X_scaled.shape[0], X_scaled.shape[1])
    if n_components < 2:
        return jsonify({"success": False, "error": "Not enough data for 2D viz"}), 400
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    clusters = []
    for cid in sorted(_clustered_df["cluster"].unique()):
        cd = _clustered_df[_clustered_df["cluster"] == cid]
        s = {
            "cluster_id": int(cid),
            "count": len(cd),
            "mean_tempo": float(cd["tempo"].mean()),
            "mean_energy": float(cd["energy"].mean()),
            "mean_danceability": float(cd["danceability"].mean()),
        }
        if _cluster_method == "gmm" and f"prob_{cid}" in _clustered_df.columns:
            s["mean_prob"] = float(cd[f"prob_{cid}"].mean())
        clusters.append(s)

    tracks = []
    prob_cols = [c for c in _clustered_df.columns if c.startswith("prob_")]
    for idx in range(len(_clustered_df)):
        row = _clustered_df.iloc[idx]
        t = {
            "track_id": row.get("track_id"),
            "cluster": int(row["cluster"]),
            "x": float(coords[idx, 0]),
            "y": float(coords[idx, 1]),
        }
        if prob_cols:
            t["probs"] = {int(c.split("_")[1]): float(row[c]) for c in prob_cols}
        tracks.append(t)

    return jsonify({
        "success": True,
        "clusters": clusters,
        "tracks": tracks,
        "cluster_method": _cluster_method,
    })


@app.route('/api/tracks', methods=['POST'])
def get_tracks():
    """
    Get tracks for a given BPM and cluster using KNN.
    Body: bpm (or pace_value + pace_unit via recs), cluster_id (optional), topk, prob_threshold (GMM only).
    """
    global _clustered_df, _scaler, _cluster_method

    try:
        data = request.get_json() or {}
        target_bpm = data.get("bpm")
        if target_bpm is None:
            target_bpm = _resolve_target_bpm(data)
        cluster_id = data.get("cluster_id")
        topk = data.get("topk", 10)
        prob_threshold = data.get("prob_threshold")

        if target_bpm is None:
            return jsonify({"success": False, "error": "BPM or pace_value+pace_unit is required"}), 400

        if _clustered_df is None or _scaler is None:
            df = load_features()
            _clustered_df, _, _scaler = run_kmeans(df, n_clusters=4)
            _cluster_method = "kmeans"

        if cluster_id is None:
            cluster_id, _mt, _td = pick_cluster_by_tempo(_clustered_df, target_bpm)

        df_candidates = _cluster_df_for_knn(
            _clustered_df, cluster_id, prob_threshold, _cluster_method
        )
        log.info(f"Cluster {cluster_id}: Found {len(df_candidates)} candidate tracks for KNN (requested topk={topk})")
        if df_candidates.empty:
            return jsonify({"success": False, "error": "No tracks found"}), 404

        # If cluster has very few tracks (<= topk), return all tracks sorted by tempo similarity
        # Otherwise use KNN to find best matches
        if len(df_candidates) <= topk:
            log.info(f"Cluster {cluster_id} has {len(df_candidates)} tracks (<= {topk}), returning all tracks")
            # Sort by tempo similarity to target_bpm
            df_candidates = df_candidates.copy()
            df_candidates["tempo_diff"] = (df_candidates["tempo"] - target_bpm).abs()
            picks = df_candidates.nsmallest(len(df_candidates), "tempo_diff")
            # Add metadata and format
            picks_dict = picks.to_dict("records")
            formatted_picks = []
            for pick in picks_dict:
                tid = str(pick["track_id"])
                metadata = _dataset_metadata.get(tid, {})
                track_name = pick.get("name") or metadata.get("track_name", "")
                artists = pick.get("artists") or metadata.get("artists", "")
                formatted_picks.append({
                    "track_id": tid,
                    "name": track_name,
                    "artists": artists,
                    "cluster": int(pick.get("cluster", cluster_id)),
                    "tempo": float(pick["tempo"]),
                    "energy": float(pick.get("energy", 0)),
                    "danceability": float(pick.get("danceability", 0)),
                    "valence": float(pick.get("valence", 0)),
                    "loudness": float(pick.get("loudness", 0)),
                    "distance": float(pick.get("tempo_diff", 0)),
                    "rank": len(formatted_picks) + 1,
                })
            picks = pd.DataFrame(formatted_picks)
        else:
            picks = _knn_on_df(
                df_candidates, _scaler, target_bpm, topk, cluster_id_override=cluster_id
            )
        log.info(f"Returning {len(picks)} tracks for cluster {cluster_id} (requested topk={topk})")
        if picks.empty:
            return jsonify({"success": False, "error": "No tracks found"}), 404

        tracks = picks.to_dict("records")
        return jsonify({
            "success": True,
            "cluster_id": cluster_id,
            "tracks": tracks,
            "count": len(tracks),
        })

    except Exception as e:
        log.error(f"Error in get_tracks: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/tracks/from-track', methods=['POST'])
def get_tracks_from_track():
    """
    KNN using a track's feature vector as query (not BPM).
    Body: track_id (required), cluster_id (optional), topk, use_recs (optional, default True).

    Priority:
    1. Recs lookup (fast path for tracks in training set)
    2. Predicted features cache (from recent clustering)
    3. Metadata prediction (on-the-fly)
    4. Fallback to in-memory clustering
    """
    global _clustered_df, _scaler, _predicted_features_cache, _spotify

    try:
        data = request.get_json() or {}
        track_id = data.get("track_id")
        cluster_id = data.get("cluster_id")
        topk = min(data.get("topk", 10), MAX_TRACKS_FOR_KNN)  # Cap at rate limit safe max for queue extension
        use_recs_flag = data.get("use_recs", True)

        if not track_id:
            return jsonify({"success": False, "error": "track_id is required"}), 400

        # Try recs model lookup first (pre-trained lookup): fastest path
        if RECS_AVAILABLE and use_recs_flag and get_cluster_and_neighbors is not None:
            try:
                result = get_cluster_and_neighbors(str(track_id), top_k=int(topk))
                if result is not None:
                    recs_cluster_id, neighbor_ids = result
                    if cluster_id is not None and int(cluster_id) != recs_cluster_id:
                        neighbor_ids = []  # client asked for a different cluster
                    if not neighbor_ids:
                        tracks_out = []
                    else:
                        details = _fetch_spotify_track_details_internal(neighbor_ids)
                        tracks_out = []
                        for rank, tid in enumerate(neighbor_ids[: int(topk)], start=1):
                            d = next((x for x in details if str(x.get("track_id") or x.get("id")) == str(tid)), None)
                            if d:
                                d = dict(d)
                                d["cluster"] = recs_cluster_id
                                d["rank"] = rank
                                tracks_out.append(d)
                            else:
                                tracks_out.append({
                                    "track_id": tid,
                                    "cluster": recs_cluster_id,
                                    "rank": rank,
                                })
                    return jsonify({
                        "success": True,
                        "cluster_id": recs_cluster_id,
                        "tracks": tracks_out,
                        "count": len(tracks_out),
                        "source": "recs_lookup",
                    })
            except FileNotFoundError:
                log.warning("recs track lookup not found, trying predicted features")
            except Exception as e:
                log.warning("recs get_cluster_and_neighbors failed: %r", e)

        # Try predicted features cache (from recent clustering with metadata)
        if RECS_AVAILABLE and predict_cluster_from_features is not None:
            # Check if we have predicted features for this track
            predicted_features = _predicted_features_cache.get(str(track_id))
            if predicted_features:
                try:
                    pred_cluster = predict_cluster_from_features(predicted_features)
                    if pred_cluster is not None:
                        # Use KNN with predicted features against all tracks in same cluster
                        # For now, return neighbors from lookup in same cluster
                        # TODO: Implement KNN using predicted features against predicted features cache
                        return jsonify({
                            "success": True,
                            "cluster_id": pred_cluster,
                            "tracks": [],  # Would need to implement KNN on predicted features
                            "count": 0,
                            "source": "predicted_features_cache",
                            "message": "Track clustered but KNN on predicted features not yet implemented",
                        })
                except Exception as e:
                    log.warning("Predicted features KNN failed: %r", e)

        # Fallback: in-memory clustering (track must be in _clustered_df)
        if _clustered_df is None or _scaler is None:
            return jsonify({
                "success": False,
                "error": "Track not in recs lookup. Run POST /api/clusters first so we can use in-memory KNN.",
            }), 400

        picks = _knn_from_track(
            _clustered_df, _scaler, str(track_id), topk=int(topk), cluster_id=cluster_id
        )
        if picks.empty:
            return jsonify({"success": False, "error": "Track not found or no neighbors"}), 404

        # Get Spotify track details for playback (batch efficiently)
        track_ids = picks["track_id"].tolist()
        tracks_with_details = []
        
        if SPOTIFY_AVAILABLE:
            if _spotify is None:
                try:
                    _spotify = SpotifyIntegration()
                    _spotify.connect()
                except Exception as e:
                    log.warning(f"Spotify connection failed: {e}, using dataset metadata only")
            
            if _spotify and _spotify.is_connected():
                try:
                    # Batch fetch: Spotify allows 50 tracks per request
                    spotify_tracks = _spotify.get_tracks(track_ids)
                    spotify_map = {t["id"]: _spotify.format_track_for_display(t) for t in spotify_tracks if t}
                    
                    # Merge Spotify details with KNN metadata
                    picks_dict = picks.to_dict("records")
                    for pick in picks_dict:
                        tid = str(pick["track_id"])
                        spotify_detail = spotify_map.get(tid, {})
                        # Merge: Spotify details override, but keep KNN metadata
                        merged = {**pick, **spotify_detail}
                        tracks_with_details.append(merged)
                except Exception as e:
                    log.warning(f"Failed to fetch Spotify details: {e}, using dataset metadata")
                    tracks_with_details = picks.to_dict("records")
            else:
                tracks_with_details = picks.to_dict("records")
        else:
            tracks_with_details = picks.to_dict("records")
        
        return jsonify({
            "success": True,
            "cluster_id": int(cluster_id) if cluster_id is not None else None,
            "tracks": tracks_with_details,
            "count": len(tracks_with_details),
            "source": "in_memory",
        })
    except Exception as e:
        log.error(f"Error in get_tracks_from_track: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/tracks/details', methods=['POST'])
def get_track_details():
    """
    Get track details from dataset metadata (CSV) or Spotify API fallback.
    Since we're clustering dataset tracks, prefer CSV metadata (faster, no API calls).
    """
    global _dataset_metadata
    try:
        data = request.get_json() or {}
        track_ids = data.get("track_ids", [])

        if not track_ids:
            return jsonify({"success": False, "error": "track_ids is required"}), 400

        # Try to get metadata from dataset first (fast, no API calls)
        formatted_tracks = []
        missing_ids = []
        
        for tid in track_ids:
            tid_str = str(tid)
            metadata = _dataset_metadata.get(tid_str)
            if metadata:
                # Use CSV metadata
                formatted_tracks.append({
                    "id": tid_str,
                    "track_id": tid_str,
                    "name": metadata.get("track_name", ""),
                    "artist_names": metadata.get("artists", ""),
                    "artists": metadata.get("artists", "").split(", ") if metadata.get("artists") else [],
                    "album": metadata.get("album_name", ""),
                    "preview_url": None,  # Not available in CSV
                    "external_urls": f"https://open.spotify.com/track/{tid_str}",
                })
            else:
                missing_ids.append(tid_str)
        
        # If some tracks not in dataset, try Spotify API (but this should be rare now)
        if missing_ids and SPOTIFY_AVAILABLE:
            try:
                spotify_tracks = _fetch_spotify_track_details_internal(missing_ids)
                formatted_tracks.extend(spotify_tracks)
            except Exception as e:
                log.warning(f"Could not fetch {len(missing_ids)} tracks from Spotify: {e}")
                # Add placeholders for missing tracks
                for tid in missing_ids:
                    formatted_tracks.append({
                        "id": tid,
                        "track_id": tid,
                        "name": f"Track {tid[:8]}...",
                        "artist_names": "Unknown",
                    })

        return jsonify({
            "success": True,
            "tracks": formatted_tracks,
            "count": len(formatted_tracks),
        })

    except Exception as e:
        log.error(f"Error in get_track_details: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/cluster/tracks', methods=['POST'])
def get_cluster_tracks():
    """
    Combined: Get tracks for BPM + cluster, then fetch full details.
    Body: bpm (or pace_value + pace_unit via recs), cluster_id (optional), topk, prob_threshold (GMM only).
    """
    global _clustered_df, _scaler, _spotify, _cluster_method, _dataset_metadata

    try:
        data = request.get_json() or {}
        target_bpm = data.get("bpm")
        if target_bpm is None:
            target_bpm = _resolve_target_bpm(data)
        cluster_id = data.get("cluster_id")
        topk = min(data.get("topk", 10), MAX_TRACKS_PER_CLUSTER)  # Cap at rate limit safe max
        prob_threshold = data.get("prob_threshold")

        if target_bpm is None:
            return jsonify({"success": False, "error": "BPM or pace_value+pace_unit is required"}), 400

        if _clustered_df is None or _scaler is None:
            df = load_features()
            _clustered_df, _, _scaler = run_kmeans(df, n_clusters=4)
            _cluster_method = "kmeans"

        if cluster_id is None:
            cluster_id, _mt, _td = pick_cluster_by_tempo(_clustered_df, target_bpm)

        df_candidates = _cluster_df_for_knn(
            _clustered_df, cluster_id, prob_threshold, _cluster_method
        )
        if df_candidates.empty:
            return jsonify({"success": False, "error": "No tracks found"}), 404

        picks = _knn_on_df(
            df_candidates, _scaler, target_bpm, topk, cluster_id_override=cluster_id
        )

        if picks.empty:
            return jsonify({"success": False, "error": "No tracks found"}), 404

        # Get Spotify track details for playback (batch efficiently, max 50 per request)
        track_ids = picks["track_id"].tolist()
        tracks_with_details = []
        
        if SPOTIFY_AVAILABLE:
            if _spotify is None:
                try:
                    _spotify = SpotifyIntegration()
                    _spotify.connect()
                except Exception as e:
                    log.warning(f"Spotify connection failed: {e}, using dataset metadata only")
            
            if _spotify and _spotify.is_connected():
                try:
                    # Batch fetch: Spotify allows 50 tracks per request
                    spotify_tracks = _spotify.get_tracks(track_ids)
                    spotify_map = {t["id"]: _spotify.format_track_for_display(t) for t in spotify_tracks if t}
                    
                    # Merge Spotify details with KNN metadata
                    picks_dict = picks.to_dict("records")
                    for pick in picks_dict:
                        tid = str(pick["track_id"])
                        spotify_detail = spotify_map.get(tid, {})
                        # Merge: Spotify details override, but keep KNN metadata
                        merged = {**pick, **spotify_detail}
                        tracks_with_details.append(merged)
                except Exception as e:
                    log.warning(f"Failed to fetch Spotify details: {e}, using dataset metadata")
                    tracks_with_details = picks.to_dict("records")
            else:
                tracks_with_details = picks.to_dict("records")
        else:
            tracks_with_details = picks.to_dict("records")
        
        return jsonify({
            "success": True,
            "cluster_id": cluster_id,
            "tracks": tracks_with_details,
            "count": len(tracks_with_details),
        })

    except Exception as e:
        log.error(f"Error in get_cluster_tracks: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/playback/start', methods=['POST'])
def playback_start():
    """
    Start playback of track(s). Requires Spotify OAuth (user).
    Body: uris — list of Spotify URIs, e.g. ["spotify:track:ID"].
    """
    global _spotify

    if not SPOTIFY_AVAILABLE or SpotifyIntegration is None:
        return jsonify({"success": False, "error": "Spotify integration not available"}), 503
    try:
        if _spotify is None:
            _spotify = SpotifyIntegration()
            _spotify.connect()
        if not _spotify or not _spotify.is_connected() or not _spotify.sp:
            return jsonify({"success": False, "error": "Spotify not connected"}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 401

    data = request.get_json() or {}
    uris = data.get("uris", [])
    if not uris:
        return jsonify({"success": False, "error": "uris required"}), 400
    # Ensure spotify:track: form
    uris = [u if u.startswith("spotify:") else f"spotify:track:{u}" for u in uris]

    try:
        _spotify.sp.start_playback(uris=uris)
        return jsonify({"success": True})
    except Exception as e:
        log.error("playback start failed: %r", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/playback/queue', methods=['POST'])
def playback_queue():
    """
    Add a track to the playback queue. Requires Spotify OAuth (user).
    Body: uri — Spotify URI, e.g. "spotify:track:ID".
    """
    global _spotify

    if not SPOTIFY_AVAILABLE or SpotifyIntegration is None:
        return jsonify({"success": False, "error": "Spotify integration not available"}), 503
    try:
        if _spotify is None:
            _spotify = SpotifyIntegration()
            _spotify.connect()
        if not _spotify or not _spotify.is_connected() or not _spotify.sp:
            return jsonify({"success": False, "error": "Spotify not connected"}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 401

    data = request.get_json() or {}
    uri = data.get("uri") or data.get("track_id")
    if not uri:
        return jsonify({"success": False, "error": "uri or track_id required"}), 400
    if not uri.startswith("spotify:"):
        uri = f"spotify:track:{uri}"

    try:
        _spotify.sp.add_to_queue(uri)
        return jsonify({"success": True})
    except Exception as e:
        log.error("playback queue failed: %r", e)
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
            "/api/clusters/viz",
            "/api/tracks",
            "/api/tracks/from-track",
            "/api/tracks/details",
            "/api/cluster/tracks",
            "/api/recs/coverage",
            "/api/playback/start",
            "/api/playback/queue",
        ]
    }), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8888))  # Using 8888 to match Spotify redirect URI
    host = "127.0.0.1"
    print(f"\n🚀 HeartBeats API Server")
    print(f"📍 Running on http://{host}:{port}")
    print(f"🔗 Health check: http://localhost:{port}/api/health")
    print(f"\nAvailable endpoints:")
    print(f"  POST /api/clusters     (bpm, use_recs_model, cluster_method, n_clusters)")
    print(f"  GET  /api/clusters/viz")
    print(f"  POST /api/tracks       (bpm or pace_value+pace_unit, cluster_id, topk)")
    print(f"  POST /api/tracks/from-track  (track_id, topk, use_recs)")
    print(f"  POST /api/tracks/details")
    print(f"  POST /api/cluster/tracks")
    print(f"  GET  /api/recs/coverage  (test Spotify library vs recs lookup)")
    print(f"  POST /api/playback/start  (uris)")
    print(f"  POST /api/playback/queue  (uri)")
    print(f"\n⚠️  Note: Frontend should be accessed via Vite dev server (usually http://localhost:5173)\n")
    app.run(host=host, port=port, debug=True)
