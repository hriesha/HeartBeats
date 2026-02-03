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

from flask import Flask, request, jsonify
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

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]

# Anna's Archive (local) audio features DB (gitignored, but present locally)
ANNAS_DB_PATH = os.environ.get(
    "ANNAS_ARCHIVE_DB",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "annas_archive_data", "spotify_clean_audio_features.sqlite3"),
)
ANNAS_DEFAULT_LIMIT = int(os.environ.get("ANNAS_ARCHIVE_LIMIT", "25000"))
SPOTIFY_LIBRARY_LIMIT = int(os.environ.get("SPOTIFY_LIBRARY_LIMIT", "2000"))
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
    if df.empty:
        return pd.DataFrame()
    X = df[FEATURE_COLS].values.astype(float)
    X_scaled = scaler.transform(X)
    n_neighbors = min(topk, len(df))
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
    rows = []
    for rank, (idx, d) in enumerate(zip(inds, dists), start=1):
        r = df.iloc[idx]
        c = int(cluster_id_override) if cluster_id_override is not None else int(r.get("cluster"))
        out = {
            "track_id": r.get("track_id"),
            "name": r.get("name"),
            "artists": r.get("artists"),
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
        c = int(cluster_id) if cluster_id is not None else int(r.get("cluster"))
        out = {
            "track_id": r.get("track_id"),
            "name": r.get("name"),
            "artists": r.get("artists"),
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

    # Fetch genres and related artists for each track
    for track_info in tracks_info:
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
                artist_info = sp.artist(main_artist_id)
                genres = artist_info.get("genres", [])

                # Get related artists (top 5)
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
    """Load tempo lookup map from training CSV (track_id -> tempo)."""
    global _tempo_lookup_cache
    if _tempo_lookup_cache is not None:
        return _tempo_lookup_cache

    _tempo_lookup_cache = {}
    try:
        # Try to load from training CSV (test_set.csv has all tracks)
        model_dir = Path(__file__).resolve().parent.parent / "recs" / "model"
        test_set_path = model_dir / "test_set.csv"
        train_set_path = model_dir.parent.parent / "enriched_kaggle.csv"  # Full training set

        for csv_path in [test_set_path, train_set_path]:
            if csv_path.exists():
                log.info(f"Loading tempo lookup from {csv_path}")
                df = pd.read_csv(csv_path)
                if "track_id" in df.columns and "tempo" in df.columns:
                    for _, row in df.iterrows():
                        tid = str(row["track_id"])
                        tempo = float(row["tempo"])
                        if tempo > 0:
                            _tempo_lookup_cache[tid] = tempo
                    log.info(f"Loaded {len(_tempo_lookup_cache)} tempo values from {csv_path}")
                    break
    except Exception as e:
        log.warning(f"Could not load tempo lookup: {e}")

    return _tempo_lookup_cache


def _load_audio_features_lookup() -> Dict[str, Dict[str, float]]:
    """Load audio features lookup map from training CSV (track_id -> {tempo, energy, ...})."""
    global _audio_features_lookup_cache
    if _audio_features_lookup_cache is not None:
        return _audio_features_lookup_cache

    _audio_features_lookup_cache = {}
    try:
        model_dir = Path(__file__).resolve().parent.parent / "recs" / "model"
        test_set_path = model_dir / "test_set.csv"
        train_set_path = model_dir.parent.parent / "enriched_kaggle.csv"

        for csv_path in [test_set_path, train_set_path]:
            if csv_path.exists():
                log.info(f"Loading audio features lookup from {csv_path}")
                df = pd.read_csv(csv_path)
                required_cols = ["track_id"] + FEATURE_COLS
                if all(col in df.columns for col in required_cols):
                    for _, row in df.iterrows():
                        tid = str(row["track_id"])
                        tempo = float(row["tempo"])
                        if tempo > 0:  # Valid track
                            _audio_features_lookup_cache[tid] = {
                                "tempo": float(row["tempo"]),
                                "energy": float(row["energy"]),
                                "danceability": float(row["danceability"]),
                                "valence": float(row["valence"]),
                                "loudness": float(row["loudness"]),
                            }
                    log.info(f"Loaded {len(_audio_features_lookup_cache)} audio feature sets from {csv_path}")
                    break
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
            if not use_spotify_library:
                return jsonify({
                    "success": False,
                    "error": "use_spotify_library must be True when using recs model.",
                }), 400

            # Check Spotify connection
            try:
                _get_user_saved_track_ids()
            except RuntimeError as e:
                return jsonify({
                    "success": False,
                    "error": f"Spotify not connected: {str(e)}. Please connect Spotify first.",
                }), 401

        if use_recs_model and RECS_AVAILABLE and get_cluster_for_track is not None and use_spotify_library:
            try:
                # Resolve target BPM from pace or bpm
                target_bpm = _resolve_target_bpm(data)
                if target_bpm is None:
                    return jsonify({
                        "success": False,
                        "error": "Must provide either 'bpm' or 'pace_value' + 'pace_unit'",
                    }), 400

                ids = _get_user_saved_track_ids()
                if not ids:
                    return jsonify({
                        "success": True,
                        "clusters": [],
                        "total_tracks": 0,
                        "cluster_method": "recs",
                        "message": "No saved tracks found. Save some tracks in Spotify first.",
                    })

                log.info(f"Filtering {len(ids)} user tracks by BPM {target_bpm} +/- 15...")

                # Fetch full metadata for all tracks (including genres, related artists)
                all_metadata = _fetch_full_metadata_for_tracks(ids)

                # Step 1: Filter tracks by tempo (+/- 15 BPM)
                filtered_tracks: List[tuple] = []  # (track_id, tempo, metadata, audio_features)
                predicted_features_map: Dict[str, dict] = {}  # track_id -> predicted audio features

                # Load audio features lookup for tracks in training set
                audio_features_lookup = _load_audio_features_lookup()
                log.info(f"Loaded {len(audio_features_lookup)} tracks from audio features lookup")

                tracks_with_tempo = 0
                tracks_without_tempo = 0
                tracks_in_range = 0
                tracks_out_of_range = 0
                tracks_from_lookup = 0
                tracks_from_prediction = 0

                for tid in ids:
                    metadata = all_metadata.get(tid, {})

                    # Get tempo: try lookup first, then metadata prediction
                    tempo = _get_tempo_for_track(tid, metadata)

                    if tempo is None:
                        tracks_without_tempo += 1
                        continue  # Skip tracks without tempo

                    tracks_with_tempo += 1

                    # Filter: keep only tracks within +/- 15 BPM
                    if abs(tempo - target_bpm) <= 15:
                        tracks_in_range += 1
                        # Get audio features: try lookup first, then prediction
                        audio_features = audio_features_lookup.get(tid)

                        if audio_features:
                            tracks_from_lookup += 1
                        else:
                            # Predict from metadata
                            if predict_audio_features_from_metadata:
                                try:
                                    pred_features = predict_audio_features_from_metadata(metadata)
                                    if pred_features:
                                        audio_features = pred_features
                                        predicted_features_map[tid] = pred_features
                                        tracks_from_prediction += 1
                                except Exception as e:
                                    log.debug(f"Metadata prediction failed for {tid}: {e}")

                        if audio_features:
                            filtered_tracks.append((tid, tempo, metadata, audio_features))
                    else:
                        tracks_out_of_range += 1

                log.info(f"Tempo stats: {tracks_with_tempo} with tempo, {tracks_without_tempo} without tempo")
                log.info(f"Range stats: {tracks_in_range} in range ({target_bpm}±15), {tracks_out_of_range} out of range")
                log.info(f"Features source: {tracks_from_lookup} from lookup, {tracks_from_prediction} from prediction")

                log.info(f"Filtered to {len(filtered_tracks)} tracks within {target_bpm} +/- 15 BPM")

                if not filtered_tracks:
                    return jsonify({
                        "success": True,
                        "clusters": [],
                        "total_tracks": 0,
                        "filtered_tracks": 0,
                        "cluster_method": "recs",
                        "message": f"No tracks found within {target_bpm} +/- 15 BPM. Try a different pace.",
                    })

                # Step 2: Cluster only the filtered tracks
                # Build DataFrame from filtered tracks with audio features
                filtered_data = []
                for tid, tempo, metadata, audio_features in filtered_tracks:
                    filtered_data.append({
                        "track_id": tid,
                        "tempo": audio_features.get("tempo", tempo),
                        "energy": audio_features.get("energy", 0.5),
                        "danceability": audio_features.get("danceability", 0.5),
                        "valence": audio_features.get("valence", 0.5),
                        "loudness": audio_features.get("loudness", -10.0),
                    })

                df_filtered = pd.DataFrame(filtered_data)

                # Cluster using KMeans
                n_clusters_actual = min(n_clusters, len(df_filtered))  # Don't cluster more than tracks
                if n_clusters_actual < 2:
                    n_clusters_actual = 1

                X = df_filtered[FEATURE_COLS].values.astype(float)
                scaler_filtered = StandardScaler()
                X_scaled = scaler_filtered.fit_transform(X)

                km = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                df_filtered["cluster"] = labels

                # Store for KNN later
                global _clustered_df, _scaler
                _clustered_df = df_filtered
                _scaler = scaler_filtered
                _cluster_method = "kmeans"
                _last_bpm = target_bpm

                # Build cluster stats
                by_cluster: Dict[int, List[str]] = {}
                for _, row in df_filtered.iterrows():
                    cid = int(row["cluster"])
                    tid = str(row["track_id"])
                    by_cluster.setdefault(cid, []).append(tid)

                all_tids = df_filtered["track_id"].tolist()
                details = _fetch_spotify_track_details_internal(all_tids)
                tid_to_artists = {}
                for d in details:
                    tid = str(d.get("track_id") or d.get("id") or "")
                    tid_to_artists[tid] = _parse_artists(d.get("artist_names") or "")

                top_artists_map = {}
                colors = ["#EAE2B7", "#FCBF49", "#F77F00", "#D62828", "#003049", "#9B59B6", "#2ECC71", "#3498DB"]
                for cid, tids in sorted(by_cluster.items()):
                    cnt = Counter()
                    for tid in tids[:200]:
                        for a in tid_to_artists.get(tid, []):
                            cnt[a] += 1
                    top = [k for k, _ in cnt.most_common(3)]
                    top_artists_map[cid] = top

                cluster_stats = []
                for cid in sorted(by_cluster.keys()):
                    cluster_data = df_filtered[df_filtered["cluster"] == cid]
                    tids = by_cluster[cid]
                    cluster_stats.append({
                        "cluster_id": int(cid),
                        "count": len(tids),
                        "mean_tempo": float(cluster_data["tempo"].mean()),
                        "mean_energy": float(cluster_data["energy"].mean()),
                        "mean_danceability": float(cluster_data["danceability"].mean()),
                        "name": ", ".join(top_artists_map.get(cid, [])) or f"Vibe {cid}",
                        "top_artists": top_artists_map.get(cid, []),
                        "tags": [],
                        "color": colors[int(cid) % len(colors)],
                    })

                # Store predicted features for KNN
                if predicted_features_map:
                    log.info(f"Stored predicted features for {len(predicted_features_map)} tracks")
                    global _predicted_features_cache
                    _predicted_features_cache = predicted_features_map

                return jsonify({
                    "success": True,
                    "clusters": cluster_stats,
                    "total_tracks": len(ids),
                    "filtered_tracks": len(filtered_tracks),
                    "cluster_method": "recs",
                    "target_bpm": target_bpm,
                    "bpm_range": f"{target_bpm - 15:.1f} - {target_bpm + 15:.1f}",
                    "stats": {
                        "tracks_with_tempo": tracks_with_tempo,
                        "tracks_without_tempo": tracks_without_tempo,
                        "tracks_in_range": tracks_in_range,
                        "tracks_out_of_range": tracks_out_of_range,
                        "tracks_from_lookup": tracks_from_lookup,
                        "tracks_from_prediction": tracks_from_prediction,
                    },
                })
            except Exception as e:
                log.error("use_recs_model path failed: %r", e, exc_info=True)
                if use_recs_model and not (RECS_AVAILABLE and get_cluster_for_track):
                    return jsonify({"success": False, "error": "recs model not available"}), 503
                # Re-raise to be handled by outer try/except
                raise

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
        if df_candidates.empty:
            return jsonify({"success": False, "error": "No tracks found"}), 404

        picks = _knn_on_df(
            df_candidates, _scaler, target_bpm, topk, cluster_id_override=cluster_id
        )
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
    global _clustered_df, _scaler, _predicted_features_cache

    try:
        data = request.get_json() or {}
        track_id = data.get("track_id")
        cluster_id = data.get("cluster_id")
        topk = data.get("topk", 10)
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

        tracks = picks.to_dict("records")
        return jsonify({
            "success": True,
            "cluster_id": int(cluster_id) if cluster_id is not None else None,
            "tracks": tracks,
            "count": len(tracks),
            "source": "in_memory",
        })
    except Exception as e:
        log.error(f"Error in get_tracks_from_track: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/tracks/details', methods=['POST'])
def get_track_details():
    """
    Get full track details from Spotify API for a list of track IDs.
    """
    try:
        data = request.get_json() or {}
        track_ids = data.get("track_ids", [])

        if not track_ids:
            return jsonify({"success": False, "error": "track_ids is required"}), 400

        formatted_tracks = _fetch_spotify_track_details_internal(track_ids)

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
    global _clustered_df, _scaler, _spotify, _cluster_method

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
        if df_candidates.empty:
            return jsonify({"success": False, "error": "No tracks found"}), 404

        picks = _knn_on_df(
            df_candidates, _scaler, target_bpm, topk, cluster_id_override=cluster_id
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
    port = int(os.environ.get("PORT", 5001))  # Changed from 5000 to 5001 (5000 is used by AirPlay)
    print(f"\n🚀 HeartBeats API Server")
    print(f"📍 Running on http://0.0.0.0:{port}")
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
    app.run(host="0.0.0.0", port=port, debug=True)
