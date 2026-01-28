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
    return jsonify({"status": "ok"})


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
                        "album": (t.get("album") or {}).get("name"),
                        "album_id": (t.get("album") or {}).get("id"),
                        "duration_ms": t.get("duration_ms"),
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


@app.route('/api/clusters', methods=['POST'])
def get_clusters():
    """
    Run clustering on user's library.
    Body: bpm (optional), cluster_method "kmeans"|"gmm", n_clusters, csv_path, etc.
    When bpm provided: filter by |tempo - bpm| <= BPM_TOLERANCE, then cluster. Counts change with BPM.
    Returns cluster information including name, tags, color per cluster.
    """
    global _clustered_df, _kmeans_model, _gmm_model, _scaler, _cluster_method, _last_bpm

    try:
        data = request.get_json() or {}
        csv_path = data.get("csv_path")
        n_clusters = data.get("n_clusters", 4)
        annas_limit = data.get("annas_limit")
        use_spotify_library = data.get("use_spotify_library", True)
        cluster_method = (data.get("cluster_method") or "kmeans").lower()
        if cluster_method not in ("kmeans", "gmm"):
            cluster_method = "kmeans"
        bpm = data.get("bpm")

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
    Body: bpm, cluster_id (optional), topk, prob_threshold (optional, 0–1; GMM only).
    When GMM + cluster_id + prob_threshold: only tracks with P(cluster) >= threshold.
    """
    global _clustered_df, _scaler, _cluster_method

    try:
        data = request.get_json() or {}
        target_bpm = data.get("bpm")
        cluster_id = data.get("cluster_id")
        topk = data.get("topk", 10)
        prob_threshold = data.get("prob_threshold")

        if target_bpm is None:
            return jsonify({"success": False, "error": "BPM is required"}), 400

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
    Body: track_id (required), cluster_id (optional), topk.
    Use when "now playing" changes to get next recommendations.
    """
    global _clustered_df, _scaler

    try:
        data = request.get_json() or {}
        track_id = data.get("track_id")
        cluster_id = data.get("cluster_id")
        topk = data.get("topk", 10)

        if not track_id:
            return jsonify({"success": False, "error": "track_id is required"}), 400
        if _clustered_df is None or _scaler is None:
            return jsonify({"success": False, "error": "Run POST /api/clusters first"}), 400

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
    Body: bpm, cluster_id (optional), topk, prob_threshold (optional; GMM only).
    """
    global _clustered_df, _scaler, _spotify, _cluster_method

    try:
        data = request.get_json() or {}
        target_bpm = data.get("bpm")
        cluster_id = data.get("cluster_id")
        topk = data.get("topk", 10)
        prob_threshold = data.get("prob_threshold")

        if target_bpm is None:
            return jsonify({"success": False, "error": "BPM is required"}), 400

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
    print(f"  POST /api/clusters     (bpm, cluster_method, n_clusters)")
    print(f"  GET  /api/clusters/viz")
    print(f"  POST /api/tracks       (bpm, cluster_id, prob_threshold)")
    print(f"  POST /api/tracks/from-track  (track_id, cluster_id, topk)")
    print(f"  POST /api/tracks/details")
    print(f"  POST /api/cluster/tracks")
    print(f"  POST /api/playback/start  (uris)")
    print(f"  POST /api/playback/queue  (uri)")
    print(f"\n⚠️  Note: Frontend should be accessed via Vite dev server (usually http://localhost:5173)\n")
    app.run(host="0.0.0.0", port=port, debug=True)
