#!/usr/bin/env python3
"""
HeartBeats API Backend
Flask endpoint to handle:
1. Genre-based vibes (powered by Deezer chart + search)
2. BPM-filtered track recommendations
3. Apple Music developer token serving
"""

import os
import sys
import logging
import math
import random
import time
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Add project root and api dir to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_API_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _API_DIR)

from deezer_client import DeezerClient
from vibe_config import get_vibes_for_bpm, VIBE_DEFINITIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger("heartbeats_api")

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:*",
    "http://127.0.0.1:*",
    "capacitor://localhost",
    "*",
])

BPM_TOLERANCE = 10

# Module-level Deezer client (singleton)
_deezer = DeezerClient()

# Cache: "vibe_id:bpm" -> (timestamp, list of tracks)
_vibe_tracks_cache: Dict[str, tuple] = {}
_VIBE_CACHE_TTL = 300  # 5 min — keeps BPM analysis cached but rotates songs


# =============================================================================
# Helpers
# =============================================================================

KM_PER_MILE = 1.609344


def _resolve_target_bpm(data: dict) -> Optional[float]:
    """Resolve target BPM from body: bpm directly, or pace_value + pace_unit."""
    bpm = data.get("bpm")
    if bpm is not None:
        return float(bpm)
    pace_value = data.get("pace_value")
    pace_unit = data.get("pace_unit")
    if pace_value is not None and pace_unit:
        try:
            pv = float(pace_value)
            if pv <= 0 or not math.isfinite(pv):
                return None
            speed_mph = 60.0 / pv
            if pace_unit == "min/km":
                speed_mph /= KM_PER_MILE
            raw = 125.0 + 5.5 * speed_mph  # linear cadence model
            return max(140.0, min(200.0, round(raw)))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _deezer_track_to_heartbeats(
    dt: Dict[str, Any],
    vibe_id: int,
    rank: int = 0,
    bpm: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert a Deezer track dict to HeartBeats Track format."""
    artist = dt.get("artist", {})
    album = dt.get("album", {})

    images = []
    for size_key, dim in [("cover_xl", 1000), ("cover_big", 500),
                          ("cover_medium", 250), ("cover_small", 56)]:
        url = album.get(size_key)
        if url:
            images.append({"url": url, "height": dim, "width": dim})

    return {
        "track_id": str(dt.get("id", "")),
        "id": str(dt.get("id", "")),
        "name": dt.get("title", dt.get("title_short", "")),
        "artists": artist.get("name", ""),
        "artist_names": artist.get("name", ""),
        "cluster": vibe_id,
        "tempo": bpm or 0,
        "distance": 0,
        "rank": rank,
        "album": album.get("title", ""),
        "duration_ms": (dt.get("duration", 0)) * 1000,
        "images": images,
    }


def _fetch_vibe_tracks(
    vibe: Dict[str, Any],
    target_bpm: float,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """
    Fetch tracks for a vibe (genre) at the given BPM.

    Sources:
    1. Chart tracks for the genre (popular, recent) — random offset
    2. BPM-filtered search using randomly picked keywords + random offsets
    3. Top tracks from genre's top artists (randomized selection)
    """
    vibe_id = vibe["vibe_id"]
    genre_id = vibe["genre_id"]
    bpm_min = int(target_bpm - BPM_TOLERANCE)
    bpm_max = int(target_bpm + BPM_TOLERANCE)

    all_tracks: Dict[int, Dict] = {}  # dedup by Deezer track ID

    # Source 1: Chart tracks (most popular first)
    chart_tracks = _deezer.get_chart_tracks(genre_id, limit=50)
    for dt in chart_tracks:
        tid = dt.get("id")
        if tid and tid not in all_tracks:
            all_tracks[tid] = dt

    # Source 2: BPM-filtered search — use top 2 keywords (most popular results)
    picked_keywords = vibe.get("search_keywords", [])[:2]
    for keyword in picked_keywords:
        results, _ = _deezer.search_tracks_by_bpm(
            keyword, bpm_min, bpm_max, limit=40
        )
        for dt in results:
            tid = dt.get("id")
            if tid and tid not in all_tracks:
                all_tracks[tid] = dt

    # Source 3: Top artist tracks — only when pool is thin
    if len(all_tracks) < limit:
        genre_artists = _deezer.get_genre_artists(genre_id)
        if genre_artists:
            picked_artists = random.sample(genre_artists, min(2, len(genre_artists)))
            for artist in picked_artists:
                artist_tracks = _deezer.get_artist_top_tracks(artist["id"], limit=10)
                for dt in artist_tracks:
                    tid = dt.get("id")
                    if tid and tid not in all_tracks:
                        all_tracks[tid] = dt

    # Sort candidates by popularity — chart hits first
    candidates = list(all_tracks.items())
    candidates.sort(key=lambda x: -(x[1].get("rank", 0) or 0))

    # Quick BPM check: use Deezer metadata (instant, no librosa)
    # For tracks with BPM 0 (most popular songs), just use target BPM —
    # they're from genre charts/BPM search so they're appropriate
    result = []
    artist_count: Dict[str, int] = {}
    MAX_PER_ARTIST = 2

    for tid, dt in candidates:
        # Try Deezer metadata BPM first (instant, cached)
        detail = _deezer.get_track_detail(tid)
        meta_bpm = detail.get("bpm", 0) if detail else 0

        if meta_bpm and meta_bpm > 0:
            # Has real BPM — check if in range
            if abs(meta_bpm - target_bpm) > BPM_TOLERANCE:
                continue
            actual_bpm = float(meta_bpm)
        else:
            # No BPM metadata — trust genre chart / BPM search placement
            actual_bpm = target_bpm

        artist_name = dt.get("artist", {}).get("name", "").lower()
        if artist_count.get(artist_name, 0) >= MAX_PER_ARTIST:
            continue

        artist_count[artist_name] = artist_count.get(artist_name, 0) + 1
        result.append(_deezer_track_to_heartbeats(
            dt, vibe_id, bpm=actual_bpm
        ))
        if len(result) >= limit:
            break

    # Shuffle to avoid same order every time
    random.shuffle(result)

    for i, track in enumerate(result, start=1):
        track["rank"] = i

    return result


# =============================================================================
# Endpoints
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "deezer_available": True,
        "apple_music_token_available": bool(os.environ.get("VITE_APPLE_MUSIC_DEVELOPER_TOKEN")),
    })


@app.route('/api/apple-music/developer-token', methods=['GET'])
def apple_music_developer_token():
    token = os.environ.get("VITE_APPLE_MUSIC_DEVELOPER_TOKEN")
    if not token:
        return jsonify({"success": False, "error": "Developer token not configured"}), 503
    return jsonify({"success": True, "developer_token": token})


@app.route('/api/clusters', methods=['POST'])
def get_clusters():
    """Return genre-based vibes for the given BPM."""
    try:
        data = request.get_json() or {}
        target_bpm = _resolve_target_bpm(data)

        if target_bpm is None:
            return jsonify({
                "success": False,
                "error": "Must provide either 'bpm' or 'pace_value' + 'pace_unit'",
            }), 400

        vibes = get_vibes_for_bpm(target_bpm)
        bpm_min = int(target_bpm - BPM_TOLERANCE)
        bpm_max = int(target_bpm + BPM_TOLERANCE)

        cluster_stats = []
        for vibe in vibes:
            # Quick chart fetch (cached) for track count + top artists
            chart_tracks = _deezer.get_chart_tracks(vibe["genre_id"], limit=50)
            _, search_total = _deezer.search_tracks_by_bpm(
                vibe["search_keywords"][0], bpm_min, bpm_max, limit=1,
            )
            track_count = max(len(chart_tracks), search_total, 30)

            top_artist_names = []
            seen = set()
            for ct in chart_tracks[:20]:
                aname = ct.get("artist", {}).get("name", "")
                if aname and aname not in seen:
                    seen.add(aname)
                    top_artist_names.append(aname)
                if len(top_artist_names) >= 3:
                    break

            cluster_stats.append({
                "cluster_id": vibe["vibe_id"],
                "count": track_count,
                "mean_tempo": target_bpm,
                "mean_energy": 0.6,
                "mean_danceability": 0.6,
                "name": vibe["name"],
                "top_artists": top_artist_names,
                "tags": vibe["tags"],
                "color": vibe["color"],
            })

        return jsonify({
            "success": True,
            "clusters": cluster_stats,
            "total_tracks": sum(c["count"] for c in cluster_stats),
            "cluster_method": "genre_vibe",
            "target_bpm": target_bpm,
        })

    except Exception as e:
        log.error(f"Error in get_clusters: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/tracks', methods=['POST'])
def get_tracks():
    """Get tracks for a given BPM and vibe (cluster_id)."""
    try:
        data = request.get_json() or {}
        target_bpm = data.get("bpm")
        if target_bpm is None:
            target_bpm = _resolve_target_bpm(data)
        cluster_id = data.get("cluster_id")
        topk = min(data.get("topk", 20), 50)

        if target_bpm is None:
            return jsonify({"success": False, "error": "BPM or pace is required"}), 400

        target_bpm = float(target_bpm)

        # Find the vibe definition
        vibe = None
        if cluster_id is not None:
            for v in VIBE_DEFINITIONS:
                if v["vibe_id"] == int(cluster_id):
                    vibe = v
                    break
        if vibe is None:
            vibes = get_vibes_for_bpm(target_bpm)
            vibe = vibes[0] if vibes else VIBE_DEFINITIONS[0]

        # Check cache (with TTL)
        cache_key = f"{vibe['vibe_id']}:{int(target_bpm)}"
        cached = _vibe_tracks_cache.get(cache_key)
        now = time.time()

        if cached and (now - cached[0]) < _VIBE_CACHE_TTL and len(cached[1]) >= topk:
            tracks = cached[1][:topk]
        else:
            tracks = _fetch_vibe_tracks(vibe, target_bpm, limit=max(topk, 30))
            _vibe_tracks_cache[cache_key] = (now, tracks)
            tracks = tracks[:topk]

        return jsonify({
            "success": True,
            "cluster_id": vibe["vibe_id"],
            "tracks": tracks,
            "count": len(tracks),
        })

    except Exception as e:
        log.error(f"Error in get_tracks: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/tracks/from-track', methods=['POST'])
def get_tracks_from_track():
    """Get more tracks similar to a given track (for queue extension)."""
    try:
        data = request.get_json() or {}
        track_id = data.get("track_id")
        cluster_id = data.get("cluster_id")
        topk = min(data.get("topk", 10), 30)
        exclude_ids = set(str(x) for x in data.get("exclude_ids", []))

        if not track_id:
            return jsonify({"success": False, "error": "track_id is required"}), 400

        # Get source track detail from Deezer
        detail = _deezer.get_track_detail(int(track_id))
        source_artist_id = None
        source_bpm = 0
        if detail:
            source_artist_id = detail.get("artist", {}).get("id")
            source_bpm = detail.get("bpm", 0) or 0

        # Resolve vibe
        vibe = None
        if cluster_id is not None:
            for v in VIBE_DEFINITIONS:
                if v["vibe_id"] == int(cluster_id):
                    vibe = v
                    break
        if vibe is None:
            vibe = VIBE_DEFINITIONS[0]

        result_tracks: List[Dict] = []
        seen_ids = set(exclude_ids)
        seen_ids.add(str(track_id))

        bpm_center = source_bpm if source_bpm > 0 else 150

        # Source 1: More from same artist
        if source_artist_id:
            artist_tracks = _deezer.get_artist_top_tracks(source_artist_id, limit=15)
            for dt in artist_tracks:
                tid = str(dt.get("id", ""))
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    result_tracks.append(
                        _deezer_track_to_heartbeats(dt, vibe["vibe_id"], bpm=bpm_center)
                    )

        # Source 2: BPM search in same genre (offset for fresh results)
        if len(result_tracks) < topk:
            for keyword in vibe.get("search_keywords", [])[:2]:
                more, _ = _deezer.search_tracks_by_bpm(
                    keyword,
                    int(bpm_center - BPM_TOLERANCE),
                    int(bpm_center + BPM_TOLERANCE),
                    limit=30,
                    index=50,
                )
                for dt in more:
                    tid = str(dt.get("id", ""))
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        result_tracks.append(
                            _deezer_track_to_heartbeats(dt, vibe["vibe_id"], bpm=bpm_center)
                        )
                if len(result_tracks) >= topk:
                    break

        # Source 3: Chart tracks (different offset)
        if len(result_tracks) < topk:
            chart = _deezer.get_chart_tracks(vibe["genre_id"], limit=50, index=50)
            for dt in chart:
                tid = str(dt.get("id", ""))
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    result_tracks.append(
                        _deezer_track_to_heartbeats(dt, vibe["vibe_id"], bpm=bpm_center)
                    )
                if len(result_tracks) >= topk:
                    break

        result_tracks = result_tracks[:topk]
        for i, t in enumerate(result_tracks, start=1):
            t["rank"] = i

        return jsonify({
            "success": True,
            "cluster_id": vibe["vibe_id"],
            "tracks": result_tracks,
            "count": len(result_tracks),
            "source": "deezer_similar",
        })

    except Exception as e:
        log.error(f"Error in get_tracks_from_track: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/tracks/details', methods=['POST'])
def get_track_details():
    """Get track details from Deezer by track IDs."""
    try:
        data = request.get_json() or {}
        track_ids = data.get("track_ids", [])

        if not track_ids:
            return jsonify({"success": False, "error": "track_ids is required"}), 400

        formatted = []
        for tid in track_ids:
            detail = _deezer.get_track_detail(int(tid))
            if detail:
                track_bpm = detail.get("bpm", 0) or 0
                formatted.append(_deezer_track_to_heartbeats(detail, 0, bpm=track_bpm))
            else:
                formatted.append({
                    "id": str(tid),
                    "track_id": str(tid),
                    "name": f"Track {str(tid)[:8]}...",
                    "artist_names": "",
                })

        return jsonify({
            "success": True,
            "tracks": formatted,
            "count": len(formatted),
        })

    except Exception as e:
        log.error(f"Error in get_track_details: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/cluster/tracks', methods=['POST'])
def get_cluster_tracks():
    """Alias for /api/tracks (kept for backwards compat)."""
    return get_tracks()


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": f"Route not found: {request.path}",
        "available_routes": [
            "/api/health",
            "/api/apple-music/developer-token",
            "/api/clusters",
            "/api/tracks",
            "/api/tracks/from-track",
            "/api/tracks/details",
            "/api/cluster/tracks",
        ]
    }), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8888))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n🚀 HeartBeats API Server (Deezer-powered)")
    print(f"📍 Running on http://{host}:{port}")
    print(f"🔗 Health check: http://localhost:{port}/api/health")
    print(f"\nAvailable endpoints:")
    print(f"  GET  /api/health")
    print(f"  GET  /api/apple-music/developer-token")
    print(f"  POST /api/clusters     (pace_value + pace_unit, or bpm)")
    print(f"  POST /api/tracks       (bpm, cluster_id, topk)")
    print(f"  POST /api/tracks/from-track  (track_id, cluster_id, topk)")
    print(f"  POST /api/tracks/details")
    print(f"  POST /api/cluster/tracks")
    print(f"\n⚠️  Note: Frontend should be accessed via Vite dev server (usually http://localhost:5173)\n")
    app.run(host=host, port=port, debug=True)
