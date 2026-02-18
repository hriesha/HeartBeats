"""
Deezer API client with in-memory caching and rate limiting.

All Deezer API interactions go through this module.
No authentication required. Rate limit: ~45 req/5s (conservative).
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import requests

log = logging.getLogger("deezer_client")

DEEZER_BASE = "https://api.deezer.com"

RATE_LIMIT_REQUESTS = 45
RATE_LIMIT_WINDOW_SECONDS = 5

CACHE_TTL_CHART = 3600        # 1 hour
CACHE_TTL_SEARCH = 1800       # 30 min
CACHE_TTL_TRACK_DETAIL = 86400  # 24 hours
CACHE_TTL_ARTIST_TRACKS = 3600  # 1 hour


@dataclass
class _CacheEntry:
    data: Any
    expires_at: float

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class _RateLimiter:
    """Sliding window rate limiter."""

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS,
                 window: float = RATE_LIMIT_WINDOW_SECONDS):
        self._max = max_requests
        self._window = window
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                cutoff = now - self._window
                self._timestamps = [t for t in self._timestamps if t > cutoff]
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
            time.sleep(0.15)


class DeezerClient:
    """Thread-safe Deezer API client with caching and rate limiting."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._cache: Dict[str, _CacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._rate_limiter = _RateLimiter()

    def _cache_get(self, key: str) -> Optional[Any]:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired:
                return entry.data
            if entry:
                del self._cache[key]
        return None

    def _cache_set(self, key: str, data: Any, ttl: float) -> None:
        with self._cache_lock:
            self._cache[key] = _CacheEntry(data=data, expires_at=time.time() + ttl)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def _get(self, path: str, params: Optional[Dict] = None,
             retries: int = 3) -> Optional[Dict]:
        url = f"{DEEZER_BASE}{path}"
        for attempt in range(retries):
            self._rate_limiter.acquire()
            try:
                resp = self._session.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" in data:
                        log.warning("Deezer API error: %s", data["error"])
                        return None
                    return data
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    log.warning("Deezer 429, backing off %ds", wait)
                    time.sleep(wait)
                    continue
                log.warning("Deezer HTTP %d for %s", resp.status_code, url)
            except requests.RequestException as e:
                log.warning("Deezer request error (attempt %d): %s", attempt + 1, e)
                if attempt < retries - 1:
                    time.sleep(1)
        return None

    # ------------------------------------------------------------------ public

    def get_chart_tracks(self, genre_id: int, limit: int = 50,
                         index: int = 0) -> List[Dict[str, Any]]:
        cache_key = f"chart:{genre_id}:{limit}:{index}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        data = self._get(f"/chart/{genre_id}/tracks",
                         params={"limit": limit, "index": index})
        if not data or "data" not in data:
            return []

        tracks = data["data"]
        self._cache_set(cache_key, tracks, CACHE_TTL_CHART)
        return tracks

    def search_tracks_by_bpm(self, query: str, bpm_min: int, bpm_max: int,
                             limit: int = 50, index: int = 0) -> Tuple[List[Dict], int]:
        cache_key = f"search:{query}:{bpm_min}:{bpm_max}:{limit}:{index}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        data = self._get("/search/track", params={
            "q": query,
            "bpm_min": bpm_min,
            "bpm_max": bpm_max,
            "limit": limit,
            "index": index,
        })
        if not data or "data" not in data:
            return [], 0

        result = (data["data"], data.get("total", 0))
        self._cache_set(cache_key, result, CACHE_TTL_SEARCH)
        return result

    def get_track_detail(self, track_id: int) -> Optional[Dict[str, Any]]:
        cache_key = f"track:{track_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        data = self._get(f"/track/{track_id}")
        if not data or "error" in data:
            return None

        self._cache_set(cache_key, data, CACHE_TTL_TRACK_DETAIL)
        return data

    def get_genre_artists(self, genre_id: int) -> List[Dict[str, Any]]:
        cache_key = f"genre_artists:{genre_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        data = self._get(f"/genre/{genre_id}/artists")
        if not data or "data" not in data:
            return []

        artists = data["data"]
        self._cache_set(cache_key, artists, CACHE_TTL_CHART)
        return artists

    def get_artist_top_tracks(self, artist_id: int,
                              limit: int = 25) -> List[Dict[str, Any]]:
        cache_key = f"artist_top:{artist_id}:{limit}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        data = self._get(f"/artist/{artist_id}/top",
                         params={"limit": limit})
        if not data or "data" not in data:
            return []

        tracks = data["data"]
        self._cache_set(cache_key, tracks, CACHE_TTL_ARTIST_TRACKS)
        return tracks

    def get_track_bpm(self, track_id: int) -> Optional[float]:
        """Get BPM for a track. Tries Deezer metadata first, falls back to audio analysis."""
        # Check BPM cache first
        cache_key = f"bpm:{track_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        detail = self.get_track_detail(track_id)
        if not detail:
            return None

        # Try Deezer's metadata BPM
        bpm = detail.get("bpm", 0)
        if bpm and bpm > 0:
            self._cache_set(cache_key, float(bpm), CACHE_TTL_TRACK_DETAIL)
            return float(bpm)

        # Fallback: detect BPM from 30s audio preview
        preview_url = detail.get("preview")
        if preview_url:
            detected = self._detect_bpm_from_preview(preview_url)
            if detected:
                self._cache_set(cache_key, detected, CACHE_TTL_TRACK_DETAIL)
                return detected

        return None

    def _detect_bpm_from_preview(self, preview_url: str) -> Optional[float]:
        """Download Deezer 30s preview and detect BPM via audio analysis."""
        try:
            import tempfile
            import os
            import signal

            resp = self._session.get(preview_url, timeout=8)
            if resp.status_code != 200:
                return None

            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name

            try:
                import librosa
                import numpy as np
                # Use mono, lower sample rate for speed
                audio, sr = librosa.load(tmp_path, sr=11025, mono=True)
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                bpm = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])
                return round(bpm, 1) if bpm > 0 else None
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            log.warning("BPM detection failed for preview: %s", e)
            return None

    def batch_get_track_bpms(self, track_ids: List[int]) -> Dict[int, float]:
        result: Dict[int, float] = {}
        for tid in track_ids:
            bpm = self.get_track_bpm(tid)
            if bpm is not None:
                result[tid] = bpm
        return result
