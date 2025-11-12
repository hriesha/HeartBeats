#!/usr/bin/env python3
# heartbeats-matching.py

import os, time, logging, requests
from typing import List, Dict, Any, Iterable, Optional

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# ---------------------------
# Setup
# ---------------------------
load_dotenv()  # expects SPOTIPY_CLIENT_ID / SECRET / REDIRECT_URI in .env

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("heartbeats")

SCOPES = "user-library-read"
REDIRECT = os.environ.get("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(scope=SCOPES, redirect_uri=REDIRECT, cache_path=".cache-heartbeats")
)

# ---------------------------
# Helpers
# ---------------------------
def chunked(seq: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def get_saved_track_ids(limit: int = 50, max_pages: int = 20) -> List[str]:
    ids: List[str] = []
    offset = 0
    for page in range(max_pages):
        log.info(f"Fetching saved tracks: page {page+1}")
        res = sp.current_user_saved_tracks(limit=limit, offset=offset)
        items = res.get("items", [])
        if not items:
            break
        for it in items:
            tr = (it or {}).get("track") or {}
            if tr.get("is_local") or tr.get("type") != "track":
                continue
            tid = tr.get("id")
            if tid:
                ids.append(tid)
        offset += limit
    # de-dupe, keep order
    ids = list(dict.fromkeys(ids))
    log.info(f"Collected {len(ids)} saved track IDs")
    return ids

def get_track_meta(track_ids: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Return dict[track_id] -> {"name": str, "artists": "A, B"}
    """
    meta: Dict[str, Dict[str, str]] = {}
    for batch in chunked(track_ids, 50):  # up to 50 per call
        trks = sp.tracks(batch).get("tracks", [])
        for t in trks:
            if not t:
                continue
            tid = t.get("id")
            if not tid:
                continue
            name = t.get("name") or ""
            artists = ", ".join([a.get("name", "") for a in (t.get("artists") or [])])
            meta[tid] = {"name": name, "artists": artists}
        time.sleep(0.05)
    return meta

def spotify_ids_to_isrcs(track_ids: List[str]) -> Dict[str, str]:
    """
    Return dict[track_id] -> ISRC (external_ids.isrc)
    """
    id2isrc: Dict[str, str] = {}
    for batch in chunked(track_ids, 50):
        tr = sp.tracks(batch).get("tracks", [])
        for t in tr:
            if not t:
                continue
            tid = t.get("id")
            isrc = ((t.get("external_ids") or {}).get("isrc")) if t else None
            if tid and isrc:
                id2isrc[tid] = isrc
        time.sleep(0.05)
    log.info(f"Resolved ISRC for {len(id2isrc)}/{len(track_ids)} tracks")
    return id2isrc

# ------- Deezer lookups -------
def deezer_bpm_from_isrc(isrc: str) -> Optional[float]:
    """
    Query Deezer public API for BPM by ISRC. Returns float or None.
    """
    url = f"https://api.deezer.com/track/isrc:{isrc}"
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            data = r.json()
            bpm = data.get("bpm")
            return float(bpm) if bpm else None
    except Exception as e:
        log.warning(f"Deezer ISRC error for {isrc}: {e}")
    return None

def deezer_bpm_from_search(artist: str, title: str) -> Optional[float]:
    """
    Fallback: search by artist+title; returns BPM or None.
    """
    q = requests.utils.quote(f'artist:"{artist}" track:"{title}"')
    url = f"https://api.deezer.com/search?q={q}&limit=1"
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                bpm = data[0].get("bpm")
                return float(bpm) if bpm else None
    except Exception as e:
        log.debug(f"Deezer search error for {artist} - {title}: {e}")
    return None

def build_bpm_dataframe(
    track_ids: List[str],
    id2isrc: Dict[str, str],
    meta: Dict[str, Dict[str, str]],
    sleep_s: float = 0.05
) -> pd.DataFrame:
    """
    Build DF with columns: track_id, isrc, tempo, tempo_source
    Uses ISRC first; then searches Deezer by artist/title for IDs w/o ISRC.
    """
    rows: List[Dict[str, Any]] = []
    misses = 0

    # ISRC path
    for tid, isrc in id2isrc.items():
        bpm = deezer_bpm_from_isrc(isrc)
        if bpm:
            rows.append({"track_id": tid, "isrc": isrc, "tempo": bpm, "tempo_source": "deezer"})
        else:
            misses += 1
        time.sleep(sleep_s)

    if misses:
        log.info(f"Deezer BPM not found for {misses} tracks (ISRC path)")

    # Fallback for tracks lacking ISRC
    no_isrc_ids = [tid for tid in track_ids if tid not in id2isrc]
    if no_isrc_ids:
        log.info(f"Trying Deezer search fallback for {len(no_isrc_ids)} tracks without ISRC")
        for tid in no_isrc_ids:
            m = meta.get(tid, {})
            bpm = deezer_bpm_from_search(m.get("artists", ""), m.get("name", ""))
            if bpm:
                rows.append({"track_id": tid, "isrc": None, "tempo": bpm, "tempo_source": "deezer_search"})
            time.sleep(sleep_s)

    df = pd.DataFrame(rows, columns=["track_id", "isrc", "tempo", "tempo_source"]).drop_duplicates("track_id")
    return df

# ------- Convenience -------
def filter_by_tempo_band(df: pd.DataFrame, target_bpm: float, pct: float = 0.06) -> pd.DataFrame:
    low = target_bpm * (1 - pct)
    high = target_bpm * (1 + pct)
    return df[(df["tempo"] >= low) & (df["tempo"] <= high)]

import argparse

def best_tempo_match(song_bpm: float, target: float, consider_multiples: bool = True):
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

def closest_songs(df: pd.DataFrame, target_bpm: float, topk: int = 10, consider_multiples: bool = True) -> pd.DataFrame:
    """
    df columns expected: track_id, name, artists, isrc, tempo, tempo_source
    Returns a sorted DF of the topk closest to target_bpm (ascending delta).
    """
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
            "isrc": r.get("isrc"),
            "tempo": float(tempo),
            "tempo_source": r.get("tempo_source"),
            "matched_tempo": matched,
            "multiplier": mult,
            "delta": delta
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["delta", "tempo_source"]).head(topk).reset_index(drop=True)
    return out

# ---------------------------
# Main
# ---------------------------
def main():
    # 1) Saved track IDs
    track_ids = get_saved_track_ids()
    if not track_ids:
        log.warning("No saved tracks found.")
        return

    # 2) Metadata + ISRCs
    meta = get_track_meta(track_ids)
    id2isrc = spotify_ids_to_isrcs(track_ids)

    # 3) BPMs from Deezer (ISRC + search fallback)
    bpm_df = build_bpm_dataframe(track_ids, id2isrc, meta)

    log.info(f"Features rows: {len(bpm_df)}")
    print(bpm_df.head())

    # 4) Merge metadata for pretty CSV
    meta_df = (
        pd.DataFrame.from_dict(meta, orient="index")
        .reset_index()
        .rename(columns={"index": "track_id"})
    )  # cols: track_id, name, artists

    out = bpm_df.merge(meta_df, on="track_id", how="left")[[
        "track_id", "name", "artists", "isrc", "tempo", "tempo_source"
    ]].drop_duplicates("track_id")

    # 5) Write outputs
    out.to_csv("audio_features.csv", index=False)
    log.info(f"Wrote {len(out)} rows → audio_features.csv")

    have_bpm = set(out["track_id"])
    missing = [tid for tid in track_ids if tid not in have_bpm]
    if missing:
        pd.DataFrame({"track_id": missing}).to_csv("bpm_missing.csv", index=False)
        log.info(f"Wrote {len(missing)} unresolved IDs → bpm_missing.csv")

    # ---- NEW: pick closest songs to a heart rate ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr", type=float, help="Target heart rate in BPM")
    parser.add_argument("--k", type=int, default=10, help="How many songs to return")
    parser.add_argument("--no-multiples", action="store_true", help="Disable half/double-time matching")
    args, _ = parser.parse_known_args()

    target_hr = args.hr
    if target_hr is None:
        try:
            s = input("Enter heart rate BPM (or press Enter to skip): ").strip()
            target_hr = float(s) if s else None
        except Exception:
            target_hr = None

    if target_hr is not None:
        picks = closest_songs(
            out, target_bpm=target_hr, topk=args.k, consider_multiples=not args.no_multiples
        )
        if picks.empty:
            log.warning("No candidates found for the given heart rate.")
        else:
            picks.to_csv("closest_songs.csv", index=False)
            log.info(f"Top {len(picks)} closest → closest_songs.csv")
            # pretty print
            cols = ["name", "artists", "tempo", "matched_tempo", "multiplier", "delta"]
            print("\nClosest songs:")
            print(picks[cols].to_string(index=False, justify='left'))

    # also log unresolved IDs for later
    have_bpm = set(out["track_id"])
    missing = [tid for tid in track_ids if tid not in have_bpm]
    if missing:
        pd.DataFrame({"track_id": missing}).to_csv("bpm_missing.csv", index=False)
        log.info(f"Wrote {len(missing)} unresolved IDs → bpm_missing.csv")

    # 6) Optional: write a tempo band if TARGET_BPM is set in .env
    target = os.environ.get("TARGET_BPM")
    if target:
        try:
            target_bpm = float(target)
            band = filter_by_tempo_band(out, target_bpm, pct=0.06)
            band.to_csv("tempo_candidates.csv", index=False)
            log.info(f"Target {target_bpm:.1f} BPM → {len(band)} candidates → tempo_candidates.csv")
        except ValueError:
            log.warning("TARGET_BPM in .env is not a float; skipping tempo band file.")

if __name__ == "__main__":
    main()