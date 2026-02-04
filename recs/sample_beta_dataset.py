#!/usr/bin/env python3
"""
Sample a beta training dataset with BPM stratification + popularity weighting.

1. Reads source CSV in chunks (memory-efficient for large files)
2. Assigns BPM buckets: 60-80, 80-100, 100-120, 120-140, 140-160, 160-180
3. Samples ~25k tracks per bucket (or all if bucket has fewer)
4. Fetches popularity from Spotify API for sampled tracks
5. Within each bucket: keeps 60% top by popularity + 40% random
6. Writes final CSV for training

Usage:
  python -m recs.sample_beta_dataset --input path/to/source.csv --output recs/model/beta_training.csv
  python -m recs.sample_beta_dataset --input source.csv --output beta.csv --train  # also run training
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# BPM buckets (min inclusive, max exclusive except last)
BPM_BUCKETS = [
    (60, 80),    # Very slow
    (80, 100),   # Slow jog
    (100, 120),  # Moderate
    (120, 140),  # Running
    (140, 160),  # Fast run
    (160, 181),  # Very fast (181 to include 180)
]
TARGET_PER_BUCKET = 25_000
POPULARITY_TOP_PCT = 0.60
POPULARITY_RANDOM_PCT = 0.40
CHUNK_SIZE = 100_000

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]


def get_track_id_column(df: pd.DataFrame) -> str:
    if "track_id" in df.columns:
        return "track_id"
    if "id" in df.columns:
        return "id"
    raise ValueError("CSV must have 'track_id' or 'id' column")


def assign_bpm_bucket(tempo: float) -> int | None:
    """Return bucket index 0..5, or None if out of range."""
    if pd.isna(tempo) or tempo <= 0:
        return None
    for i, (lo, hi) in enumerate(BPM_BUCKETS):
        if lo <= tempo < hi:
            return i
    return None


def sample_from_chunked_csv(
    csv_path: Path,
    target_per_bucket: int = TARGET_PER_BUCKET,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """Read CSV in chunks, sample per BPM bucket."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input not found: {csv_path}")

    buckets: dict[int, list[pd.DataFrame]] = {i: [] for i in range(len(BPM_BUCKETS))}
    id_col = None

    print(f"Reading {csv_path} in chunks of {chunk_size}...")
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        if id_col is None:
            id_col = get_track_id_column(chunk)
        # Ensure tempo is numeric
        chunk["tempo"] = pd.to_numeric(chunk["tempo"], errors="coerce")
        chunk = chunk.dropna(subset=["tempo"])
        chunk = chunk[chunk["tempo"] > 0]

        chunk["_bucket"] = chunk["tempo"].apply(assign_bpm_bucket)
        chunk = chunk[chunk["_bucket"].notna()]
        chunk["_bucket"] = chunk["_bucket"].astype(int)

        for b in range(len(BPM_BUCKETS)):
            sub = chunk[chunk["_bucket"] == b]
            if len(sub) > 0:
                buckets[b].append(sub.drop(columns=["_bucket"]))

        if (chunk_idx + 1) % 10 == 0:
            print(f"  Processed {(chunk_idx + 1) * chunk_size:,} rows...")

    # Concat and sample per bucket
    samples = []
    for b in range(len(BPM_BUCKETS)):
        if not buckets[b]:
            continue
        combined = pd.concat(buckets[b], ignore_index=True)
        combined = combined.drop_duplicates(subset=[id_col], keep="first")
        n = len(combined)
        take = min(target_per_bucket, n)
        if take < n:
            combined = combined.sample(n=take, random_state=42)
        samples.append(combined)
        print(f"  Bucket {BPM_BUCKETS[b][0]}-{BPM_BUCKETS[b][1]} BPM: {n:,} available, sampled {take:,}")

    if not samples:
        raise ValueError("No valid tracks found in any BPM bucket")

    df = pd.concat(samples, ignore_index=True)
    df = df.drop_duplicates(subset=[id_col], keep="first")
    if id_col != "track_id":
        df = df.rename(columns={id_col: "track_id"})
    print(f"Total sampled: {len(df):,} tracks")
    return df


def fetch_popularity(track_ids: list[str], batch_size: int = 50, delay: float = 0.4) -> dict[str, int]:
    """Fetch popularity for track IDs from Spotify API. Returns track_id -> popularity."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
    except ImportError as e:
        print(f"Spotify fetch requires: pip install spotipy python-dotenv. {e}")
        return {tid: 0 for tid in track_ids}

    cid = os.getenv("SPOTIPY_CLIENT_ID")
    secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    if not cid or not secret:
        print("SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET not set. Using popularity=0 for all.")
        return {tid: 0 for tid in track_ids}

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
    result: dict[str, int] = {}

    print(f"Fetching popularity for {len(track_ids):,} tracks from Spotify API...")
    MAX_RETRIES = 1  # Don't retry too much to avoid flagging
    MAX_WAIT_SEC = 120  # Never wait longer than 2 min on rate limit
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i : i + batch_size]
        ok = False
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = sp.tracks(batch)
                for t in resp.get("tracks", []):
                    if t and t.get("id"):
                        result[t["id"]] = int(t.get("popularity", 0))
                ok = True
                break
            except Exception as e:
                err_str = str(e).lower()
                is_429 = "429" in err_str or "rate" in err_str or "limit" in err_str
                # Parse "Retry will occur after: 82019 s" - if > MAX_WAIT, skip and use 0
                retry_after = None
                if "retry" in err_str and "after" in err_str:
                    import re
                    m = re.search(r"after:\s*(\d+)\s*s", str(e), re.I)
                    if m:
                        retry_after = int(m.group(1))
                if is_429 and retry_after is not None and retry_after > MAX_WAIT_SEC:
                    print(f"  Rate limited: Spotify says retry after {retry_after}s. Skipping remaining API calls, using popularity=0.")
                    for tid in track_ids[i:]:
                        if tid not in result:
                            result[tid] = 0
                    # Break out of outer loop
                    for tid in track_ids:
                        if tid not in result:
                            result[tid] = 0
                    return result
                if is_429 and attempt < MAX_RETRIES:
                    wait = min(60, MAX_WAIT_SEC)
                    print(f"  Rate limited at batch {i}, backing off {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  API error at batch {i}: {e}")
                    for tid in batch:
                        if tid not in result:
                            result[tid] = 0
                    break
        time.sleep(delay)
        if (i + batch_size) % 5000 < batch_size:
            print(f"  Fetched {min(i + batch_size, len(track_ids)):,} / {len(track_ids):,}")

    for tid in track_ids:
        if tid not in result:
            result[tid] = 0
    return result


def apply_popularity_selection(
    df: pd.DataFrame,
    popularity_map: dict[str, int],
    top_pct: float = POPULARITY_TOP_PCT,
    random_pct: float = POPULARITY_RANDOM_PCT,
) -> pd.DataFrame:
    """Within each BPM bucket: keep top_pct by popularity + random_pct random."""
    df = df.copy()
    df["popularity"] = df["track_id"].map(popularity_map).fillna(0).astype(int)
    df["_bucket"] = df["tempo"].apply(assign_bpm_bucket)
    df = df[df["_bucket"].notna()]
    df["_bucket"] = df["_bucket"].astype(int)

    kept = []
    for b in range(len(BPM_BUCKETS)):
        sub = df[df["_bucket"] == b].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("popularity", ascending=False).reset_index(drop=True)
        n = len(sub)
        n_top = max(1, int(n * top_pct))
        n_rand = max(0, int(n * random_pct))
        n_rand = min(n_rand, n - n_top)

        top = sub.iloc[:n_top]
        rest = sub.iloc[n_top:]
        if n_rand > 0 and len(rest) > 0:
            rand_idx = rest.sample(n=min(n_rand, len(rest)), random_state=42).index
            rand = sub.loc[rand_idx]
            combined = pd.concat([top, rand]).drop_duplicates(subset=["track_id"], keep="first")
        else:
            combined = top

        kept.append(combined.drop(columns=["_bucket", "popularity"]))

    out = pd.concat(kept, ignore_index=True)
    out = out.drop_duplicates(subset=["track_id"], keep="first")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample beta dataset: BPM buckets + popularity weighting"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to source CSV (can be large)")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path (default: recs/model/beta_training.csv)")
    parser.add_argument("--target", type=int, default=TARGET_PER_BUCKET, help=f"Target per bucket (default {TARGET_PER_BUCKET})")
    parser.add_argument("--no-api", action="store_true", help="Skip Spotify API; use popularity=0 (random selection only)")
    parser.add_argument("--train", action="store_true", help="Run training on output CSV after sampling")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Chunk size for reading (default {CHUNK_SIZE})")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else Path(__file__).parent / "model" / "beta_training.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Sample from source
    df = sample_from_chunked_csv(
        args.input,
        target_per_bucket=args.target,
        chunk_size=args.chunk_size,
    )

    # 2. Fetch popularity
    track_ids = df["track_id"].astype(str).tolist()
    if args.no_api:
        popularity_map = {tid: 0 for tid in track_ids}
    else:
        popularity_map = fetch_popularity(track_ids)

    # 3. Apply 60/40 selection
    df_final = apply_popularity_selection(df, popularity_map)
    print(f"Final dataset: {len(df_final):,} tracks")

    # 4. Ensure required columns for training
    missing = [c for c in FEATURE_COLS if c not in df_final.columns]
    if missing:
        print(f"ERROR: Output missing columns: {missing}")
        return 1

    df_final.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    # 5. Optional: train
    if args.train:
        print("\n--- Training model ---")
        import subprocess
        r = subprocess.run(
            [
                sys.executable, "-m", "recs.train",
                "--csv", str(out_path),
                "--sample", "0",
                "--clusters", "6",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
        )
        return r.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
