#!/usr/bin/env python3
"""
Enrich Kaggle dataset with Spotify metadata (batch processing with resume capability).

Fetches metadata for Kaggle tracks in small batches to avoid overloading:
- Track name, artist, album
- Genres (from artist)
- Related artists
- Popularity
- Release date
- etc.

Usage:
  python -m recs.enrich_kaggle_metadata --csv /path/to/kaggle.csv --output enriched.csv --batch-size 50 --delay 0.5
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    from spotipy.exceptions import SpotifyException
except ImportError:
    print("ERROR: spotipy not installed. Run: pip install spotipy")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os

# Feature columns we need from Kaggle
FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]


def get_spotify_client():
    """Get Spotify client using client credentials (no user auth needed for public data)."""
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be set in .env file"
        )

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def fetch_track_metadata_batch(
    sp: spotipy.Spotify,
    track_ids: List[str],
) -> Dict[str, Dict]:
    """
    Fetch metadata for a batch of tracks (max 50).

    Returns dict[track_id] -> metadata dict with:
    - name, artists (list), artist_names (str)
    - album_name, album_id
    - popularity
    - release_date
    - genres (from artist)
    - related_artists (from first artist)
    """
    if len(track_ids) > 50:
        raise ValueError("Batch size cannot exceed 50")

    metadata = {}

    try:
        # Fetch tracks (batch endpoint, up to 50)
        tracks_resp = sp.tracks(track_ids)
        tracks = tracks_resp.get("tracks", [])

        for track in tracks:
            if not track or not track.get("id"):
                continue

            track_id = track["id"]
            artists = track.get("artists", [])
            artist_ids = [a.get("id") for a in artists if a.get("id")]
            main_artist_id = artist_ids[0] if artist_ids else None

            # Get artist genres and related artists
            genres = []
            related_artist_ids = []
            if main_artist_id:
                try:
                    artist_info = sp.artist(main_artist_id)
                    genres = artist_info.get("genres", [])

                    # Get related artists (from first artist)
                    related = sp.artist_related_artists(main_artist_id)
                    related_artist_ids = [a.get("id") for a in related.get("artists", [])[:5]]  # Top 5
                except Exception as e:
                    print(f"  Warning: Could not fetch artist info for {main_artist_id}: {e}")

            metadata[track_id] = {
                "track_id": track_id,
                "name": track.get("name", ""),
                "artists": [a.get("name", "") for a in artists],
                "artist_names": ", ".join([a.get("name", "") for a in artists]),
                "artist_ids": artist_ids,
                "main_artist_id": main_artist_id,
                "album_name": track.get("album", {}).get("name", ""),
                "album_id": track.get("album", {}).get("id", ""),
                "popularity": track.get("popularity", 0),
                "release_date": track.get("album", {}).get("release_date", ""),
                "duration_ms": track.get("duration_ms", 0),
                "explicit": track.get("explicit", False),
                "genres": genres,
                "related_artist_ids": related_artist_ids,
            }

    except SpotifyException as e:
        print(f"  Spotify API error: {e.http_status} - {e.msg}")
        if e.http_status == 429:  # Rate limited
            retry_after = int(e.headers.get("Retry-After", 60))
            print(f"  Rate limited! Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            # Retry once
            return fetch_track_metadata_batch(sp, track_ids)
    except Exception as e:
        print(f"  Error fetching batch: {e}")

    return metadata


def load_progress(db_path: Path) -> set:
    """Load already-processed track IDs from progress DB."""
    if not db_path.exists():
        return set()

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT track_id FROM progress").fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def save_progress(db_path: Path, track_id: str, metadata: Dict):
    """Save progress for a track."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS progress (
                track_id TEXT PRIMARY KEY,
                metadata_json TEXT
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO progress (track_id, metadata_json) VALUES (?, ?)",
            (track_id, json.dumps(metadata))
        )
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enrich Kaggle dataset with Spotify metadata (batch processing)"
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to Kaggle CSV")
    parser.add_argument("--output", type=str, default="enriched_kaggle.csv", help="Output CSV path")
    parser.add_argument("--progress-db", type=str, default="metadata_progress.db", help="Progress DB path")
    parser.add_argument("--batch-size", type=int, default=50, help="Tracks per batch (max 50)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between batches (seconds)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks to process (for testing)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    print("Loading Kaggle dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tracks")

    # Filter to tracks with required features
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"ERROR: Missing required column: {col}")
            return 1

    df = df.dropna(subset=FEATURE_COLS)
    df = df[df["tempo"] > 0]  # Filter invalid tempo
    print(f"Valid tracks: {len(df)}")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} tracks (testing mode)")

    # Load progress
    progress_db = Path(args.progress_db)
    processed = load_progress(progress_db)
    print(f"Already processed: {len(processed)} tracks")

    # Get tracks to process
    track_ids = df["track_id"].astype(str).tolist()
    to_process = [tid for tid in track_ids if tid not in processed]
    print(f"Remaining to process: {len(to_process)} tracks")

    if not to_process:
        print("All tracks already processed!")
        return 0

    # Initialize Spotify client
    print("\nConnecting to Spotify API...")
    try:
        sp = get_spotify_client()
        print("✅ Connected (using client credentials)")
    except Exception as e:
        print(f"ERROR: Failed to connect to Spotify: {e}")
        return 1

    # Process in batches
    batch_size = min(args.batch_size, 50)  # Spotify limit
    total_batches = (len(to_process) + batch_size - 1) // batch_size

    print(f"\nProcessing {len(to_process)} tracks in {total_batches} batches...")
    print(f"Batch size: {batch_size}, Delay: {args.delay}s\n")

    all_metadata = {}
    start_time = time.time()

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(to_process))
        batch_ids = to_process[start_idx:end_idx]

        print(f"Batch {batch_idx + 1}/{total_batches}: {len(batch_ids)} tracks...", end=" ", flush=True)

        try:
            batch_metadata = fetch_track_metadata_batch(sp, batch_ids)
            all_metadata.update(batch_metadata)

            # Save progress
            for tid, meta in batch_metadata.items():
                save_progress(progress_db, tid, meta)

            processed_count = len(processed) + len(all_metadata)
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            remaining = len(to_process) - len(all_metadata)
            eta = remaining / rate if rate > 0 else 0

            print(f"✅ ({len(batch_metadata)} fetched, {processed_count}/{len(track_ids)} total, "
                  f"ETA: {eta/60:.1f} min)")

            # Delay between batches (except last)
            if batch_idx < total_batches - 1:
                time.sleep(args.delay)

        except KeyboardInterrupt:
            print("\n\nInterrupted! Progress saved. Resume by running the same command.")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Merge metadata with Kaggle data
    print(f"\nMerging metadata with Kaggle data...")
    metadata_df = pd.DataFrame(list(all_metadata.values()))

    if metadata_df.empty:
        print("No metadata fetched. Check your Spotify credentials and API access.")
        return 1

    # Merge on track_id
    enriched = df.merge(metadata_df, on="track_id", how="left")

    # Save enriched dataset
    output_path = Path(args.output)
    enriched.to_csv(output_path, index=False)
    print(f"✅ Saved enriched dataset to: {output_path}")
    print(f"   Total tracks: {len(enriched)}")
    print(f"   With metadata: {enriched['name'].notna().sum()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
