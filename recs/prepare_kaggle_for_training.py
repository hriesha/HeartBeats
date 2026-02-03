#!/usr/bin/env python3
"""
Prepare Kaggle dataset for metadata model training (no API calls needed).

The Kaggle dataset already has metadata, so we just need to format it correctly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Kaggle dataset for metadata training")
    parser.add_argument("--csv", type=str, required=True, help="Path to Kaggle CSV")
    parser.add_argument("--output", type=str, default="enriched_kaggle.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    print("Loading Kaggle dataset...")
    df = pd.read_csv(csv_path)

    # Check required columns
    required = FEATURE_COLS + ["track_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return 1

    # Filter valid tracks
    df = df.dropna(subset=FEATURE_COLS)
    df = df[df["tempo"] > 0]
    print(f"Valid tracks: {len(df)}")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} tracks (testing mode)")

    # Format metadata columns to match expected format
    enriched = df.copy()

    # Map existing columns
    if "track_name" in enriched.columns:
        enriched["name"] = enriched["track_name"]
    elif "name" not in enriched.columns:
        enriched["name"] = ""

    if "artists" in enriched.columns:
        enriched["artist_names"] = enriched["artists"]
        # Extract first artist as main artist (simplified)
        enriched["main_artist_id"] = None  # Kaggle doesn't have artist IDs
    else:
        enriched["artist_names"] = ""
        enriched["main_artist_id"] = None

    if "album_name" in enriched.columns:
        enriched["album_name"] = enriched["album_name"]
    else:
        enriched["album_name"] = ""

    # Genres: Kaggle has track_genre (single genre), convert to list format
    if "track_genre" in enriched.columns:
        enriched["genres"] = enriched["track_genre"].fillna("").astype(str)
    else:
        enriched["genres"] = ""

    # Related artists: not available in Kaggle, leave empty
    enriched["related_artist_ids"] = ""

    # Ensure we have all expected columns
    expected_meta_cols = [
        "track_id", "name", "artists", "artist_names", "album_name",
        "popularity", "release_date", "duration_ms", "explicit",
        "genres", "main_artist_id", "related_artist_ids",
    ]

    # Add missing columns with defaults
    if "release_date" not in enriched.columns:
        enriched["release_date"] = ""  # Kaggle doesn't have this

    # Reorder columns: audio features first, then metadata
    output_cols = (
        ["track_id"] +
        FEATURE_COLS +
        [c for c in expected_meta_cols if c not in ["track_id"] + FEATURE_COLS]
    )

    # Only include columns that exist
    output_cols = [c for c in output_cols if c in enriched.columns]
    enriched = enriched[output_cols]

    # Save
    output_path = Path(args.output)
    enriched.to_csv(output_path, index=False)
    print(f"\n✅ Saved enriched dataset to: {output_path}")
    print(f"   Total tracks: {len(enriched)}")
    print(f"   Columns: {len(enriched.columns)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
