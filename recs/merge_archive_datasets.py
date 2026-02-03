#!/usr/bin/env python3
"""
Merge the archive datasets (high/low popularity) with current dataset.

These datasets are smaller (~4.8k tracks) but have good metadata including:
- playlist_genre, playlist_subgenre
- track_popularity
- All audio features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]


def load_archive_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and standardize archive dataset."""
    print(f"Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} tracks")

    # Standardize column names
    column_mapping = {
        'track_id': 'track_id',
        'id': 'track_id',
        'track_name': 'name',
        'name': 'name',
        'track_artist': 'artists',
        'artists': 'artists',
        'track_album_name': 'album_name',
        'album_name': 'album_name',
        'playlist_genre': 'genres',  # Use playlist_genre as genres
        'genres': 'genres',
        'track_popularity': 'popularity',
        'popularity': 'popularity',
        'track_album_release_date': 'release_date',
        'release_date': 'release_date',
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})

    # Ensure required columns exist
    required = ['track_id'] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")
        return pd.DataFrame()

    # Filter valid tracks
    initial_count = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    df = df[df["tempo"] > 0]
    print(f"  Valid tracks: {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")

    # Add missing columns with defaults
    if "popularity" not in df.columns:
        df["popularity"] = 50
    if "release_date" not in df.columns:
        df["release_date"] = ""
    if "duration_ms" not in df.columns:
        df["duration_ms"] = 180000
    if "explicit" not in df.columns:
        df["explicit"] = False
    if "genres" not in df.columns:
        df["genres"] = ""
    if "artist_names" not in df.columns:
        df["artist_names"] = df.get("artists", "")
    if "main_artist_id" not in df.columns:
        df["main_artist_id"] = None
    if "related_artist_ids" not in df.columns:
        df["related_artist_ids"] = ""
    if "name" not in df.columns:
        df["name"] = df.get("track_name", "")
    if "album_name" not in df.columns:
        df["album_name"] = ""

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge archive datasets (high/low popularity) with current dataset"
    )
    parser.add_argument("--high-pop", type=str,
                       default="/Users/saachidhamija/Downloads/archive (3)/high_popularity_spotify_data.csv",
                       help="Path to high popularity CSV")
    parser.add_argument("--low-pop", type=str,
                       default="/Users/saachidhamija/Downloads/archive (3)/low_popularity_spotify_data.csv",
                       help="Path to low popularity CSV")
    parser.add_argument("--current", type=str, default="enriched_kaggle.csv",
                       help="Path to current enriched dataset")
    parser.add_argument("--output", type=str, default="merged_with_archive.csv",
                       help="Output CSV path")
    args = parser.parse_args()

    datasets = []

    # Load current dataset
    current_path = Path(args.current)
    if current_path.exists():
        print(f"Loading current dataset: {current_path}")
        df_current = pd.read_csv(current_path)
        print(f"  Loaded {len(df_current):,} tracks")
        datasets.append(df_current)
    else:
        print(f"WARNING: Current dataset not found: {current_path}")

    # Load high popularity
    high_path = Path(args.high_pop)
    if high_path.exists():
        df_high = load_archive_dataset(high_path)
        if not df_high.empty:
            datasets.append(df_high)
    else:
        print(f"WARNING: High popularity dataset not found: {high_path}")

    # Load low popularity
    low_path = Path(args.low_pop)
    if low_path.exists():
        df_low = load_archive_dataset(low_path)
        if not df_low.empty:
            datasets.append(df_low)
    else:
        print(f"WARNING: Low popularity dataset not found: {low_path}")

    if not datasets:
        print("ERROR: No datasets to merge")
        return 1

    # Merge datasets
    print(f"\nMerging {len(datasets)} datasets...")
    merged = pd.concat(datasets, ignore_index=True)
    print(f"Combined: {len(merged):,} tracks")

    # Remove duplicates (keep first occurrence)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["track_id"], keep='first')
    print(f"After deduplication: {len(merged):,} tracks (removed {before_dedup - len(merged):,} duplicates)")

    # Save merged dataset
    output_path = Path(args.output).resolve()
    print(f"\nSaving merged dataset to {output_path}...")

    # Try to create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        merged.to_csv(output_path, index=False)
    except PermissionError:
        # Try saving to Desktop if current location fails
        desktop_path = Path.home() / "Desktop" / output_path.name
        print(f"Permission denied. Trying Desktop: {desktop_path}")
        merged.to_csv(desktop_path, index=False)
        output_path = desktop_path

    print(f"\n✅ Merged dataset saved!")
    print(f"   Total tracks: {len(merged):,}")
    print(f"   Columns: {len(merged.columns)}")

    # Show statistics
    print(f"\nDataset statistics:")
    print(f"   Tracks with tempo: {merged['tempo'].notna().sum():,}")
    print(f"   Tracks with genres: {merged['genres'].notna().sum():,}")
    print(f"   Unique genres: {merged['genres'].nunique()}")
    print(f"   Tempo range: {merged['tempo'].min():.1f} - {merged['tempo'].max():.1f} BPM")
    print(f"   Mean tempo: {merged['tempo'].mean():.1f} BPM")

    return 0


if __name__ == "__main__":
    sys.exit(main())
