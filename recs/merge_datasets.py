#!/usr/bin/env python3
"""
Merge multiple Spotify datasets for training.

Can combine:
1. Current enriched_kaggle.csv (~113k tracks)
2. Spotify 12M songs dataset (much larger)

This will significantly improve model performance and lookup coverage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]


def load_kaggle_12m(csv_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load Spotify 12M songs dataset.

    Expected columns (based on typical Spotify datasets):
    - track_id, name, artists, album_name
    - tempo, energy, danceability, valence, loudness
    - popularity, release_date, duration_ms, explicit
    - track_genre (or genres)
    """
    print(f"Loading Spotify 12M dataset from {csv_path}...")

    # Try to read in chunks if it's very large
    chunks = []
    chunk_size = 100000

    try:
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
            if limit and len(chunks) * chunk_size >= limit:
                remaining = limit - len(chunks) * chunk_size
                chunk = chunk.head(remaining)
                chunks.append(chunk)
                break
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                print(f"  Loaded {len(chunks) * chunk_size:,} rows...")

        df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying to load without chunks...")
        df = pd.read_csv(csv_path, nrows=limit, low_memory=False)

    print(f"Loaded {len(df):,} tracks")

    # Standardize column names
    column_mapping = {
        'id': 'track_id',
        'track_id': 'track_id',
        'track_name': 'name',
        'name': 'name',
        'artist_name': 'artists',
        'artists': 'artists',
        'album_name': 'album_name',
        'album': 'album_name',
        'track_genre': 'genres',
        'genre': 'genres',
        'genres': 'genres',
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})

    # Ensure required columns exist
    required = ['track_id'] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"WARNING: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    # Filter valid tracks
    print("Filtering valid tracks...")
    initial_count = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    df = df[df["tempo"] > 0]
    print(f"Valid tracks: {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")

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

    return df


def merge_datasets(
    datasets: List[pd.DataFrame],
    dedupe_by: str = "track_id"
) -> pd.DataFrame:
    """
    Merge multiple datasets, removing duplicates.
    """
    if not datasets:
        return pd.DataFrame()

    print(f"\nMerging {len(datasets)} datasets...")

    # Combine all datasets
    combined = pd.concat(datasets, ignore_index=True)
    print(f"Combined: {len(combined):,} tracks")

    # Remove duplicates (keep first occurrence)
    if dedupe_by in combined.columns:
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=[dedupe_by], keep='first')
        print(f"After deduplication: {len(combined):,} tracks (removed {before_dedup - len(combined):,} duplicates)")

    return combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge multiple Spotify datasets for training"
    )
    parser.add_argument("--kaggle-12m", type=str, help="Path to Spotify 12M songs CSV")
    parser.add_argument("--current", type=str, default="enriched_kaggle.csv",
                       help="Path to current enriched dataset")
    parser.add_argument("--output", type=str, default="merged_spotify_dataset.csv",
                       help="Output CSV path")
    parser.add_argument("--limit-12m", type=int, default=None,
                       help="Limit 12M dataset to N tracks (for testing)")
    parser.add_argument("--sample", type=int, default=None,
                       help="Random sample final dataset to N tracks")
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

    # Load 12M dataset if provided
    if args.kaggle_12m:
        kaggle_path = Path(args.kaggle_12m)
        if kaggle_path.exists():
            df_12m = load_kaggle_12m(kaggle_path, limit=args.limit_12m)
            if not df_12m.empty:
                datasets.append(df_12m)
        else:
            print(f"ERROR: 12M dataset not found: {kaggle_path}")
            return 1

    if not datasets:
        print("ERROR: No datasets to merge")
        return 1

    # Merge datasets
    merged = merge_datasets(datasets)

    if merged.empty:
        print("ERROR: Merged dataset is empty")
        return 1

    # Random sample if requested
    if args.sample and len(merged) > args.sample:
        print(f"\nSampling {args.sample:,} tracks randomly...")
        merged = merged.sample(n=args.sample, random_state=42).reset_index(drop=True)

    # Save merged dataset
    output_path = Path(args.output)
    print(f"\nSaving merged dataset to {output_path}...")
    merged.to_csv(output_path, index=False)

    print(f"\n✅ Merged dataset saved!")
    print(f"   Total tracks: {len(merged):,}")
    print(f"   Columns: {len(merged.columns)}")
    print(f"   Features: {', '.join(FEATURE_COLS)}")

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
