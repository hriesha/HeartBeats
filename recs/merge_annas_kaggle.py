#!/usr/bin/env python3
"""
Merge Anna's Archive sample with Kaggle tracks_features dataset.

Usage:
  python -m recs.merge_annas_kaggle --annas annas_archive_data/annas_archive_sample.csv \
    --kaggle "/Users/saachidhamija/Downloads/tracks_features 2.csv" \
    --output recs/model/merged_training.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across datasets."""
    if "id" in df.columns and "track_id" not in df.columns:
        df = df.rename(columns={"id": "track_id"})
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Anna's Archive + Kaggle datasets")
    parser.add_argument("--annas", required=True, help="Anna's Archive CSV")
    parser.add_argument("--kaggle", required=True, help="Kaggle tracks_features CSV")
    parser.add_argument("--output", required=True, help="Output merged CSV")
    args = parser.parse_args()

    annas_path = Path(args.annas)
    kaggle_path = Path(args.kaggle)
    out_path = Path(args.output)

    if not annas_path.exists():
        print(f"ERROR: Anna's Archive CSV not found: {annas_path}")
        return 1
    if not kaggle_path.exists():
        print(f"ERROR: Kaggle CSV not found: {kaggle_path}")
        return 1

    print(f"Loading Anna's Archive: {annas_path}")
    df_annas = pd.read_csv(annas_path)
    df_annas = standardize_columns(df_annas)
    print(f"  Loaded {len(df_annas):,} tracks")

    print(f"Loading Kaggle: {kaggle_path}")
    df_kaggle = pd.read_csv(kaggle_path)
    df_kaggle = standardize_columns(df_kaggle)
    print(f"  Loaded {len(df_kaggle):,} tracks")

    # Ensure required columns
    missing_annas = [c for c in FEATURE_COLS if c not in df_annas.columns]
    missing_kaggle = [c for c in FEATURE_COLS if c not in df_kaggle.columns]
    if missing_annas:
        print(f"ERROR: Anna's Archive missing: {missing_annas}")
        return 1
    if missing_kaggle:
        print(f"ERROR: Kaggle missing: {missing_kaggle}")
        return 1

    # Filter valid (tempo > 0, no NaN in features)
    print("Filtering valid tracks...")
    for df, name in [(df_annas, "Anna's"), (df_kaggle, "Kaggle")]:
        before = len(df)
        df = df.dropna(subset=FEATURE_COLS)
        df = df[df["tempo"] > 0]
        print(f"  {name}: {len(df):,} / {before:,} valid")
        if name == "Anna's":
            df_annas = df
        else:
            df_kaggle = df

    # Merge
    print(f"\nMerging datasets...")
    merged = pd.concat([df_annas, df_kaggle], ignore_index=True)
    print(f"  Combined: {len(merged):,} tracks")

    # Deduplicate by track_id
    if "track_id" in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["track_id"], keep="first")
        print(f"  After dedup: {len(merged):,} tracks (removed {before - len(merged):,} duplicates)")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\n✅ Saved to {out_path}")
    print(f"   Total: {len(merged):,} tracks")
    print(f"   Tempo range: {merged['tempo'].min():.1f} - {merged['tempo'].max():.1f} BPM")

    return 0


if __name__ == "__main__":
    sys.exit(main())
