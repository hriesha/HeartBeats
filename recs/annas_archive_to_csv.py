#!/usr/bin/env python3
"""
Export Anna's Archive SQLite (track_audio_features) to CSV.

Reads in batches to handle large databases. Output has columns needed for
training: track_id, tempo, energy, danceability, valence, loudness (plus extras).

Usage:
  python -m recs.annas_archive_to_csv --output annas_archive_export.csv
  python -m recs.annas_archive_to_csv --output export.csv --limit 500000  # first 500k rows
  python -m recs.annas_archive_to_csv --db /path/to/spotify_clean_audio_features.sqlite3 --output out.csv
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

DB_PATHS = [
    "annas_archive_data/spotify_clean_audio_features.sqlite3",
    os.path.expanduser("~/Downloads/annas_archive_spotify_2025_07_metadata/spotify_clean_audio_features.sqlite3"),
    os.path.join(os.environ.get("ANNAS_ARCHIVE_DIR", "."), "spotify_clean_audio_features.sqlite3"),
]

BATCH_SIZE = 500_000
REQUIRED_COLS = ["track_id", "tempo", "energy", "danceability", "valence", "loudness"]


def find_db(db_path: str | None) -> Path:
    if db_path:
        p = Path(db_path).expanduser()
        if p.exists():
            return p
    root = Path(__file__).resolve().parent.parent
    for p in DB_PATHS:
        full = Path(p).expanduser() if str(p).startswith("~") else root / p
        if full.exists():
            return full
    raise FileNotFoundError(
        "Anna's Archive DB not found. Set --db /path/to/spotify_clean_audio_features.sqlite3"
    )


def export_to_csv(
    db_path: Path,
    output_path: Path,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> int:
    conn = sqlite3.connect(str(db_path))
    where = "WHERE null_response = 0 AND tempo IS NOT NULL AND tempo > 0"
    limit_clause = f"LIMIT {limit}" if limit else ""
    offset = 0
    total = 0
    first = True

    while True:
        if limit and offset >= limit:
            break
        fetch = min(batch_size, limit - offset) if limit else batch_size
        query = f"""
            SELECT track_id, tempo, energy, danceability, valence, loudness,
                   duration_ms, key, mode, time_signature,
                   acousticness, instrumentalness, speechiness, liveness
            FROM track_audio_features
            {where}
            LIMIT {fetch} OFFSET {offset}
        """
        df = pd.read_sql_query(query, conn)
        if df.empty:
            break
        df.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
        first = False
        total += len(df)
        print(f"  Wrote {total:,} rows...")
        if limit and total >= limit:
            break
        offset += len(df)
        if len(df) < batch_size:
            break

    conn.close()
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Anna's Archive SQLite to CSV")
    parser.add_argument("--db", help="Path to spotify_clean_audio_features.sqlite3")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to export")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Rows per batch")
    args = parser.parse_args()

    db_path = find_db(args.db)
    print(f"Using DB: {db_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = export_to_csv(
        db_path,
        output_path,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(f"Done. Exported {total:,} rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
