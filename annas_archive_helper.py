"""
Helper module to access Anna's Archive Spotify backup data.

This module provides functions to:
1. Download/access the metadata database from Anna's Archive
2. Look up audio features for track IDs
3. Fall back gracefully if data isn't available

The Anna's Archive Spotify backup includes:
- SQLite database: spotify_clean_track_files.sqlite3 (track metadata)
- JSON files: spotify_audio_features.jsonl.zst (audio features)
- See: https://annas-archive.li/blog/backing-up-spotify.html
"""

import os
import logging
import sqlite3
import json
from typing import List, Optional
import pandas as pd

log = logging.getLogger("annas_archive")

# Path to Anna's Archive data directory
ARCHIVE_DATA_DIR = os.environ.get("ANNAS_ARCHIVE_DIR", "./annas_archive_data")

# Expected database names (compressed and uncompressed)
DB_NAME = "spotify_clean_track_files.sqlite3"
AUDIO_FEATURES_DB = "spotify_clean_audio_features.sqlite3"
AUDIO_FEATURES_DB_ZST = "spotify_clean_audio_features.sqlite3.zst"
FEATURES_JSONL = "spotify_audio_features.jsonl.zst"


def get_archive_db_path() -> Optional[str]:
    """Get path to the SQLite database if it exists."""
    # Try audio features database first (what we actually need)
    audio_features_path = os.path.join(ARCHIVE_DATA_DIR, AUDIO_FEATURES_DB)
    if os.path.exists(audio_features_path):
        return audio_features_path

    # Fallback to track files database
    db_path = os.path.join(ARCHIVE_DATA_DIR, DB_NAME)
    if os.path.exists(db_path):
        return db_path

    # Check for compressed version that needs extraction
    zst_path = os.path.join(ARCHIVE_DATA_DIR, AUDIO_FEATURES_DB_ZST)
    if os.path.exists(zst_path):
        log.warning("Found compressed database at %s. You need to extract it first.", zst_path)
        log.warning("Install zstandard: pip install zstandard")
        log.warning("Then extract: zstd -d %s", zst_path)

    return None


def get_archive_features_path() -> Optional[str]:
    """Get path to the audio features JSONL file if it exists."""
    features_path = os.path.join(ARCHIVE_DATA_DIR, FEATURES_JSONL)
    if os.path.exists(features_path):
        return features_path
    return None


def load_features_from_db(track_ids: List[str]) -> pd.DataFrame:
    """
    Load audio features from Anna's Archive SQLite database.

    Note: This requires the database to be downloaded and extracted.
    The database structure may vary - this is a placeholder implementation.
    """
    db_path = get_archive_db_path()
    if not db_path:
        log.warning("Anna's Archive database not found at %s", ARCHIVE_DATA_DIR)
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)

        # First, try to understand the schema
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        log.info("Found tables in database: %s", tables)

        # Use the actual schema: table is 'track_audio_features'
        placeholders = ','.join(['?' for _ in track_ids])

        # Query the actual table structure - get all available audio features
        query = f"""
            SELECT track_id, tempo, energy, danceability, valence, loudness,
                   acousticness, instrumentalness, speechiness, liveness,
                   key, mode, time_signature, duration_ms
            FROM track_audio_features
            WHERE track_id IN ({placeholders})
        """

        try:
            df = pd.read_sql_query(query, conn, params=track_ids)
            log.info("Successfully queried track_audio_features table")
        except Exception as query_error:
            log.error("Query failed: %r", query_error)
            df = pd.DataFrame()

        conn.close()

        if df.empty:
            log.warning("Could not find matching schema. Available tables: %s", tables)
            log.warning("You may need to check the database schema and update the queries")

        return df
    except Exception as e:
        log.error("Error reading from Anna's Archive database: %r", e)
        import traceback
        log.error(traceback.format_exc())
        return pd.DataFrame()


def load_features_from_jsonl(track_ids: List[str]) -> pd.DataFrame:
    """
    Load audio features from Anna's Archive JSONL file.

    This is a more efficient approach if you have the JSONL file extracted.
    """
    features_path = get_archive_features_path()
    if not features_path:
        log.warning("Anna's Archive features file not found at %s", ARCHIVE_DATA_DIR)
        return pd.DataFrame()

    track_ids_set = set(track_ids)
    features_list = []

    try:
        # Handle .zst compressed files (requires zstandard library)
        if features_path.endswith('.zst'):
            try:
                import zstandard as zstd
                with zstd.open(features_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        if len(features_list) >= len(track_ids):
                            break
                        try:
                            data = json.loads(line)
                            track_id = data.get('id')
                            if track_id in track_ids_set:
                                features_list.append({
                                    'track_id': track_id,
                                    'tempo': data.get('tempo'),
                                    'energy': data.get('energy'),
                                    'danceability': data.get('danceability'),
                                    'valence': data.get('valence'),
                                    'loudness': data.get('loudness'),
                                })
                        except json.JSONDecodeError:
                            continue
            except ImportError:
                log.error("zstandard library not installed. Install with: pip install zstandard")
                return pd.DataFrame()
        else:
            # Regular JSONL file
            with open(features_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(features_list) >= len(track_ids):
                        break
                    try:
                        data = json.loads(line)
                        track_id = data.get('id')
                        if track_id in track_ids_set:
                            features_list.append({
                                'track_id': track_id,
                                'tempo': data.get('tempo'),
                                'energy': data.get('energy'),
                                'danceability': data.get('danceability'),
                                'valence': data.get('valence'),
                                'loudness': data.get('loudness'),
                            })
                    except json.JSONDecodeError:
                        continue

        return pd.DataFrame(features_list)
    except Exception as e:
        log.error("Error reading from Anna's Archive features file: %r", e)
        return pd.DataFrame()


def get_audio_features(track_ids: List[str]) -> pd.DataFrame:
    """
    Get audio features for track IDs from Anna's Archive data.

    Tries multiple methods:
    1. SQLite database
    2. JSONL file
    3. Returns empty DataFrame if neither is available
    """
    if not track_ids:
        return pd.DataFrame()

    # Try database first (usually faster)
    df = load_features_from_db(track_ids)
    if not df.empty:
        log.info("Loaded %d features from Anna's Archive database", len(df))
        return df

    # Try JSONL file
    df = load_features_from_jsonl(track_ids)
    if not df.empty:
        log.info("Loaded %d features from Anna's Archive JSONL", len(df))
        return df

    log.warning("Could not load features from Anna's Archive")
    return pd.DataFrame()


def is_archive_available() -> bool:
    """Check if Anna's Archive data is available."""
    return get_archive_db_path() is not None or get_archive_features_path() is not None
