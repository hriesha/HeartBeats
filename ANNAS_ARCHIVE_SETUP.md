# Anna's Archive Spotify Data Setup

This project uses [Anna's Archive Spotify backup](https://annas-archive.li/blog/backing-up-spotify.html) to avoid Spotify API rate limits and 403 errors when fetching audio features.

## What is Anna's Archive?

Anna's Archive has backed up Spotify's metadata database containing:
- **256 million tracks** with metadata
- Audio features (tempo, energy, danceability, valence, loudness)
- Track information (name, artists, ISRC, etc.)
- ~300TB total (but we only need the metadata, not the audio files)

## Setup Instructions

### Option 1: Download Metadata Only (Recommended)

The metadata is much smaller than the full archive. You need:

1. **Download the metadata torrent** from Anna's Archive:
   - Go to: https://annas-archive.li/torrents
   - Look for: `annas_archive_spotify_2025_07_metadata.torrent`
   - This contains the SQLite database and JSON files

2. **Extract the data**:
   ```bash
   # Create data directory
   mkdir -p annas_archive_data

   # Extract the torrent contents to annas_archive_data/
   # You should have files like:
   # - spotify_clean_track_files.sqlite3 (SQLite database)
   # - spotify_audio_features.jsonl.zst (compressed JSONL file)
   ```

3. **Set environment variable** (optional):
   ```bash
   export ANNAS_ARCHIVE_DIR="./annas_archive_data"
   ```

### Option 2: Use Existing CSV/Data

If you already have audio features in CSV format:
- Place it in the project root or update the path in code
- The format should include: `track_id`, `tempo`, `energy`, `danceability`, `valence`, `loudness`

### Option 3: Fallback to Spotify API

If Anna's Archive data is not available, the system will automatically fall back to the Spotify API. However, you may encounter:
- Rate limiting (429 errors)
- 403 Forbidden errors (if scopes are missing)
- Slower performance

## Database Schema (Expected)

The SQLite database should have a `tracks` table with columns like:
- `track_id` (Spotify track ID)
- `tempo`
- `energy`
- `danceability`
- `valence`
- `loudness`
- `name` (optional)
- `artists` (optional)

**Note**: The actual schema may vary. You may need to adjust the queries in `annas_archive_helper.py` based on the actual database structure from Anna's Archive.

## Troubleshooting

### "Anna's Archive database not found"
- Make sure you've downloaded and extracted the metadata torrent
- Check that `annas_archive_data/spotify_clean_track_files.sqlite3` exists
- Or verify the path set in `ANNAS_ARCHIVE_DIR` environment variable

### "zstandard library not installed"
- Install it: `pip install zstandard`
- This is needed to read `.zst` compressed files

### Database query errors
- The database schema may differ from what we expect
- Check the actual schema: `sqlite3 spotify_clean_track_files.sqlite3 ".schema"`
- Update queries in `annas_archive_helper.py` accordingly

## Benefits of Using Anna's Archive

1. **No API rate limits** - Access data locally
2. **Faster** - No network calls needed
3. **Reliable** - No 403/429 errors
4. **Complete** - Includes all 256 million tracks
5. **Offline** - Works without internet (after initial download)

## References

- [Anna's Archive Spotify Backup Blog](https://annas-archive.li/blog/backing-up-spotify.html)
- [Anna's Archive Torrents Page](https://annas-archive.li/torrents)
