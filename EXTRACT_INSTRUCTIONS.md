# Quick Setup Guide - Anna's Archive Audio Features

## What You Actually Need

**Only download this ONE file:**
- ✅ `spotify_clean_audio_features.sqlite3.zst` (17.73 GB)

**You DON'T need:**
- ❌ Audiobooks files
- ❌ Playlists database
- ❌ Shows/episodes
- ❌ Other metadata files

**Total download: ~18 GB instead of 200 GB!**

## Steps

### 1. In Transmission Dialog (Current Step)

1. **Uncheck all files** except:
   - `spotify_clean_audio_features.sqlite3.zst` (17.73 GB)

2. Click **"Add"** button

3. Wait for download (will take a while for 18 GB, but much faster than 200 GB!)

### 2. After Download Completes

The file will be downloaded to `~/Downloads/` as:
```
spotify_clean_audio_features.sqlite3.zst
```

### 3. Extract the File

The `.zst` file is compressed. You need to extract it:

```bash
# Install zstandard if needed
pip3 install zstandard

# Extract the file
cd ~/Downloads
zstd -d spotify_clean_audio_features.sqlite3.zst

# This creates: spotify_clean_audio_features.sqlite3
```

### 4. Move to Project Directory

```bash
# Copy to project
mkdir -p /Users/saachidhamija/Desktop/projects/HeartBeats/annas_archive_data
cp ~/Downloads/spotify_clean_audio_features.sqlite3 /Users/saachidhamija/Desktop/projects/HeartBeats/annas_archive_data/
```

### 5. Done!

The API server will automatically detect and use this database.

## Quick Commands (After Download)

```bash
# Install extractor
pip3 install zstandard

# Extract
cd ~/Downloads
zstd -d spotify_clean_audio_features.sqlite3.zst

# Move to project
mkdir -p ~/Desktop/projects/HeartBeats/annas_archive_data
mv spotify_clean_audio_features.sqlite3 ~/Desktop/projects/HeartBeats/annas_archive_data/

# Restart API server and it will use the archive!
```
