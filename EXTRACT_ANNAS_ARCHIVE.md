# How to Extract Anna's Archive Spotify Data

## Step 1: Download the Torrent

You have the `.torrent` file in your Downloads. Now you need to:

1. **Open the torrent file** with a BitTorrent client:
   - On macOS, you can use:
     - **Transmission** (built-in or download from App Store)
     - **qBittorrent** (free, recommended)
     - **Deluge** (free)

2. **Add the torrent**:
   - Double-click `annas_archive_spotify_2025_07_metadata.torrent`
   - OR open your torrent client and File → Add Torrent
   - Select the `.torrent` file

3. **Wait for download to complete**:
   - The metadata torrent is usually several GB (much smaller than the full 300TB!)
   - Check the torrent client's progress
   - It will download to a default location (usually `~/Downloads/` or `~/Downloads/Torrents/`)

## Step 2: Find the Downloaded Files

Once the download completes, look for:
- A folder named something like: `annas_archive_spotify_2025_07_metadata/`
- Files like:
  - `spotify_clean_track_files.sqlite3`
  - `spotify_*.jsonl.zst` files
  - Or `.tar` / `.zip` archives that need extraction

## Step 3: Extract (if needed)

The files might be:
- **Already extracted** (if they're SQLite/JSONL files, you're done!)
- **Compressed** in `.tar`, `.tar.gz`, or `.zip` files

### If you have .tar files:
```bash
cd ~/Downloads  # or wherever the torrent downloaded to
tar -xf annas_archive_spotify_2025_07_metadata.tar
# or if it's .tar.gz:
tar -xzf annas_archive_spotify_2025_07_metadata.tar.gz
```

### If you have .zip files:
```bash
unzip annas_archive_spotify_2025_07_metadata.zip
```

### If you have .zst files:
These are already extracted but compressed. We'll handle them in code.

## Step 4: Move to Project Directory

Once extracted, move/copy the data to your project:

```bash
# Create the directory (already done)
cd /Users/saachidhamija/Desktop/projects/HeartBeats

# Copy the SQLite database
cp ~/Downloads/[torrent_folder]/spotify_clean_track_files.sqlite3 annas_archive_data/

# Or copy the entire folder
cp -r ~/Downloads/[torrent_folder]/* annas_archive_data/
```

## Step 5: Verify

Check that files are in place:

```bash
ls -lh annas_archive_data/
```

You should see at least:
- `spotify_clean_track_files.sqlite3` (or similar)
- OR JSONL files with audio features

## Quick Commands

Once you find where your torrent client downloaded the files:

```bash
# Find the downloaded files
find ~/Downloads -name "spotify_clean*" -o -name "*spotify*metadata*" 2>/dev/null

# Create target directory
mkdir -p /Users/saachidhamija/Desktop/projects/HeartBeats/annas_archive_data

# Copy files (adjust path as needed)
cp -r ~/Downloads/[FOLDER_NAME]/* /Users/saachidhamija/Desktop/projects/HeartBeats/annas_archive_data/
```

## Need Help?

If you can't find the files:
1. Check your torrent client's download folder settings
2. Check the torrent client's "Completed Downloads" or "Seeding" list
3. The files might be in `~/Downloads/[torrent_name]/` or similar

Let me know where the files are and I can help you extract/set them up!
