# Installing Transmission (BitTorrent Client)

## Option 1: Mac App Store (Easiest - Free)

1. **Click this link** to open the App Store:
   - https://apps.apple.com/app/transmission/id435390014

2. **Or search manually**:
   - Open the Mac App Store
   - Search for "Transmission"
   - Click "Get" or "Install"

3. **After installation**:
   - Open Transmission from Applications
   - File → Open Torrent File
   - Select `annas_archive_spotify_2025_07_metadata.torrent` from Downloads

## Option 2: Direct Download (Alternative)

1. **Visit Transmission website**:
   - https://transmissionbt.com/download/

2. **Download for macOS**:
   - Click the macOS download link
   - Open the `.dmg` file
   - Drag Transmission to Applications folder

3. **Open Transmission**:
   - Open Applications → Transmission
   - File → Open Torrent File
   - Select your torrent file

## Option 3: Homebrew (If you have it)

```bash
brew install --cask transmission
```

## After Installing

1. **Open Transmission**
2. **Add the torrent**:
   - File → Open Torrent File
   - Navigate to `~/Downloads/annas_archive_spotify_2025_07_metadata.torrent`
   - Click Open

3. **Wait for download**:
   - The download will start automatically
   - Check the progress in Transmission
   - Files will download to `~/Downloads/` by default (or check Transmission preferences)

4. **Once download completes**:
   - Files will be in `~/Downloads/annas_archive_spotify_2025_07_metadata/`
   - Then we can extract and set them up!

## Quick Check

After installing, you can verify it works:

```bash
open -a Transmission ~/Downloads/annas_archive_spotify_2025_07_metadata.torrent
```

This should open Transmission and start the download automatically.
