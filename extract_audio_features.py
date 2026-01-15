#!/usr/bin/env python3
"""
Extract spotify_clean_audio_features.sqlite3 from .zst compressed file
"""

import os
import sys
import zstandard as zstd

def extract_zst(input_path: str, output_path: str):
    """Extract .zst file to output path."""
    print(f"Extracting {input_path}...")
    print(f"This may take a few minutes for a 17GB file...")

    with open(input_path, 'rb') as input_file:
        dctx = zstd.ZstdDecompressor()
        with open(output_path, 'wb') as output_file:
            dctx.copy_stream(input_file, output_file)

    print(f"✓ Extracted to {output_path}")

    # Check file size
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"✓ Extracted file size: {size_gb:.2f} GB")

if __name__ == '__main__':
    # File is in a subdirectory
    input_file = os.path.expanduser('~/Downloads/annas_archive_spotify_2025_07_metadata/spotify_clean_audio_features.sqlite3.zst')
    output_file = os.path.expanduser('~/Downloads/annas_archive_spotify_2025_07_metadata/spotify_clean_audio_features.sqlite3')

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Make sure the file is in your Downloads folder")
        sys.exit(1)

    extract_zst(input_file, output_file)

    # Move to project directory
    project_dir = '/Users/saachidhamija/Desktop/projects/HeartBeats/annas_archive_data'
    os.makedirs(project_dir, exist_ok=True)

    target_path = os.path.join(project_dir, 'spotify_clean_audio_features.sqlite3')

    print(f"\nMoving to project directory...")
    import shutil
    shutil.move(output_file, target_path)
    print(f"✓ Moved to {target_path}")
    print(f"\n✅ Setup complete! The API server will now use Anna's Archive data.")
