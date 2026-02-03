# Merging Spotify 12M Songs Dataset

## Overview

The [Spotify 12M Songs Dataset](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) contains **12 million tracks** - much larger than our current ~113k dataset. Merging this will significantly improve:

1. **Model Performance**: More training data = better predictions
2. **Lookup Coverage**: More tracks in lookup = faster inference
3. **Genre Diversity**: More genres and artists = better generalization

## Benefits

- **~100x more training data** (12M vs 113k)
- **Better tempo prediction** (more examples to learn from)
- **Higher lookup coverage** (more tracks have precomputed features)
- **More diverse genres** (better genre-based predictions)

## Steps to Merge

### 1. Download the Dataset

From Kaggle:
```bash
# Install kaggle CLI if needed
pip install kaggle

# Download dataset
kaggle datasets download -d rodolfofigueroa/spotify-12m-songs

# Unzip
unzip spotify-12m-songs.zip
```

Or download manually from: https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs

### 2. Merge Datasets

```bash
# Merge current dataset with 12M dataset
python3 -m recs.merge_datasets \
  --current enriched_kaggle.csv \
  --kaggle-12m /path/to/spotify-12m-songs.csv \
  --output merged_spotify_dataset.csv

# Or limit 12M dataset for testing (first 1M tracks)
python3 -m recs.merge_datasets \
  --current enriched_kaggle.csv \
  --kaggle-12m /path/to/spotify-12m-songs.csv \
  --limit-12m 1000000 \
  --output merged_spotify_dataset.csv

# Or sample final dataset (random 500k tracks)
python3 -m recs.merge_datasets \
  --current enriched_kaggle.csv \
  --kaggle-12m /path/to/spotify-12m-songs.csv \
  --sample 500000 \
  --output merged_spotify_dataset.csv
```

### 3. Train with Merged Dataset

```bash
# Train improved model on merged dataset
python3 -m recs.train_metadata_model_improved \
  --enriched-csv merged_spotify_dataset.csv \
  --model-type xgboost \
  --top-genres 50

# This will take longer (~30-60 min for 1M tracks)
```

### 4. Retrain Core Clustering Model

```bash
# Retrain clustering model with merged dataset
python3 -m recs.train \
  --csv merged_spotify_dataset.csv \
  --sample 500000 \
  --clusters 6

# This builds the lookup database with more tracks
```

## Expected Improvements

With 12M tracks:

- **Tempo R²**: 0.057 → **0.15-0.25** (3-4x improvement)
- **Energy R²**: 0.23 → **0.40-0.50** (2x improvement)
- **Lookup Coverage**: ~10% → **50-70%** of user's library
- **Prediction Accuracy**: Much better for rare genres/artists

## Memory Considerations

The 12M dataset is large (~2-5 GB). Options:

1. **Use a subset**: `--limit-12m 1000000` (1M tracks)
2. **Sample randomly**: `--sample 500000` (500k tracks)
3. **Process in chunks**: The script handles this automatically

## Notes

- The script automatically handles column name differences
- Duplicates are removed (by track_id)
- Only tracks with valid audio features are kept
- Missing metadata is filled with defaults
