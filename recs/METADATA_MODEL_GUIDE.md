# Metadata → Audio Features Prediction Model

This guide explains how to train and use the metadata prediction model that allows clustering **ALL** tracks in a user's library, even if they're not in the Kaggle training dataset.

## Overview

The model predicts audio features (tempo, energy, danceability, valence, loudness) from Spotify metadata, then uses those predicted features to assign clusters. This means **every track gets clustered**, not just those in the training set.

## Workflow

### Step 1: Enrich Kaggle Dataset with Spotify Metadata

Fetch metadata for all Kaggle tracks (one-time, ~10-20 minutes):

```bash
python -m recs.enrich_kaggle_metadata \
  --csv /Users/saachidhamija/Downloads/spotify-kaggle-dataset.csv \
  --output enriched_kaggle.csv \
  --batch-size 50 \
  --delay 0.5
```

**Features fetched:**
- Track name, artists, album
- Popularity, release_date, duration_ms, explicit
- Genres (from artist)
- Related artists (top 5)

**Progress:** Saves to `metadata_progress.db` - can resume if interrupted.

### Step 2: Train Metadata → Audio Features Model

Train models to predict audio features from metadata (~5-10 minutes):

```bash
python -m recs.train_metadata_model \
  --enriched-csv enriched_kaggle.csv \
  --output-dir recs/model/
```

**Output:**
- `metadata_model_{feature}.joblib` - One model per audio feature (5 models)
- `metadata_feature_names.json` - Feature names (for inference)
- `metadata_model_metrics.json` - Training metrics (MAE, R²)

**Feature Vector (~28-30 features):**
- `popularity` (0-1)
- `release_year` (normalized)
- `duration` (normalized)
- `explicit` (0/1)
- `artist_encoded` (label encoded, normalized)
- `genre_{genre}` (20 binary features for top genres)
- `num_genres` (normalized)
- `related_tempo`, `related_energy`, `related_danceability`, `related_valence`, `related_loudness` (from related artists)

### Step 3: Use in API

The API automatically uses metadata prediction when clustering user's library:

**POST /api/clusters** with `use_recs_model: true`:
1. Fetches user's saved tracks
2. For each track:
   - **Fast path:** If in Kaggle lookup → use precomputed cluster
   - **Prediction path:** If NOT in lookup → fetch metadata → predict features → predict cluster
3. **Result:** ALL tracks get clustered (100% coverage)

**Coverage info** returned in response:
```json
{
  "coverage": {
    "total": 500,
    "clustered": 500,
    "pct": 100.0
  }
}
```

## How It Works

### Training Phase
```
Kaggle Dataset (114k tracks)
  ↓
Enrich with Spotify Metadata
  ↓
Train: Metadata Features → Audio Features (5 separate models)
  ↓
Save models to recs/model/
```

### Inference Phase
```
User's Track (not in Kaggle)
  ↓
Fetch Spotify Metadata (already have from auth)
  ↓
Predict Audio Features (using trained models)
  ↓
Scale Features (using trained scaler)
  ↓
Assign to Nearest Centroid → Cluster ID
  ↓
Use Predicted Features for KNN
```

## Feature Vector Details

**Current features (~28-30):**
1. **Popularity** - Track popularity (0-100 → normalized to 0-1)
2. **Release Year** - Extracted from release_date, normalized (1950-2020 range)
3. **Duration** - Track length in ms, normalized (0-300k ms → 0-1)
4. **Explicit** - Binary flag (0/1)
5. **Artist Encoded** - Label-encoded main artist ID, normalized
6. **Genres** - Multi-hot encoding for top 20 genres (20 binary features)
7. **Num Genres** - Count of genres, normalized
8. **Related Artists Features** - Average audio features from related artists' songs in training set (5 features: tempo, energy, danceability, valence, loudness)

## Files Created

- `recs/enrich_kaggle_metadata.py` - Batch metadata fetching script
- `recs/train_metadata_model.py` - Training script
- `recs/inference.py` - Updated with `predict_audio_features_from_metadata()`
- `recs/model/metadata_model_*.joblib` - Trained models (5 files)
- `recs/model/metadata_feature_names.json` - Feature names
- `recs/model/metadata_model_metrics.json` - Training metrics

## Usage in Code

```python
from recs.inference import predict_audio_features_from_metadata, predict_cluster_from_features

# Predict features from metadata
metadata = {
    "name": "Song Name",
    "artists": ["Artist"],
    "popularity": 75,
    "release_date": "2020-01-01",
    "duration_ms": 180000,
    "explicit": False,
    "genres": ["pop", "rock"],
    "main_artist_id": "artist_123",
}

predicted_features = predict_audio_features_from_metadata(metadata)
# Returns: {"tempo": 120.5, "energy": 0.65, ...}

# Predict cluster from features
cluster_id = predict_cluster_from_features(predicted_features)
# Returns: 3
```

## Next Steps

1. **Run enrichment** (one-time, ~10-20 min)
2. **Train model** (~5-10 min)
3. **Test with Spotify library** - All tracks should now cluster!

The API will automatically use metadata prediction when `use_recs_model: true` is set (which is now the default in the UI).
