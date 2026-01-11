# HeartBeats API Backend

Flask API server for HeartBeats matching algorithm integration.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (copy `.env.example` to `.env` and fill in your Spotify credentials)

3. Run the API server:
```bash
python api/heartbeats_api.py
```

The API will run on `http://localhost:5000`

## Endpoints

### `POST /api/clusters`
Run KMeans clustering on user's library.

**Request:**
```json
{
  "csv_path": "basic-api-demo/audio_features_sample.csv",
  "n_clusters": 4
}
```

**Response:**
```json
{
  "success": true,
  "clusters": [
    {
      "cluster_id": 0,
      "count": 25,
      "mean_tempo": 120.5,
      "mean_energy": 0.65,
      "mean_danceability": 0.72
    }
  ],
  "total_tracks": 100
}
```

### `POST /api/tracks`
Get tracks for a given BPM and cluster using KNN.

**Request:**
```json
{
  "bpm": 130,
  "cluster_id": 0,
  "topk": 10
}
```

**Response:**
```json
{
  "success": true,
  "cluster_id": 0,
  "tracks": [
    {
      "track_id": "4uLU6hMCjMI75M1A2tKUQC",
      "name": "Song Name",
      "artists": "Artist Name",
      "cluster": 0,
      "tempo": 128.5,
      "rank": 1
    }
  ],
  "count": 10
}
```

### `POST /api/tracks/details`
Get full track details from Spotify API.

**Request:**
```json
{
  "track_ids": ["4uLU6hMCjMI75M1A2tKUQC", "7qiZfU4dY1lWllzX7mPKB3"]
}
```

### `POST /api/cluster/tracks`
Combined endpoint: Get tracks for BPM + cluster, then fetch full details.

**Request:**
```json
{
  "bpm": 130,
  "cluster_id": 0,
  "topk": 10
}
```
