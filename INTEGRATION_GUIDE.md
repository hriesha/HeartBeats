# HeartBeats Integration Guide

## Overview

This integration connects your React UI with the Python backend algorithm and Spotify API to provide a complete music recommendation system based on heart rate BPM.

## Architecture

1. **Frontend (React/TypeScript)**: UI components that display clusters and track queues
2. **Backend API (Flask)**: Handles KMeans clustering and KNN matching
3. **Spotify Integration (Python)**: Fetches user's tracks and audio features

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
npm install
```

### 3. Set Up Environment Variables

Ensure your `.env` file has:
```
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

### 4. Start the Backend API Server

```bash
python api_server.py
```

The server will run on `http://localhost:5000`

### 5. Start the Frontend Dev Server

```bash
npm run dev
```

The frontend will run on `http://localhost:5173` (or similar Vite port)

## Flow

1. **User Login** → **Spotify Connection** → **Control Options**
2. **BPM Selection**: User selects target heart rate (BPM)
3. **Clustering**: Backend fetches user's saved tracks, runs KMeans clustering
4. **Vibe Selection**: UI displays 4 clusters (vibes) dynamically
5. **Track Queue**: When a vibe is selected, KNN finds best matching tracks
6. **Spotify Details**: Frontend fetches full track details (artwork, preview, etc.)

## API Endpoints

### `POST /api/cluster`
Runs KMeans clustering on user's tracks.

**Request:**
```json
{
  "bpm": 120,
  "n_clusters": 4
}
```

**Response:**
```json
{
  "clusters": [
    {
      "id": 0,
      "name": "Chill Flow",
      "color": "#EAE2B7",
      "tags": ["lo-fi", "calm"],
      "mean_tempo": 90.5,
      "track_count": 25
    }
  ],
  "df": [...]
}
```

### `GET /api/cluster/<cluster_id>/tracks?bpm=<bpm>&topk=<topk>`
Gets KNN-matched tracks for a cluster.

**Response:**
```json
{
  "tracks": [
    {
      "track_id": "...",
      "name": "Song Name",
      "artists": "Artist Name",
      "tempo": 120.5,
      "rank": 1,
      "distance": 0.23
    }
  ],
  "cluster_id": 0,
  "bpm": 120
}
```

### `POST /api/tracks/details`
Gets detailed Spotify track information.

**Request:**
```json
{
  "track_ids": ["track_id1", "track_id2"]
}
```

**Response:**
```json
{
  "tracks": [
    {
      "id": "track_id",
      "name": "Song Name",
      "artist_names": "Artist",
      "album": "Album Name",
      "preview_url": "...",
      "images": [...],
      "duration_ms": 240000
    }
  ]
}
```

## Files Created/Modified

### Backend
- `api_server.py`: Flask API server with clustering and KNN logic
- `spotify_api_integration.py`: (Already existed) Spotify API wrapper

### Frontend
- `src/utils/api.ts`: API utility functions
- `src/components/VibeSelection.tsx`: Updated to fetch clusters dynamically
- `src/components/VibeDetail.tsx`: Updated to show track queue with Spotify data
- `src/App.tsx`: Updated to pass BPM to VibeDetail

## How It Works

1. **KMeans Clustering**:
   - Takes user's saved Spotify tracks
   - Extracts audio features (tempo, energy, danceability, valence, loudness)
   - Clusters tracks into 4 groups based on these features
   - Each cluster represents a different "vibe"

2. **KNN Matching**:
   - When a cluster is selected, creates a query point with target BPM
   - Uses K-Nearest Neighbors to find the closest tracks in that cluster
   - Returns ranked list of best matching tracks

3. **Spotify Integration**:
   - Fetches full track metadata (artwork, preview URLs, etc.)
   - Enriches the track queue with visual and playback information

## Troubleshooting

- **"Failed to connect to Spotify"**: Make sure you've connected your Spotify account and have saved tracks
- **"No tracks found"**: User needs to have saved tracks in their Spotify library
- **CORS errors**: Make sure flask-cors is installed and the API server is running
- **API connection errors**: Check that the backend is running on port 5000, or update `VITE_API_URL` in frontend
