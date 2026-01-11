# HeartBeats

A music recommendation app that matches songs to your heart rate using KMeans clustering and KNN matching.

## Features

- **BPM Selection**: Choose your target heart rate or select from workout presets
- **Dynamic Clustering**: KMeans clustering of your music library creates personalized vibes
- **KNN Matching**: Find the best songs matching your BPM within each cluster
- **Spotify Integration**: Get full track details, previews, and album art
- **Queue View**: Beautiful queue interface showing your matched songs

## Setup

### Backend (Python API)

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Spotify API credentials
```

3. Run the API server:
```bash
python api/heartbeats_api.py
```

The API will run on `http://localhost:5001` (port 5000 is used by AirPlay)

### Frontend (React + TypeScript)

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables (optional, defaults to `http://localhost:5000`):
```bash
# Create .env file
echo "VITE_API_URL=http://localhost:5000" > .env
```

3. Run the development server:
```bash
npm run dev
```

## Project Structure

```
HeartBeats/
├── api/                    # Flask backend API
│   ├── heartbeats_api.py  # Main API server
│   └── README.md          # API documentation
├── src/                    # React frontend
│   ├── components/        # UI components
│   │   ├── BPMSelection.tsx
│   │   ├── VibeSelection.tsx
│   │   ├── VibeDetail.tsx
│   │   └── SongQueue.tsx
│   ├── services/          # API client
│   │   └── heartbeatsApi.ts
│   └── App.tsx            # Main app component
├── basic-api-demo/        # Algorithm scripts
│   ├── clustering-match.py
│   └── knn-kmeans-match.py
└── spotify_api_integration.py  # Spotify API wrapper
```

## How It Works

1. **User selects BPM**: Through custom input or workout preset
2. **KMeans Clustering**: User's library is clustered into vibes based on audio features (tempo, energy, danceability, valence, loudness)
3. **Cluster Selection**: User picks a vibe/cluster
4. **KNN Matching**: Within the selected cluster, KNN finds the top-K songs closest to the target BPM
5. **Queue Display**: Songs are displayed with full Spotify metadata, previews, and album art

## API Endpoints

See `api/README.md` for detailed API documentation.

## Development

- Backend: Python 3.8+, Flask, scikit-learn, spotipy
- Frontend: React 18+, TypeScript, Vite, Tailwind CSS, Framer Motion
