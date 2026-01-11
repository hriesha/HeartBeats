# HeartBeats 💓🎧🏃
A cross-platform app that synchronizes music tempo (BPM) with your heart rate or physical activity, letting you experience sound that literally moves with you. Whether you're running, cycling, meditating, or studying, the app dynamically adjusts playback speed or switches tracks to match your current rhythm (biometric input). Because common, we all cringe the same way when that one wrong song plays mid workout. No one wants that. 

## Overview
Modern music apps personalize playlists by genre or mood - but not by you.
This app bridges the gap between biometric sensing and real-time audio modulation, creating a fully adaptive soundscape. Using live BPM data from your phone sensors (or smartwatch), the system matches, accelerates, or calms the track tempo to keep your flow state optimal.

✨ AND the app is also accessible to users without smartwatches. By collecting BPM data from multiple workouts, we plan to introduce workout-specific modes with curated playlists based on predefined average BPM ranges.

Example:
- Jogging at 145 BPM → The app queues upbeat, 145 BPM electronic tracks.
- Cooling down to 95 BPM → It fades into calmer lo-fi tunes, auto-adjusting tempo via time-stretching.
- Over time, the AI learns from user activity to recommend tempo profiles and playlists automatically.

## Features 
- Heart Rate Sync: Reads BPM from phone camera, watch, or fitness device. 
- Dynamic Tempo Engine: Performs real-time time-stretching and beat-matching without pitch distortion.
- Adaptive Modes:
    - Workout Mode – maintains tempo alignment with elevated BPM zones.
    - Focus Mode – sustains a stable tempo for concentration.
    - Relax Mode – gradually decreases tempo for calmness.
    - No-Track Mode - for when you don't have a fitness watch.
- Visualization Dashboard: Displays live waveform + BPM graph.
- Smart Playlist Recommender: Matches Spotify or Apple Music tracks with your rhythm profile.
- Custom Controls: Set BPM targets, toggle auto-sync, or manually calibrate heart rate.
- AI-Powered Adaptation: Learns from historical BPM data to predict and maintain optimal tempo ranges for various activities.

## How It Works
1. Input Layer: captures real-time BPM via:
   - Smartwatch API (Apple HealthKit / Google Fit)
   - Manual input or calliberation mode.
2. AI Signal Processor: Filters noise & averages recent BPM readings. Helps forecast user's ideal tempo range.
3. Sync Engine:
   - Maps current BPM to song tempo range
   - Adjusts playback speed using `pydub`, `librosa`, `soundstretch` backend
   - Handles crossfades to avoid abrupt changes
4. Frontend (Streamlit / React): Displays your live BPM, music info, and tempo graph.


## Tech Stack
- Frontend         | React + TypeScript + Vite (Tailwind CSS, Motion, Radix UI)
- Backend          | Flask or FastAPI                               
- AI & Modeling    | `scikit-learn`, `numpy`, `pandas`               
- Audio Processing | `pydub`, `librosa`, `soundstretch`              
- BPM Detection    | `heartpy`, OpenCV (camera), wearable APIs       
- Deployment       | Vercel (frontend), Render/AWS (backend)        
- Integrations     | Spotify Web API, Apple MusicKit

## Getting Started

### Frontend Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

3. Build for production:
```bash
npm run build
```

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
HeartBeats/
├── src/                    # React frontend source code
│   ├── components/         # React components
│   ├── styles/            # Global styles
│   └── App.tsx            # Main app component
├── basic-api-demo/        # Python scripts for API testing
├── package.json           # Frontend dependencies
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Team: Saachi, Hriesha
