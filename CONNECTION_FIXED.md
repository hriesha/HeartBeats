# Connection Issue Fixed! ✅

## Problem
- Port 5000 was being used by Apple's AirPlay service
- Missing `python-dotenv` dependency was preventing API from starting

## Solutions Applied

### 1. Changed Port to 5001
- Backend API now runs on **port 5001** instead of 5000
- Updated `api/heartbeats_api.py` to use port 5001
- Updated `src/services/heartbeatsApi.ts` to connect to port 5001

### 2. Made Spotify Integration Optional
- API can now start without `python-dotenv` and `spotipy`
- Spotify features will show basic track info if not available
- Full Spotify features require: `pip3 install python-dotenv spotipy`

## Current Status

✅ **Backend API**: Running on `http://localhost:5001`
- Health check: `curl http://localhost:5001/api/health`
- Returns: `{"status":"ok"}`

⏳ **Frontend**: May need restart to pick up new port

## To Use

### Start Backend:
```bash
python3 api/heartbeats_api.py
```

### Start Frontend:
```bash
npm run dev
```

The frontend will automatically connect to `http://localhost:5001`.

## Test Connection

```bash
# Test backend health
curl http://localhost:5001/api/health

# Test clustering endpoint
curl -X POST http://localhost:5001/api/clusters \
  -H "Content-Type: application/json" \
  -d '{"n_clusters": 4}'
```

## Optional: Install Spotify Support

For full track details, previews, and album art:
```bash
pip3 install python-dotenv spotipy
```

Then add your Spotify credentials to `.env` file.
