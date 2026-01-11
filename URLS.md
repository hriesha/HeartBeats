# HeartBeats URLs

## ✅ Correct URLs to Access

### Frontend (React App)
**Open this in your browser:**
```
http://localhost:3000
```

This is where you interact with the HeartBeats app.

### Backend API (For testing/debugging)
```
http://localhost:5001/api/health
```

## ❌ Common Mistakes

### Don't access the backend directly in browser
If you go to `http://localhost:5001` in your browser, you'll get a "Not Found" error because Flask doesn't serve HTML - it only serves API endpoints.

### Don't access API routes without /api prefix
- ✅ Correct: `http://localhost:5001/api/health`
- ❌ Wrong: `http://localhost:5001/health`

## Available API Endpoints

All API endpoints are prefixed with `/api`:

- `GET /api/health` - Health check
- `POST /api/clusters` - Get clusters
- `POST /api/tracks` - Get tracks by BPM
- `POST /api/tracks/details` - Get track details from Spotify
- `POST /api/cluster/tracks` - Combined endpoint

## Frontend Proxy

The frontend (Vite) is configured to proxy `/api` requests to the backend automatically. So when the frontend makes requests to `/api/clusters`, Vite forwards them to `http://localhost:5001/api/clusters`.

## Quick Test

1. **Open frontend**: http://localhost:3000
2. **Test backend**: `curl http://localhost:5001/api/health`
