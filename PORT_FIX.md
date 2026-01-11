# Port Conflict Fix

## Issue
Port 5000 is being used by Apple's AirPlay service (AirTunes), causing connection failures.

## Solution
Changed backend API port from **5000** to **5001**.

## Updated Files
- `api/heartbeats_api.py` - Now uses port 5001
- `src/services/heartbeatsApi.ts` - Updated to connect to port 5001
- `start.sh` - Updated port references
- `start_venv.sh` - Updated port references

## How to Run

### Option 1: Use the startup script
```bash
./start.sh
```

### Option 2: Manual start
```bash
# Terminal 1 - Backend (now on port 5001)
python3 api/heartbeats_api.py

# Terminal 2 - Frontend
npm run dev
```

## Verify Connection

Test the backend:
```bash
curl http://localhost:5001/api/health
```

Should return: `{"status":"ok"}`

## Frontend Connection

The frontend will automatically connect to `http://localhost:5001` (or use `VITE_API_URL` env var if set).

If you already have the frontend running, you may need to restart it to pick up the new port.
