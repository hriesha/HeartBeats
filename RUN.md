# Running HeartBeats

## Issue: NumPy Architecture Mismatch

Your system has numpy installed for x86_64 architecture, but your Mac needs arm64 (Apple Silicon). 

## Quick Fix

Run these commands in your terminal:

```bash
# Option 1: Use a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Option 2: Reinstall numpy/pandas for your architecture
pip3 install --upgrade --force-reinstall numpy pandas scikit-learn
```

## Running the Application

### Terminal 1 - Backend API:
```bash
cd /Users/hriesha/Desktop/HeartBeats
python3 api/heartbeats_api.py
```

The API will run on `http://localhost:5000`

### Terminal 2 - Frontend:
```bash
cd /Users/hriesha/Desktop/HeartBeats
npm run dev
```

The frontend will run on `http://localhost:5173` (or similar)

## Verify It's Working

1. Open `http://localhost:5173` in your browser
2. You should see the HeartBeats login screen
3. The backend API should respond at `http://localhost:5000/api/health`

## Troubleshooting

- **If numpy still fails**: Try using `python3 -m pip install --user numpy pandas`
- **If API won't start**: Check that `basic-api-demo/audio_features_sample.csv` exists
- **If frontend won't connect**: Make sure the backend is running first
