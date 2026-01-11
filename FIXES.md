# Fixes Applied

## Issues Fixed

### 1. NumPy Architecture Mismatch
- **Problem**: NumPy was installed for x86_64 but Mac needs arm64
- **Solution**: Created `fix_numpy.sh` script to reinstall packages
- **Fix Command**: Run `./fix_numpy.sh` or manually:
  ```bash
  pip3 install --upgrade --force-reinstall numpy pandas scikit-learn
  ```

### 2. API Response Format
- **Problem**: Frontend expected nested `data` object but API returned flat structure
- **Solution**: Updated `heartbeatsApi.ts` to extract data correctly from API responses

### 3. Error Handling
- **Problem**: No clear error messages when dependencies fail
- **Solution**: Added better error messages in API startup

### 4. Startup Scripts
- **Created**: `start.sh` - Starts both backend and frontend
- **Created**: `fix_numpy.sh` - Fixes numpy installation
- **Created**: `start_backend.sh` - Alternative backend starter

## How to Run

### Option 1: Use the startup script (recommended)
```bash
./start.sh
```

### Option 2: Manual startup

1. **Fix numpy first** (if needed):
   ```bash
   ./fix_numpy.sh
   ```

2. **Start backend**:
   ```bash
   python3 api/heartbeats_api.py
   ```

3. **Start frontend** (in another terminal):
   ```bash
   npm run dev
   ```

## Troubleshooting

### Backend won't start
- Run `./fix_numpy.sh` to fix numpy
- Check that `basic-api-demo/audio_features_sample.csv` exists
- Verify Python dependencies: `pip3 list | grep -E "(flask|pandas|numpy|sklearn)"`

### Frontend won't connect
- Make sure backend is running on port 5000
- Check browser console for CORS errors
- Verify `VITE_API_URL` in `.env` matches backend URL

### API returns errors
- Check backend terminal for error messages
- Verify CSV file path is correct
- Ensure Spotify credentials are set in `.env` (optional, for full track details)
