# Quick Start Guide

## 🚀 Fastest Way to Run

1. **Fix NumPy** (one-time setup):
   ```bash
   ./fix_numpy.sh
   ```

2. **Start everything**:
   ```bash
   ./start.sh
   ```

3. **Open browser**: http://localhost:5173

## 🔧 If NumPy Fix Fails

Try these alternatives:

```bash
# Option 1: User install
pip3 install --user --upgrade --force-reinstall numpy pandas scikit-learn

# Option 2: Use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 api/heartbeats_api.py
```

## 📋 Manual Startup (if scripts don't work)

### Terminal 1 - Backend:
```bash
cd /Users/hriesha/Desktop/HeartBeats
python3 api/heartbeats_api.py
```

### Terminal 2 - Frontend:
```bash
cd /Users/hriesha/Desktop/HeartBeats
npm run dev
```

## ✅ Verify It's Working

1. Backend health check:
   ```bash
   curl http://localhost:5000/api/health
   ```
   Should return: `{"status":"ok"}`

2. Open http://localhost:5173 in browser
3. You should see the HeartBeats login screen

## 🐛 Common Issues

- **"NumPy import error"**: Run `./fix_numpy.sh`
- **"Port already in use"**: Kill the process: `lsof -ti:5000 | xargs kill`
- **"Module not found"**: Run `npm install` for frontend, `pip install -r requirements.txt` for backend
- **"CORS error"**: Make sure backend is running before frontend
