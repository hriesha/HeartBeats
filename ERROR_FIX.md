# Error Fix Guide

## 🔴 Main Issue: NumPy Architecture Mismatch

**Problem**: NumPy/Pandas/scikit-learn are installed for x86_64 architecture, but your Mac needs arm64 (Apple Silicon).

**Error Message**:
```
Error importing numpy: you should not try to import numpy from
its source directory; please exit the numpy source tree, and relaunch
your python interpreter from there.
```

## ✅ Solutions (Choose One)

### Solution 1: Use Virtual Environment (RECOMMENDED)

This is the cleanest solution and avoids system-wide Python issues:

```bash
# Step 1: Create and setup virtual environment
./setup_venv.sh

# Step 2: Start the app
./start_venv.sh
```

### Solution 2: Fix System Python

```bash
# Run the fix script
./fix_numpy.sh

# Then start normally
./start.sh
```

### Solution 3: Manual Fix

```bash
# Uninstall old packages
pip3 uninstall -y numpy pandas scikit-learn

# Reinstall for your architecture
pip3 install --upgrade --force-reinstall numpy pandas scikit-learn

# Or use --user flag if permission issues
pip3 install --user --upgrade --force-reinstall numpy pandas scikit-learn
```

## 🔍 Check What's Wrong

Run the error checker:
```bash
./check_errors.sh
```

This will show you exactly what's failing.

## 📋 Current Status

Based on the error check:
- ✅ Python3: Working
- ✅ Flask: Installed
- ✅ Node.js: Working  
- ✅ npm: Working
- ✅ Frontend dependencies: Installed
- ✅ Data files: Present
- ❌ NumPy: **FAILING** (architecture mismatch)
- ❌ Pandas: **FAILING** (depends on NumPy)
- ❌ scikit-learn: **FAILING** (depends on NumPy)

## 🚀 Quick Start (After Fixing NumPy)

Once NumPy is fixed, you can start the app:

```bash
# Option 1: Using virtual environment (best)
./start_venv.sh

# Option 2: Using system Python
./start.sh

# Option 3: Manual
# Terminal 1:
python3 api/heartbeats_api.py

# Terminal 2:
npm run dev
```

## 🐛 Other Potential Errors

### Frontend Errors
- **CORS errors**: Make sure backend is running first
- **API connection errors**: Check that backend is on port 5000
- **TypeScript errors**: Run `npm run build` to see detailed errors

### Backend Errors
- **Port already in use**: `lsof -ti:5000 | xargs kill`
- **CSV file not found**: Make sure `basic-api-demo/audio_features_sample.csv` exists
- **Spotify API errors**: Optional - app works without it (just shows basic track info)

## 📞 Still Having Issues?

1. Run `./check_errors.sh` to see what's failing
2. Check the terminal output for specific error messages
3. Check browser console (F12) for frontend errors
4. Make sure you're using the correct Python version (3.8+)
