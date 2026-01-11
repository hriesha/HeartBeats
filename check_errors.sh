#!/bin/bash
# Check for common errors in HeartBeats setup

cd "$(dirname "$0")"

echo "🔍 Checking for errors..."
echo ""

# Check Python
echo "1. Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "   ❌ Python3 not found"
else
    echo "   ✅ Python3 found: $(python3 --version)"
fi

# Check numpy
echo ""
echo "2. Checking NumPy..."
if python3 -c "import numpy" 2>/dev/null; then
    echo "   ✅ NumPy works"
    python3 -c "import numpy; print(f'   Version: {numpy.__version__}')"
else
    echo "   ❌ NumPy import failed"
    echo "   Run: ./fix_numpy.sh or ./setup_venv.sh"
fi

# Check pandas
echo ""
echo "3. Checking Pandas..."
if python3 -c "import pandas" 2>/dev/null; then
    echo "   ✅ Pandas works"
else
    echo "   ❌ Pandas import failed"
fi

# Check sklearn
echo ""
echo "4. Checking scikit-learn..."
if python3 -c "import sklearn" 2>/dev/null; then
    echo "   ✅ scikit-learn works"
else
    echo "   ❌ scikit-learn import failed"
fi

# Check Flask
echo ""
echo "5. Checking Flask..."
if python3 -c "import flask" 2>/dev/null; then
    echo "   ✅ Flask works"
else
    echo "   ❌ Flask not installed"
    echo "   Run: pip3 install flask flask-cors"
fi

# Check Node
echo ""
echo "6. Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "   ❌ Node.js not found"
else
    echo "   ✅ Node.js found: $(node --version)"
fi

# Check npm
echo ""
echo "7. Checking npm..."
if ! command -v npm &> /dev/null; then
    echo "   ❌ npm not found"
else
    echo "   ✅ npm found: $(npm --version)"
fi

# Check node_modules
echo ""
echo "8. Checking frontend dependencies..."
if [ -d "node_modules" ]; then
    echo "   ✅ node_modules exists"
else
    echo "   ⚠️  node_modules not found"
    echo "   Run: npm install"
fi

# Check CSV file
echo ""
echo "9. Checking data files..."
if [ -f "basic-api-demo/audio_features_sample.csv" ]; then
    echo "   ✅ audio_features_sample.csv exists"
else
    echo "   ⚠️  audio_features_sample.csv not found"
fi

# Check API file
echo ""
echo "10. Checking API file..."
if [ -f "api/heartbeats_api.py" ]; then
    echo "   ✅ heartbeats_api.py exists"
    # Try to import it
    if python3 -c "import sys; sys.path.insert(0, '.'); from api import heartbeats_api" 2>/dev/null; then
        echo "   ✅ API imports successfully"
    else
        echo "   ❌ API import failed (likely numpy issue)"
    fi
else
    echo "   ❌ heartbeats_api.py not found"
fi

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo ""
echo "If NumPy/Pandas failed, try:"
echo "  Option 1: ./fix_numpy.sh"
echo "  Option 2: ./setup_venv.sh (recommended)"
echo ""
echo "To start the app:"
echo "  ./start_venv.sh  (uses virtual environment)"
echo "  OR"
echo "  ./start.sh       (uses system Python)"
echo ""
