#!/bin/bash
# Start HeartBeats Application
# Starts both backend and frontend

cd "$(dirname "$0")"

echo "🚀 Starting HeartBeats..."
echo ""

# Check if numpy works
echo "Checking Python dependencies..."
if ! python3 -c "import numpy; import pandas; import sklearn" 2>/dev/null; then
    echo "❌ NumPy/Pandas not working. Please run: ./fix_numpy.sh"
    echo "   Or: pip3 install --upgrade --force-reinstall numpy pandas scikit-learn"
    exit 1
fi
echo "✅ Python dependencies OK"
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

echo ""
echo "Starting servers..."
echo "  Backend:  http://localhost:5001"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start backend in background
python3 api/heartbeats_api.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend
npm run dev &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
