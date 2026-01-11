#!/bin/bash
# Start HeartBeats using virtual environment

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Running setup..."
    ./setup_venv.sh
fi

# Activate venv
source venv/bin/activate

echo "🚀 Starting HeartBeats with virtual environment..."
echo ""

# Check dependencies
if ! python3 -c "import numpy; import pandas; import sklearn" 2>/dev/null; then
    echo "❌ Dependencies not installed. Installing..."
    pip install -r requirements.txt
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
    deactivate 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
