#!/bin/bash
# Start backend API on port 5001

cd "$(dirname "$0")"

echo "🚀 Starting HeartBeats Backend API on port 5001..."
echo ""

# Check dependencies
if ! python3 -c "import numpy; import pandas; import sklearn" 2>/dev/null; then
    echo "❌ Dependencies not working. Run: ./fix_numpy.sh"
    exit 1
fi

# Start the API
python3 api/heartbeats_api.py
