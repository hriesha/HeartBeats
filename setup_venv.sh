#!/bin/bash
# Create virtual environment and install dependencies
# This avoids numpy architecture issues

cd "$(dirname "$0")"

echo "🔧 Setting up virtual environment..."

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created virtual environment"
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use the virtual environment:"
echo "  source venv/bin/activate"
echo "  python3 api/heartbeats_api.py"
echo ""
echo "Or use the start script:"
echo "  ./start_venv.sh"
