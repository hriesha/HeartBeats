#!/bin/bash
# Fix NumPy architecture issue for Apple Silicon Macs

echo "🔧 Fixing NumPy architecture issue..."
echo ""

# Check if we're on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo "✅ Detected Apple Silicon (arm64)"
else
    echo "⚠️  Not Apple Silicon, but proceeding anyway..."
fi

echo ""
echo "Uninstalling old numpy/pandas..."
pip3 uninstall -y numpy pandas scikit-learn 2>/dev/null || true

echo ""
echo "Installing numpy/pandas for your architecture..."
pip3 install --upgrade --force-reinstall --no-cache-dir numpy pandas scikit-learn

echo ""
echo "Verifying installation..."
python3 -c "import numpy; import pandas; import sklearn; print('✅ All packages installed successfully!')" && echo "✅ Fix complete!" || echo "❌ Installation failed. Try: pip3 install --user numpy pandas scikit-learn"
