#!/usr/bin/env python3
"""
Wrapper script to run the API server.
This changes to a safe directory before importing numpy to avoid architecture issues.
"""
import os
import sys

# Change to a safe directory before importing anything
original_dir = os.path.dirname(os.path.abspath(__file__))
safe_dir = '/tmp'

# Change to safe directory
os.chdir(safe_dir)

# Add project directory to path
sys.path.insert(0, original_dir)

# Now change back to project directory but keep imports from safe context
os.chdir(original_dir)

# Import and run the app
try:
    from api.heartbeats_api import app
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting HeartBeats API on http://0.0.0.0:{port}")
    print(f"📁 Working directory: {original_dir}")
    app.run(host='0.0.0.0', port=port, debug=True)
except ImportError as e:
    print("=" * 60)
    print("ERROR: Failed to start API")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nPlease fix numpy installation:")
    print("  ./fix_numpy.sh")
    print("  OR")
    print("  pip3 install --upgrade --force-reinstall numpy pandas scikit-learn")
    print("=" * 60)
    sys.exit(1)
