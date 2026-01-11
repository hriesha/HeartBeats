#!/bin/bash
# Start HeartBeats Backend API
# This script handles numpy architecture issues

cd "$(dirname "$0")"

# Change to a safe directory before importing
cd /tmp

# Run the API from the project directory
exec python3 -c "
import sys
import os
project_dir = '$PWD'
os.chdir('/tmp')
sys.path.insert(0, project_dir)
os.chdir(project_dir)

# Now import and run
from api.heartbeats_api import app
import os
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port, debug=True)
"
