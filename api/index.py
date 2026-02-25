import sys
import os

# Ensure the api/ directory is on the path so sibling modules resolve
sys.path.insert(0, os.path.dirname(__file__))

from heartbeats_api import app

# Vercel serverless entry point
handler = app
