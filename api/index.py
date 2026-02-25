import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from heartbeats_api import app
    handler = app
except Exception as _e:
    from flask import Flask, jsonify
    handler = Flask(__name__)

    @handler.route("/api/health")
    @handler.route("/api/<path:p>")
    def _err(p=""):
        return jsonify({"import_error": str(_e), "sys_path": sys.path[:4]}), 500
