#!/usr/bin/env python3
"""
Test the recs model with your Spotify library.

Prerequisites:
  - API server running (e.g. python api/heartbeats_api.py or run_api.py)
  - Spotify connected via the app (open the frontend, connect Spotify, then run this script)

Usage:
  python scripts/test_recs_spotify.py
  python scripts/test_recs_spotify.py --base http://localhost:5001
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    import urllib.request
    import urllib.error
except ImportError:
    print("Python 3 required (urllib).")
    sys.exit(1)

DEFAULT_BASE = "http://localhost:5001"


def request(method: str, path: str, body: dict | None = None, base: str = DEFAULT_BASE) -> dict:
    url = f"{base.rstrip('/')}/api{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    if body is not None:
        req.data = json.dumps(body).encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            return json.loads(err_body) if err_body else {"error": str(e)}
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test recs model with Spotify library")
    parser.add_argument("--base", default=DEFAULT_BASE, help="API base URL")
    args = parser.parse_args()
    base = args.base

    print("HeartBeats – Test recs with Spotify\n")
    print("Make sure the API is running and Spotify is connected via the app.\n")

    # 1. Health
    print("1. Health check...")
    health = request("GET", "/health", base=base)
    if health.get("status") != "ok":
        print("   FAIL:", health)
        sys.exit(1)
    print("   OK\n")

    # 2. Coverage
    print("2. Recs coverage (your saved tracks vs recs lookup)...")
    coverage = request("GET", "/recs/coverage", base=base)
    if not coverage.get("success"):
        print("   FAIL:", coverage.get("error", coverage))
        print("   (Connect Spotify in the app first.)")
        sys.exit(1)
    total = coverage.get("total_saved", 0)
    in_lookup = coverage.get("in_lookup", 0)
    pct = coverage.get("coverage_pct", 0)
    by_cluster = coverage.get("by_cluster") or {}
    print(f"   Total saved: {total}")
    print(f"   In recs lookup: {in_lookup} ({pct}%)")
    print(f"   By cluster: {by_cluster}\n")

    # 3. Clusters (recs)
    print("3. Clusters (recs model, your library)...")
    clusters_resp = request("POST", "/clusters", body={"use_recs_model": True, "use_spotify_library": True}, base=base)
    if not clusters_resp.get("success"):
        print("   FAIL:", clusters_resp.get("error", clusters_resp.get("message", clusters_resp)))
    else:
        clusters = clusters_resp.get("clusters", [])
        print(f"   Clusters: {len(clusters)}")
        for c in clusters[:6]:
            print(f"     - {c.get('name', '?')}: {c.get('count', 0)} tracks")
    print()

    # 4. From-track (sample)
    sample_ids = (coverage.get("sample_in_lookup") or [])[:1]
    if not sample_ids:
        print("4. From-track: no sample track in lookup, skipping.")
    else:
        tid = sample_ids[0]
        print(f"4. From-track (sample track_id={tid[:20]}...)...")
        from_track = request("POST", "/tracks/from-track", body={"track_id": tid, "topk": 5}, base=base)
        if not from_track.get("success"):
            print("   FAIL:", from_track.get("error", from_track))
        else:
            source = from_track.get("source", "?")
            tracks = from_track.get("tracks", [])
            print(f"   Source: {source}, recommendations: {len(tracks)}")
            for i, t in enumerate(tracks[:5], 1):
                name = t.get("name") or t.get("track_id", "?")
                artists = t.get("artist_names") or t.get("artists") or ""
                print(f"     {i}. {name} – {artists}")
    print("\nDone.")


if __name__ == "__main__":
    main()
