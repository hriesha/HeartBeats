#!/usr/bin/env python3
"""Test API connection and endpoints."""

import requests
import sys

API_URL = "http://localhost:5001/api"

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.ok
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed: API server is not running")
        print("   Start it with: python3 api/heartbeats_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_clusters():
    """Test clusters endpoint with pace."""
    try:
        data = {
            "use_recs_model": True,
            "use_spotify_library": True,
            "pace_value": 10.0,
            "pace_unit": "min/mile"
        }
        response = requests.post(f"{API_URL}/clusters", json=data, timeout=30)
        print(f"\n✅ Clusters endpoint: {response.status_code}")
        if response.ok:
            result = response.json()
            print(f"   Success: {result.get('success')}")
            print(f"   Clusters: {len(result.get('clusters', []))}")
            print(f"   Total tracks: {result.get('total_tracks')}")
            print(f"   Filtered tracks: {result.get('filtered_tracks')}")
            if result.get('error'):
                print(f"   Error: {result.get('error')}")
        else:
            print(f"   Error response: {response.text[:200]}")
        return response.ok
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing API connection...")
    print(f"API URL: {API_URL}\n")

    if not test_health():
        sys.exit(1)

    print("\n" + "="*50)
    test_clusters()
