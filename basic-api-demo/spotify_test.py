# spotify_test.py
import os, traceback
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException

load_dotenv()  # make sure .env has SPOTIPY_* and 127.0.0.1 redirect

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="user-library-read"
))

try:
    me = sp.current_user()
    print("✅ Logged in as:", me["display_name"], f"({me['id']})")
except Exception as e:
    print("current_user failed:", repr(e))
    raise

# A known public track id (Rick Astley)
TEST_ID = "4uLU6hMCjMI75M1A2tKUQC"

# 1) simple track read (should succeed)
try:
    t = sp.track(TEST_ID)
    print("🎯 track() OK:", t["name"], "by", ", ".join(a["name"] for a in t["artists"]))
except Exception as e:
    print("track() failed:", repr(e))

# 2) audio_features for single id
try:
    f = sp.audio_features([TEST_ID])[0]
    print("🎵 audio_features() OK:", {k: f[k] for k in ("tempo", "energy", "danceability")})
except SpotifyException as e:
    print("audio_features() SpotifyException:")
    print("  http_status:", e.http_status)
    print("  code:", e.code)
    print("  msg:", e.msg)
    print("  headers:", getattr(e, "headers", None))
except Exception as e:
    print("audio_features() generic error:", repr(e))

# 3) search (extra sanity)
try:
    r = sp.search(q="never gonna give you up", type="track", limit=1)
    print("🔎 search() OK: found", r["tracks"]["items"][0]["name"])
except Exception as e:
    print("search() failed:", repr(e))