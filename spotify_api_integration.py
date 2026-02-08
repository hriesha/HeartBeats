"""
Spotify API Integration for HeartBeats

This module handles:
1. User Spotify authentication and authorization
2. User login/connection to Spotify
3. Retrieving track and album information from Spotify Web API
   (for tracks returned by the HeartBeats matching algorithm)

Expected workflow:
1. User connects/authorizes their Spotify account
2. User runs the HeartBeats matching algorithm (gets track_ids as output)
3. This module retrieves full track/album details for those track_ids
4. (Future) Display this information in the Swift UI

Input format (from algorithm):
    - List of Spotify track IDs (strings), e.g., ["4uLU6hMCjMI75M1A2tKUQC", "7qiZfU4dY1lWllzX7mPKB3"]
    - Or a pandas DataFrame with a 'track_id' column

Output format:
    - Dictionary/JSON with track details (name, artists, album, preview_url, etc.)
    - Or list of track dictionaries
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException

if TYPE_CHECKING:
    import pandas as pd

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger("heartbeats_spotify")


class SpotifyIntegration:
    """
    Handles Spotify authentication and API interactions for HeartBeats.
    """

    # Required scopes for HeartBeats functionality
    # user-library-read: Read user's saved tracks (for algorithm input)
    # user-read-private: Access user's email and subscription info
    # user-read-playback-state: Read user's playback state
    # user-modify-playback-state: Control playback (play, pause, skip)
    DEFAULT_SCOPES = "user-library-read user-read-private user-read-playback-state user-modify-playback-state streaming"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        scopes: Optional[str] = None,
        cache_path: str = ".cache-heartbeats"
    ):
        """
        Initialize Spotify integration.

        Args:
            client_id: Spotify Client ID (defaults to SPOTIPY_CLIENT_ID env var)
            client_secret: Spotify Client Secret (defaults to SPOTIPY_CLIENT_SECRET env var)
            redirect_uri: OAuth redirect URI (defaults to SPOTIPY_REDIRECT_URI env var)
            scopes: Spotify API scopes (defaults to DEFAULT_SCOPES)
            cache_path: Path to cache OAuth tokens
        """
        self.client_id = client_id or os.getenv("SPOTIPY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIPY_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
        self.scopes = scopes or self.DEFAULT_SCOPES
        self.cache_path = cache_path

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify credentials not found. Please set SPOTIPY_CLIENT_ID and "
                "SPOTIPY_CLIENT_SECRET in your .env file or pass them as arguments."
            )

        # Initialize Spotify OAuth manager
        self.auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=self.scopes,
            cache_path=self.cache_path,
            show_dialog=True  # Force re-authentication if needed
        )

        # Initialize Spotify client (will prompt for auth on first use)
        self.sp: Optional[spotipy.Spotify] = None
        self._current_user: Optional[Dict[str, Any]] = None

    def connect(self) -> bool:
        """
        Connect user's Spotify account to HeartBeats.
        This initiates the OAuth flow if not already authenticated.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Initialize Spotify client (this will trigger OAuth if needed)
            self.sp = spotipy.Spotify(auth_manager=self.auth_manager)

            # Verify connection by getting current user
            self._current_user = self.sp.current_user()

            user_name = self._current_user['display_name']
            user_id = self._current_user['id']
            log.info("✅ Successfully connected to Spotify as: %s (%s)", user_name, user_id)
            return True

        except SpotifyException as e:
            log.error("Spotify API error during connection: %s (Status: %s)", e.msg, e.http_status)
            return False
        except (AttributeError, KeyError, TypeError) as e:
            log.error("Failed to connect to Spotify: %r", e)
            return False

    def is_connected(self) -> bool:
        """
        Check if user is currently connected/authenticated.

        Returns:
            True if connected, False otherwise
        """
        if not self.sp:
            return False

        try:
            self.sp.current_user()
            return True
        except (SpotifyException, AttributeError, TypeError):
            return False

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current user's Spotify profile information.

        Returns:
            Dictionary with user info (display_name, id, email, etc.) or None if not connected
        """
        if not self.sp:
            log.warning("Not connected to Spotify. Call connect() first.")
            return None

        try:
            if not self._current_user:
                self._current_user = self.sp.current_user()
            return self._current_user
        except (SpotifyException, AttributeError, KeyError) as e:
            log.error("Failed to get user info: %r", e)
            return None

    def get_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a single track.

        Args:
            track_id: Spotify track ID (e.g., "4uLU6hMCjMI75M1A2tKUQC")

        Returns:
            Dictionary with track details including:
            - id, name, artists, album, preview_url, external_urls, duration_ms, etc.
            Returns None if track not found or error occurred
        """
        if not self.sp:
            log.warning("Not connected to Spotify. Call connect() first.")
            return None

        try:
            track = self.sp.track(track_id)
            track_name = track['name']
            artist_names = ', '.join(a['name'] for a in track['artists'])
            log.debug("Retrieved track: %s by %s", track_name, artist_names)
            return track
        except SpotifyException as e:
            log.error("Failed to get track %s: %s", track_id, e.msg)
            return None
        except (AttributeError, KeyError, TypeError) as e:
            log.error("Error retrieving track %s: %r", track_id, e)
            return None

    def get_tracks(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get detailed information about multiple tracks (up to 50 at a time).

        Args:
            track_ids: List of Spotify track IDs

        Returns:
            List of track dictionaries. Tracks that couldn't be retrieved will be None.
        """
        if not self.sp:
            log.warning("Not connected to Spotify. Call connect() first.")
            return []

        if not track_ids:
            return []

        # Spotify API allows up to 50 track IDs per request
        results = []
        batch_size = 50

        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i + batch_size]
            try:
                tracks_response = self.sp.tracks(batch)
                tracks = tracks_response.get("tracks", [])
                results.extend(tracks)  # May include None values for invalid IDs
                successful = len([t for t in tracks if t])
                log.debug("Retrieved batch of %d tracks (%d successful)", len(batch), successful)
            except SpotifyException as e:
                log.error("Failed to get tracks batch: %s", e.msg)
                results.extend([None] * len(batch))
            except (AttributeError, KeyError, TypeError) as e:
                log.error("Error retrieving tracks batch: %r", e)
                results.extend([None] * len(batch))

        return results

    def get_album(self, album_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an album.

        Args:
            album_id: Spotify album ID

        Returns:
            Dictionary with album details including:
            - id, name, artists, release_date, images, external_urls, etc.
            Returns None if album not found or error occurred
        """
        if not self.sp:
            log.warning("Not connected to Spotify. Call connect() first.")
            return None

        try:
            album = self.sp.album(album_id)
            album_name = album['name']
            artist_names = ', '.join(a['name'] for a in album['artists'])
            log.debug("Retrieved album: %s by %s", album_name, artist_names)
            return album
        except SpotifyException as e:
            log.error("Failed to get album %s: %s", album_id, e.msg)
            return None
        except (AttributeError, KeyError, TypeError) as e:
            log.error("Error retrieving album %s: %r", album_id, e)
            return None

    def get_albums(self, album_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get detailed information about multiple albums (up to 20 at a time).

        Args:
            album_ids: List of Spotify album IDs

        Returns:
            List of album dictionaries. Albums that couldn't be retrieved will be None.
        """
        if not self.sp:
            log.warning("Not connected to Spotify. Call connect() first.")
            return []

        if not album_ids:
            return []

        # Spotify API allows up to 20 album IDs per request
        results = []
        batch_size = 20

        for i in range(0, len(album_ids), batch_size):
            batch = album_ids[i:i + batch_size]
            try:
                albums_response = self.sp.albums(batch)
                albums = albums_response.get("albums", [])
                results.extend(albums)  # May include None values for invalid IDs
                successful = len([a for a in albums if a])
                log.debug("Retrieved batch of %d albums (%d successful)", len(batch), successful)
            except SpotifyException as e:
                log.error("Failed to get albums batch: %s", e.msg)
                results.extend([None] * len(batch))
            except (AttributeError, KeyError, TypeError) as e:
                log.error("Error retrieving albums batch: %r", e)
                results.extend([None] * len(batch))

        return results

    def get_tracks_from_algorithm_output(
        self,
        algorithm_output: Union[List[str], "pd.DataFrame"],
        track_id_column: str = "track_id"
    ) -> List[Dict[str, Any]]:
        """
        Get track information from HeartBeats algorithm output.

        This is a convenience method that accepts either:
        - A list of track IDs (strings)
        - A pandas DataFrame with a track_id column

        Args:
            algorithm_output: Either a list of track IDs or a DataFrame with track IDs
            track_id_column: Name of the column containing track IDs (if DataFrame provided)

        Returns:
            List of track dictionaries with full track information
        """
        if not self.sp:
            log.warning("Not connected to Spotify. Call connect() first.")
            return []

        # Handle DataFrame input
        if hasattr(algorithm_output, 'iterrows'):  # It's a pandas DataFrame
            import pandas as pd
            if track_id_column not in algorithm_output.columns:
                raise ValueError(f"DataFrame must have a '{track_id_column}' column")
            track_ids = algorithm_output[track_id_column].dropna().tolist()
        elif isinstance(algorithm_output, list):
            track_ids = algorithm_output
        else:
            raise TypeError("algorithm_output must be a list of track IDs or a pandas DataFrame")

        if not track_ids:
            log.warning("No track IDs provided")
            return []

        # Remove duplicates while preserving order
        track_ids = list(dict.fromkeys(track_ids))

        log.info("Retrieving %d tracks from Spotify...", len(track_ids))
        tracks = self.get_tracks(track_ids)

        # Filter out None values (failed retrievals)
        valid_tracks = [t for t in tracks if t is not None]
        log.info("Successfully retrieved %d/%d tracks", len(valid_tracks), len(track_ids))

        return valid_tracks

    def format_track_for_display(self, track: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a track dictionary into a simplified structure for UI display.

        Args:
            track: Full track dictionary from Spotify API

        Returns:
            Simplified dictionary with essential fields for display
        """
        if not track:
            return {}

        return {
            "id": track.get("id"),
            "name": track.get("name"),
            "artists": [artist["name"] for artist in track.get("artists", [])],
            "artist_names": ", ".join([artist["name"] for artist in track.get("artists", [])]),
            "album": track.get("album", {}).get("name"),
            "album_id": track.get("album", {}).get("id"),
            "duration_ms": track.get("duration_ms"),
            "preview_url": track.get("preview_url"),
            "external_urls": track.get("external_urls", {}).get("spotify"),
            "images": track.get("album", {}).get("images", []),
            "release_date": track.get("album", {}).get("release_date"),
        }


def main():
    """
    Example usage of the SpotifyIntegration class.
    This demonstrates the workflow for connecting and retrieving tracks.
    """
    print("=" * 60)
    print("HeartBeats - Spotify API Integration Demo")
    print("=" * 60)

    # Initialize integration
    try:
        spotify = SpotifyIntegration()
        print("\n1. Initializing Spotify integration...")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease ensure you have set up your .env file with:")
        print("  SPOTIPY_CLIENT_ID=your_client_id")
        print("  SPOTIPY_CLIENT_SECRET=your_client_secret")
        print("  SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback")
        return

    # Connect/Authenticate
    print("\n2. Connecting to Spotify...")
    if not spotify.connect():
        print("❌ Failed to connect to Spotify")
        return

    # Get user info
    print("\n3. Fetching user information...")
    user_info = spotify.get_user_info()
    if user_info:
        print(f"   User: {user_info.get('display_name')}")
        print(f"   Email: {user_info.get('email', 'N/A')}")
        print(f"   Plan: {user_info.get('product', 'N/A')}")

    # Example: Get a single track (Rick Astley - Never Gonna Give You Up)
    print("\n4. Testing track retrieval...")
    test_track_id = "4uLU6hMCjMI75M1A2tKUQC"
    track = spotify.get_track(test_track_id)
    if track:
        formatted = spotify.format_track_for_display(track)
        print(f"   Track: {formatted['name']}")
        print(f"   Artist: {formatted['artist_names']}")
        print(f"   Album: {formatted['album']}")
        print(f"   Spotify URL: {formatted['external_urls']}")

    # Example: Get multiple tracks (simulating algorithm output)
    print("\n5. Testing batch track retrieval (simulating algorithm output)...")
    example_track_ids = [
        "4uLU6hMCjMI75M1A2tKUQC",  # Rick Astley - Never Gonna Give You Up
        "3n3Ppam7vgaVa1iaRUc9Lp",  # The Beatles - Here Comes The Sun
    ]
    tracks = spotify.get_tracks_from_algorithm_output(example_track_ids)
    print(f"   Retrieved {len(tracks)} tracks:")
    for i, track in enumerate(tracks, 1):
        formatted = spotify.format_track_for_display(track)
        print(f"   {i}. {formatted['name']} - {formatted['artist_names']}")

    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Integrate this with your HeartBeats matching algorithm")
    print("2. Pass algorithm output (track IDs) to get_tracks_from_algorithm_output()")
    print("3. Use format_track_for_display() to prepare data for Swift UI")
    print("=" * 60)


if __name__ == "__main__":
    main()
