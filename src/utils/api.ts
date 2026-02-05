/**
 * API utility functions for HeartBeats backend
 */

const API_ROOT = import.meta.env.VITE_API_URL || "http://localhost:8888";
// Support either "http://host:port" or "http://host:port/api"
const API_BASE_URL = API_ROOT.endsWith("/api") ? API_ROOT : `${API_ROOT}/api`;

export interface Cluster {
  id: number;
  name: string;
  color: string;
  tags: string[];
  mean_tempo: number;
  track_count: number;
}

export interface Track {
  track_id: string;
  name: string;
  artists: string;
  cluster?: number;
  tempo?: number;
  energy?: number;
  danceability?: number;
  valence?: number;
  loudness?: number;
  distance?: number;
  rank?: number;
  // Spotify API fields
  id?: string;
  album?: string;
  album_id?: string;
  duration_ms?: number;
  preview_url?: string;
  external_urls?: string;
  images?: Array<{ url: string; height: number; width: number }>;
  release_date?: string;
  artist_names?: string;
}

export interface ClusteringResponse {
  clusters: Cluster[];
  total_tracks?: number;
  message?: string;  // Optional message when no clusters found
}

export interface ClusterTracksResponse {
  tracks: Track[];
  cluster_id: number;
  bpm: number;
}

export interface TrackDetailsResponse {
  tracks: Track[];
}

/**
 * Health check
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
}

// =============================================================================
// Spotify OAuth Functions
// =============================================================================

/**
 * Spotify user profile returned from /api/spotify/status
 */
export interface SpotifyUser {
  id: string;
  display_name: string;
  email?: string;
  product?: string;  // "premium" or "free"
  images?: Array<{ url: string }>;
}

/**
 * Response from /api/spotify/status
 */
export interface SpotifyStatus {
  connected: boolean;
  user?: SpotifyUser;
  error?: string;
}

/**
 * Check if user is authenticated with Spotify.
 * Call this on app load to see if we can skip the login screen.
 *
 * Returns { connected: true, user: {...} } if authenticated,
 * or { connected: false } if not.
 */
export async function checkSpotifyStatus(): Promise<SpotifyStatus> {
  try {
    const response = await fetch(`${API_BASE_URL}/spotify/status`);
    const data = await response.json();
    return {
      connected: data.connected || false,
      user: data.user,
      error: data.error
    };
  } catch (error) {
    console.error('Spotify status check failed:', error);
    return { connected: false, error: String(error) };
  }
}

/**
 * Get the Spotify OAuth authorization URL.
 * Frontend should redirect the user to this URL to start the login flow.
 *
 * After the user logs in on Spotify, they'll be redirected back to
 * /api/spotify/callback, which then redirects to the frontend with
 * ?spotify_connected=true or ?spotify_error=...
 */
export async function getSpotifyAuthUrl(): Promise<string | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/spotify/auth-url`);
    const data = await response.json();
    if (data.success && data.auth_url) {
      return data.auth_url;
    }
    console.error('Failed to get Spotify auth URL:', data.error);
    return null;
  } catch (error) {
    console.error('Failed to get Spotify auth URL:', error);
    return null;
  }
}

/**
 * Log out of Spotify (clear cached token on backend).
 * After this, user will need to re-authorize on next visit.
 */
export async function logoutSpotify(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/spotify/logout`, {
      method: 'POST'
    });
    const data = await response.json();
    return data.success || false;
  } catch (error) {
    console.error('Spotify logout failed:', error);
    return false;
  }
}

/**
 * Run clustering on user's tracks using recs model
 * @param paceValue - Pace value (e.g., 10.0 for 10:00 min/mile)
 * @param paceUnit - Pace unit ('min/mile' or 'min/km')
 * @param nClusters - Number of clusters (null/undefined for auto-determination)
 */
export async function runClustering(
  paceValue: number,
  paceUnit: 'min/mile' | 'min/km',
  nClusters: number | null = null
): Promise<ClusteringResponse | null> {
  try {
    // Use recs model by default - it converts pace to BPM automatically
    // Clusters tracks from the dataset (not user's library)
    const body: any = {
      use_recs_model: true,
      pace_value: paceValue,
      pace_unit: paceUnit,
    };
    // Only include n_clusters if explicitly provided (not null)
    if (nClusters !== null && nClusters !== undefined) {
      body.n_clusters = nClusters;
    }

    const response = await fetch(`${API_BASE_URL}/clusters`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      let errorMessage = 'Clustering failed';
      try {
        const error = await response.json();
        errorMessage = error.error || errorMessage;
      } catch {
        errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }

    let raw;
    try {
      raw = await response.json();
    } catch (e) {
      throw new Error(`Invalid JSON response from server. Make sure the API server is running.`);
    }

    if (!raw?.success) {
      throw new Error(raw?.error || raw?.message || "Clustering failed");
    }

    // Empty clusters array is valid (no tracks match the filter)
    if (!Array.isArray(raw?.clusters)) {
      throw new Error(raw?.error || raw?.message || "Invalid response format: clusters is not an array");
    }
    
    // If clusters is empty, include the message in the response
    if (raw?.clusters.length === 0 && raw?.message) {
      // Return clusters array but include message for frontend to display
      return { clusters: [], total_tracks: raw?.total_tracks ?? 0, message: raw.message };
    }

    const colors = ["#EAE2B7", "#FCBF49", "#F77F00", "#D62828", "#003049", "#9B59B6", "#2ECC71", "#3498DB"];

    const toNameAndTags = (meanTempo: number, meanEnergy?: number, meanDanceability?: number) => {
      const isFast = meanTempo >= 140;
      const isSlow = meanTempo < 90;
      const isHighEnergy = (meanEnergy ?? 0.5) >= 0.65;
      const isLowEnergy = (meanEnergy ?? 0.5) < 0.4;
      const isDanceable = (meanDanceability ?? 0.6) >= 0.7;
      let name = "Vibe Cluster";
      if (isFast && isHighEnergy) name = "Power Up";
      else if (isFast && !isHighEnergy) name = "Upbeat Flow";
      else if (isSlow && isLowEnergy) name = "Chill Flow";
      else if (isSlow && !isLowEnergy) name = "Deep Intensity";
      else if (isHighEnergy) name = "Energy Rush";
      else name = "Steady Burn";
      const tags: string[] = [];
      if (isFast) tags.push("high-energy");
      else if (isSlow) tags.push("slow-burn");
      else tags.push("steady");
      if (isHighEnergy) tags.push("powerful");
      if (isLowEnergy) tags.push("calm");
      if (isDanceable) tags.push("danceable");
      return { name, tags: tags.slice(0, 5) };
    };

    const clusters: Cluster[] = raw.clusters.map((c: any, idx: number) => {
      const clusterId = Number(c.cluster_id ?? c.id ?? idx);
      const meanTempo = Number(c.mean_tempo ?? 0);
      const meanEnergy = c.mean_energy !== undefined ? Number(c.mean_energy) : undefined;
      const meanDanceability = c.mean_danceability !== undefined ? Number(c.mean_danceability) : undefined;
      const backendName = c.name;
      const backendTags = Array.isArray(c.tags) ? c.tags : undefined;
      const backendColor = c.color;
      const { name: fallbackName, tags: fallbackTags } = toNameAndTags(meanTempo, meanEnergy, meanDanceability);
      return {
        id: clusterId,
        name: typeof backendName === "string" && backendName.length > 0 ? backendName : fallbackName,
        color: typeof backendColor === "string" && backendColor.length > 0 ? backendColor : (colors[clusterId % colors.length] ?? "#EAE2B7"),
        tags: backendTags !== undefined ? backendTags : fallbackTags,
        mean_tempo: meanTempo,
        track_count: Number(c.count ?? c.track_count ?? 0),
      };
    });

    return { clusters, total_tracks: Number(raw.total_tracks ?? clusters.reduce((acc, c) => acc + c.track_count, 0)) };
  } catch (error) {
    console.error('Clustering error:', error);
    return null;
  }
}

/**
 * Get KNN-matched tracks for a cluster
 */
export async function getClusterTracks(
  clusterId: number,
  bpm: number,
  topk: number = 10
): Promise<ClusterTracksResponse | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/tracks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bpm, cluster_id: clusterId, topk }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get cluster tracks');
    }

    const raw = await response.json();
    if (!raw?.success || !Array.isArray(raw?.tracks)) {
      throw new Error(raw?.error || "Failed to get cluster tracks");
    }

    return {
      tracks: raw.tracks,
      cluster_id: Number(raw.cluster_id ?? clusterId),
      bpm,
    };
  } catch (error) {
    console.error('Get cluster tracks error:', error);
    return null;
  }
}

/**
 * KNN from a track's features (for "now playing" → next recommendations).
 */
export async function getTracksFromTrack(
  trackId: string,
  clusterId?: number,
  topk: number = 10
): Promise<ClusterTracksResponse | null> {
  try {
    const body: Record<string, unknown> = { track_id: trackId, topk };
    if (clusterId != null) body.cluster_id = clusterId;
    const response = await fetch(`${API_BASE_URL}/tracks/from-track`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Failed to get tracks from track");
    }
    const raw = await response.json();
    if (!raw?.success || !Array.isArray(raw?.tracks)) throw new Error(raw?.error || "No tracks");
    return {
      tracks: raw.tracks,
      cluster_id: Number(raw.cluster_id ?? clusterId ?? 0),
      bpm: 0,
    };
  } catch (e) {
    console.error("getTracksFromTrack error:", e);
    return null;
  }
}

/**
 * Start playback on user's active device. Requires Spotify OAuth.
 */
export async function startPlayback(uris: string[]): Promise<{ success: boolean; error?: string }> {
  try {
    const r = await fetch(`${API_BASE_URL}/playback/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uris }),
    });
    const d = await r.json();
    return { success: !!d?.success, error: d?.error };
  } catch (e) {
    return { success: false, error: String(e) };
  }
}

/**
 * Add track to playback queue. Requires Spotify OAuth.
 */
export async function addToQueue(uri: string): Promise<{ success: boolean; error?: string }> {
  try {
    const r = await fetch(`${API_BASE_URL}/playback/queue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uri }),
    });
    const d = await r.json();
    return { success: !!d?.success, error: d?.error };
  } catch (e) {
    return { success: false, error: String(e) };
  }
}

/**
 * Recs model coverage: how many of the user's saved tracks are in the recs lookup.
 * Call after Spotify is connected. Returns total_saved, in_lookup, coverage_pct, by_cluster, samples.
 */
export async function getRecsCoverage(): Promise<{
  success: boolean;
  total_saved?: number;
  in_lookup?: number;
  not_in_lookup?: number;
  coverage_pct?: number;
  by_cluster?: Record<string, number>;
  sample_in_lookup?: string[];
  sample_not_in_lookup?: string[];
  error?: string;
} | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/recs/coverage`, { method: "GET" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Recs coverage failed");
    return data;
  } catch (e) {
    console.error("getRecsCoverage error:", e);
    return null;
  }
}

/**
 * Run clustering using the recs model (pre-trained clusters from track_id only).
 * No Anna's Archive needed; uses user's Spotify library and recs lookup.
 */
export async function runClusteringWithRecs(bpm?: number): Promise<ClusteringResponse | null> {
  try {
    const body: { use_recs_model: boolean; use_spotify_library: boolean; bpm?: number } = {
      use_recs_model: true,
      use_spotify_library: true,
    };
    if (bpm != null) body.bpm = bpm;
    const response = await fetch(`${API_BASE_URL}/clusters`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Recs clustering failed");
    }
    const raw = await response.json();
    if (!raw?.success || !Array.isArray(raw?.clusters))
      throw new Error(raw?.error || raw?.message || "Recs clustering failed");
    const colors = ["#EAE2B7", "#FCBF49", "#F77F00", "#D62828", "#003049", "#9B59B6", "#2ECC71", "#3498DB"];
    const clusters: Cluster[] = raw.clusters.map((c: { cluster_id?: number; count?: number; name?: string; color?: string; tags?: string[] }, idx: number) => {
      const id = Number(c.cluster_id ?? idx);
      return {
        id,
        name: c.name ?? `Vibe ${id}`,
        color: c.color ?? colors[id % colors.length],
        tags: Array.isArray(c.tags) ? c.tags : [],
        mean_tempo: 0,
        track_count: Number(c.count ?? 0),
      };
    });
    return { clusters, total_tracks: Number(raw.total_tracks ?? 0) };
  } catch (e) {
    console.error("runClusteringWithRecs error:", e);
    return null;
  }
}

/**
 * Get detailed track information from Spotify
 */
export async function getTrackDetails(trackIds: string[]): Promise<TrackDetailsResponse | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/tracks/details`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        track_ids: trackIds,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get track details');
    }

    const data: TrackDetailsResponse = await response.json();
    return data;
  } catch (error) {
    console.error('Get track details error:', error);
    return null;
  }
}
