/**
 * API utility functions for HeartBeats backend
 */

const API_ROOT = import.meta.env.VITE_API_URL || "http://localhost:5001";
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

/**
 * Connect to Spotify
 */
export async function connectSpotify(): Promise<{ success: boolean; user?: any; error?: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/spotify/connect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Spotify connection error:', error);
    return { success: false, error: String(error) };
  }
}

/**
 * Run clustering on user's tracks
 * @param bpm - Target BPM (clustering will be optimized for this BPM)
 * @param nClusters - Number of clusters (null/undefined for auto-determination)
 */
export async function runClustering(bpm: number, nClusters: number | null = null): Promise<ClusteringResponse | null> {
  try {
    const body: any = { bpm };
    // Only include n_clusters if explicitly provided (not null)
    // This allows backend to auto-determine optimal k
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
      const error = await response.json();
      throw new Error(error.error || 'Clustering failed');
    }

    const raw = await response.json();
    if (!raw?.success || !Array.isArray(raw?.clusters)) {
      throw new Error(raw?.error || "Clustering failed");
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
