/**
 * API utility functions for HeartBeats backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api';

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
  df: Track[];
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

    const response = await fetch(`${API_BASE_URL}/cluster`, {
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

    const data: ClusteringResponse = await response.json();
    return data;
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
    const response = await fetch(
      `${API_BASE_URL}/cluster/${clusterId}/tracks?bpm=${bpm}&topk=${topk}`
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get cluster tracks');
    }

    const data: ClusterTracksResponse = await response.json();
    return data;
  } catch (error) {
    console.error('Get cluster tracks error:', error);
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
