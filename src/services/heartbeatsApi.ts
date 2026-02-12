/**
 * HeartBeats API Service
 * Handles communication with the backend API
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8888';

export interface Cluster {
  cluster_id: number;
  count: number;
  mean_tempo: number;
  mean_energy: number;
  mean_danceability: number;
}

export interface Track {
  track_id: string;
  name: string;
  artists: string;
  cluster: number;
  tempo: number;
  energy: number;
  danceability: number;
  valence: number;
  loudness: number;
  distance: number;
  rank: number;
  // Metadata
  id?: string;
  artist_names?: string;
  album?: string;
  album_id?: string;
  duration_ms?: number;
  images?: Array<{ url: string; height: number; width: number }>;
  release_date?: string;
  // Apple Music
  apple_music_id?: string;
}

export interface ApiResponse<T> {
  success: boolean;
  error?: string;
  data?: T;
}

class HeartBeatsApi {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        return {
          success: false,
          error: data.error || `HTTP error! status: ${response.status}`,
        };
      }

      const { success, ...rest } = data;
      return {
        success: true,
        data: rest as T,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  async getClusters(csvPath?: string, nClusters: number = 4): Promise<ApiResponse<{ clusters: Cluster[]; total_tracks: number }>> {
    const response = await this.request<{ clusters: Cluster[]; total_tracks: number }>('/api/clusters', {
      method: 'POST',
      body: JSON.stringify({
        csv_path: csvPath,
        n_clusters: nClusters,
      }),
    });
    return response;
  }

  async getTracks(
    bpm: number,
    clusterId?: number,
    topk: number = 10
  ): Promise<ApiResponse<{ cluster_id: number; tracks: Track[]; count: number }>> {
    const response = await this.request<{ cluster_id: number; tracks: Track[]; count: number }>('/api/tracks', {
      method: 'POST',
      body: JSON.stringify({
        bpm,
        cluster_id: clusterId,
        topk,
      }),
    });
    return response;
  }

  async getTrackDetails(trackIds: string[]): Promise<ApiResponse<{ tracks: Track[]; count: number }>> {
    const response = await this.request<{ tracks: Track[]; count: number }>('/api/tracks/details', {
      method: 'POST',
      body: JSON.stringify({
        track_ids: trackIds,
      }),
    });
    return response;
  }

  async getClusterTracks(
    bpm: number,
    clusterId?: number,
    topk: number = 10
  ): Promise<ApiResponse<{ cluster_id: number; tracks: Track[]; count: number }>> {
    return this.request('/api/cluster/tracks', {
      method: 'POST',
      body: JSON.stringify({
        bpm,
        cluster_id: clusterId,
        topk,
      }),
    });
  }
}

export const heartbeatsApi = new HeartBeatsApi();
