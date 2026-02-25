/**
 * API utility functions for HeartBeats backend
 */

const _apiRoot = import.meta.env.VITE_API_URL ?? "";
const API_BASE_URL = _apiRoot
  ? (_apiRoot.endsWith("/api") ? _apiRoot : `${_apiRoot}/api`)
  : "/api";

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
  // Metadata fields
  id?: string;
  album?: string;
  album_id?: string;
  duration_ms?: number;
  images?: Array<{ url: string; height: number; width: number }>;
  release_date?: string;
  artist_names?: string;
  // Apple Music
  apple_music_id?: string;
}

export interface ClusteringResponse {
  clusters: Cluster[];
  total_tracks?: number;
  message?: string;
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
 * Run clustering on tracks
 */
export async function runClustering(
  paceValue: number,
  paceUnit: 'min/mile' | 'min/km',
): Promise<ClusteringResponse | null> {
  try {
    const body: any = {
      pace_value: paceValue,
      pace_unit: paceUnit,
    };

    const response = await fetch(`${API_BASE_URL}/clusters`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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

    if (!Array.isArray(raw?.clusters)) {
      throw new Error(raw?.error || raw?.message || "Invalid response format: clusters is not an array");
    }

    if (raw?.clusters.length === 0 && raw?.message) {
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
 * Get curated artist suggestions for a vibe.
 */
export async function getVibeArtists(vibeId: number): Promise<string[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/vibes/${vibeId}/artists`);
    if (!response.ok) return [];
    const data = await response.json();
    return Array.isArray(data.artists) ? data.artists : [];
  } catch {
    return [];
  }
}

/**
 * Get KNN-matched tracks for a cluster, optionally filtered by one or more artists.
 */
export async function getClusterTracks(
  clusterId: number,
  bpm: number,
  topk: number = 10,
  artistNames?: string[],
): Promise<ClusterTracksResponse | null> {
  try {
    const body: Record<string, unknown> = { bpm, cluster_id: clusterId, topk };
    if (artistNames?.length) body.artist_names = artistNames;
    const response = await fetch(`${API_BASE_URL}/tracks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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
  topk: number = 10,
  excludeIds?: string[],
): Promise<ClusterTracksResponse | null> {
  try {
    const body: Record<string, unknown> = { track_id: trackId, topk };
    if (clusterId != null) body.cluster_id = clusterId;
    if (excludeIds?.length) body.exclude_ids = excludeIds;
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
 * Get detailed track information
 */
export async function getTrackDetails(trackIds: string[]): Promise<TrackDetailsResponse | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/tracks/details`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track_ids: trackIds }),
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

// =============================================================================
// Apple Music ID Resolution
// =============================================================================

// In-memory cache for resolved Apple Music IDs
const _appleMusicIdCache = new Map<string, string>();

/**
 * Resolve a track name + artist to an Apple Music catalog ID.
 * Uses the MusicKit JS search API (client-side, no backend needed).
 */
export async function resolveAppleMusicId(
  trackName: string,
  artistName: string
): Promise<string | null> {
  const cacheKey = `${trackName}|||${artistName}`.toLowerCase();
  if (_appleMusicIdCache.has(cacheKey)) {
    return _appleMusicIdCache.get(cacheKey)!;
  }

  try {
    const music = window.MusicKit?.getInstance?.();
    if (!music) {
      console.warn('MusicKit not initialized');
      return null;
    }

    const query = `${trackName} ${artistName}`.trim();
    const storefront = music.storefrontId || 'us';

    let songs: any[] | null = null;

    // Try MusicKit JS v3 api.music() first
    try {
      if (typeof (music as any).api?.music === 'function') {
        const result = await (music as any).api.music(
          `/v1/catalog/${storefront}/search`,
          { term: query, types: 'songs', limit: 5 }
        );
        songs = result?.data?.results?.songs?.data ?? null;
      }
    } catch (e) {
      console.warn('MusicKit api.music() failed, trying direct fetch:', e);
    }

    // Fallback: direct Apple Music API fetch with developer token
    if (!songs) {
      const devToken = import.meta.env.VITE_APPLE_MUSIC_DEVELOPER_TOKEN;
      const userToken = (music as any).musicUserToken || '';
      if (devToken) {
        const params = new URLSearchParams({ term: query, types: 'songs', limit: '5' });
        const headers: Record<string, string> = { Authorization: `Bearer ${devToken}` };
        if (userToken) headers['Music-User-Token'] = userToken;
        const resp = await fetch(
          `https://api.music.apple.com/v1/catalog/${storefront}/search?${params}`,
          { headers }
        );
        if (resp.ok) {
          const json = await resp.json();
          songs = json?.results?.songs?.data ?? null;
        }
      }
    }

    if (!songs || songs.length === 0) return null;

    // Try exact match first
    const normalizedName = trackName.toLowerCase().trim();
    const normalizedArtist = artistName.toLowerCase().trim();

    for (const song of songs) {
      const sName = (song.attributes?.name || '').toLowerCase().trim();
      const sArtist = (song.attributes?.artistName || '').toLowerCase().trim();
      if (sName === normalizedName && sArtist.includes(normalizedArtist)) {
        _appleMusicIdCache.set(cacheKey, song.id);
        return song.id;
      }
    }

    // Fuzzy: name contains match
    for (const song of songs) {
      const sName = (song.attributes?.name || '').toLowerCase().trim();
      if (sName.includes(normalizedName) || normalizedName.includes(sName)) {
        _appleMusicIdCache.set(cacheKey, song.id);
        return song.id;
      }
    }

    // Fallback: first result
    const firstId = songs[0].id;
    _appleMusicIdCache.set(cacheKey, firstId);
    return firstId;
  } catch (error) {
    console.warn('Apple Music search failed for:', trackName, '-', artistName, error);
    return null;
  }
}

/**
 * Resolve Apple Music IDs for an array of tracks.
 * Batches requests (5 at a time) to avoid rate limits.
 * Updates each track's apple_music_id field in-place and returns them.
 */
export async function resolveAppleMusicIds(tracks: Track[]): Promise<Track[]> {
  const BATCH_SIZE = 5;

  for (let i = 0; i < tracks.length; i += BATCH_SIZE) {
    const batch = tracks.slice(i, i + BATCH_SIZE);
    await Promise.all(
      batch.map(async (track) => {
        if (track.apple_music_id) return;

        const artistName = track.artist_names || track.artists || '';
        const trackName = track.name || '';
        if (!trackName) return;

        const id = await resolveAppleMusicId(trackName, artistName);
        if (id) {
          track.apple_music_id = id;
        }
      })
    );
  }

  return tracks;
}
