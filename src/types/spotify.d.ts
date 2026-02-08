/**
 * TypeScript declarations for the Spotify Web Playback SDK.
 * The SDK is loaded via <script> tag and exposes window.Spotify.
 */

declare namespace Spotify {
  interface PlayerInit {
    name: string;
    getOAuthToken: (cb: (token: string) => void) => void;
    volume?: number;
  }

  interface WebPlaybackPlayer {
    device_id: string;
  }

  interface WebPlaybackState {
    paused: boolean;
    position: number;
    duration: number;
    track_window: {
      current_track: WebPlaybackTrack;
      previous_tracks: WebPlaybackTrack[];
      next_tracks: WebPlaybackTrack[];
    };
    repeat_mode: number;
    shuffle: boolean;
  }

  interface WebPlaybackTrack {
    uri: string;
    id: string | null;
    type: 'track' | 'episode' | 'ad';
    media_type: 'audio' | 'video';
    name: string;
    is_playable: boolean;
    album: {
      uri: string;
      name: string;
      images: Array<{ url: string; height: number; width: number }>;
    };
    artists: Array<{ uri: string; name: string }>;
    duration_ms: number;
  }

  interface WebPlaybackError {
    message: string;
  }

  type ErrorTypes =
    | 'initialization_error'
    | 'authentication_error'
    | 'account_error'
    | 'playback_error';

  class Player {
    constructor(options: PlayerInit);

    connect(): Promise<boolean>;
    disconnect(): void;

    addListener(
      event: 'ready',
      callback: (data: WebPlaybackPlayer) => void
    ): void;
    addListener(
      event: 'not_ready',
      callback: (data: WebPlaybackPlayer) => void
    ): void;
    addListener(
      event: 'player_state_changed',
      callback: (state: WebPlaybackState | null) => void
    ): void;
    addListener(
      event: ErrorTypes,
      callback: (error: WebPlaybackError) => void
    ): void;

    removeListener(event: string): void;

    getCurrentState(): Promise<WebPlaybackState | null>;
    setName(name: string): Promise<void>;
    getVolume(): Promise<number>;
    setVolume(volume: number): Promise<void>;
    pause(): Promise<void>;
    resume(): Promise<void>;
    togglePlay(): Promise<void>;
    seek(position_ms: number): Promise<void>;
    previousTrack(): Promise<void>;
    nextTrack(): Promise<void>;
    activateElement(): Promise<void>;
  }
}

interface Window {
  onSpotifyWebPlaybackSDKReady: (() => void) | undefined;
  Spotify: {
    Player: typeof Spotify.Player;
  } | undefined;
}
