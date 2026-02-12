/**
 * TypeScript declarations for Apple MusicKit JS v3.
 * The SDK is loaded via <script> tag and exposes window.MusicKit.
 */

declare namespace MusicKit {
  interface Config {
    developerToken: string;
    app: {
      name: string;
      build?: string;
    };
  }

  function configure(config: Config): Promise<MusicKitInstance>;
  function getInstance(): MusicKitInstance;

  interface MusicKitInstance {
    authorize(): Promise<string>;
    unauthorize(): Promise<void>;
    isAuthorized: boolean;
    developerToken: string;
    musicUserToken: string;
    storefrontId: string;

    // Player
    player: Player;
    volume: number;
    playbackState: PlaybackStates;
    nowPlayingItem: MediaItem | null;

    // Queue
    queue: Queue;
    setQueue(descriptor: SetQueueOptions): Promise<Queue>;

    // Playback controls
    play(): Promise<void>;
    pause(): void;
    stop(): void;
    skipToNextItem(): Promise<void>;
    skipToPreviousItem(): Promise<void>;
    seekToTime(time: number): Promise<void>;
    changeToMediaAtIndex(index: number): Promise<void>;

    // API
    api: API;

    // Events
    addEventListener(name: string, callback: (...args: any[]) => void): void;
    removeEventListener(name: string, callback: (...args: any[]) => void): void;
  }

  interface Player {
    currentPlaybackTime: number;
    currentPlaybackDuration: number;
    currentPlaybackTimeRemaining: number;
    isPlaying: boolean;
    nowPlayingItem: MediaItem | null;
    playbackState: PlaybackStates;
    volume: number;
    queue: Queue;
  }

  interface MediaItem {
    id: string;
    title: string;
    artistName: string;
    albumName: string;
    artwork?: Artwork;
    duration: number; // in seconds
    playParams?: { id: string; kind: string };
    attributes?: Record<string, any>;
  }

  interface Artwork {
    url: string;
    width: number;
    height: number;
  }

  interface Queue {
    items: MediaItem[];
    currentItem: MediaItem | null;
    position: number;
    length: number;
  }

  interface SetQueueOptions {
    song?: string;
    songs?: string[];
    album?: string;
    playlist?: string;
    url?: string;
    startPlaying?: boolean;
    startWith?: number;
  }

  interface API {
    search(term: string, options?: SearchOptions): Promise<SearchResult>;
    song(id: string): Promise<MediaItem>;
    songs(ids: string[]): Promise<MediaItem[]>;
  }

  interface SearchOptions {
    types?: string[];
    limit?: number;
    offset?: number;
    l?: string; // locale
  }

  interface SearchResult {
    songs?: {
      data: SearchSong[];
      next?: string;
    };
    albums?: {
      data: any[];
    };
  }

  interface SearchSong {
    id: string;
    type: string;
    attributes: {
      name: string;
      artistName: string;
      albumName: string;
      durationInMillis: number;
      artwork: {
        url: string;
        width: number;
        height: number;
      };
      previews?: Array<{ url: string }>;
      url: string;
    };
  }

  enum PlaybackStates {
    none = 0,
    loading = 1,
    playing = 2,
    paused = 3,
    stopped = 4,
    ended = 5,
    seeking = 6,
    waiting = 8,
    stalled = 9,
    completed = 10,
  }

  namespace Events {
    const playbackStateDidChange: string;
    const nowPlayingItemDidChange: string;
    const authorizationStatusDidChange: string;
    const queueItemsDidChange: string;
    const queuePositionDidChange: string;
    const playbackTimeDidChange: string;
    const playbackDurationDidChange: string;
  }
}

interface Window {
  MusicKit: typeof MusicKit;
}
