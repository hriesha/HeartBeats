/**
 * React Hook for Spotify Web Playback SDK with Volume-Based Crossfade
 *
 * Provides in-browser full-track playback via the Spotify Web Playback SDK.
 * Implements a "breathing" crossfade: fade volume down at end of track,
 * switch to next track, fade volume back up.
 *
 * The SDK only allows ONE active playback stream, so true dual-stream
 * crossfade is impossible. This volume-based approach creates a smooth
 * transition that sounds natural.
 *
 * Exposes the same interface as useCrossfade for easy component swap.
 *
 * Requires: Spotify Premium account, streaming scope in OAuth token.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { getSpotifyToken } from '../utils/api';

export interface UseSpotifyPlayerOptions {
  crossfadeDuration?: number;  // Total crossfade duration in ms (default: 5000)
  onTrackEnd?: () => void;     // Called when track is about to end (for auto-advance)
  onError?: (error: string) => void;
  onReady?: (deviceId: string) => void;
}

export interface UseSpotifyPlayerReturn {
  // Playback controls (same shape as UseCrossfadeReturn)
  play: (uri: string) => Promise<void>;
  pause: () => void;
  resume: () => Promise<void>;
  skipTo: (uri: string) => Promise<void>;
  crossfadeTo: (uri: string) => Promise<void>;
  seek: (time: number) => void;

  // State (same shape as UseCrossfadeReturn)
  isPlaying: boolean;
  isCrossfading: boolean;
  currentTime: number;   // in seconds
  duration: number;      // in seconds
  volume: number;

  // Volume control
  setVolume: (volume: number) => void;

  // Configuration
  setCrossfadeDuration: (ms: number) => void;

  // SDK-specific state
  isReady: boolean;
  deviceId: string | null;
  sdkError: string | null;

  // Cleanup
  cleanup: () => void;
}

export function useSpotifyPlayer(
  options: UseSpotifyPlayerOptions = {}
): UseSpotifyPlayerReturn {
  const {
    crossfadeDuration: initialCrossfadeDuration = 5000,
    onTrackEnd,
    onError,
    onReady,
  } = options;

  // ---- State ----
  const [isPlaying, setIsPlaying] = useState(false);
  const [isCrossfading, setIsCrossfading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolumeState] = useState(1);
  const [isReady, setIsReady] = useState(false);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [sdkError, setSdkError] = useState<string | null>(null);

  // ---- Refs ----
  const playerRef = useRef<Spotify.Player | null>(null);
  const deviceIdRef = useRef<string | null>(null);
  const onTrackEndRef = useRef(onTrackEnd);
  const onErrorRef = useRef(onError);
  const onReadyRef = useRef(onReady);
  const masterVolumeRef = useRef(1);
  const isCrossfadingRef = useRef(false);
  const crossfadeTimerRef = useRef<number | null>(null);
  const positionIntervalRef = useRef<number | null>(null);
  const trackEndCheckRef = useRef<number | null>(null);
  const tokenRefreshTimerRef = useRef<number | null>(null);
  const isConnectedRef = useRef(false);
  // Crossfade timing: 60% fade-out, 40% fade-in
  const fadeOutDurationRef = useRef(Math.round(initialCrossfadeDuration * 0.6));
  const fadeInDurationRef = useRef(Math.round(initialCrossfadeDuration * 0.4));

  // Keep callback refs current
  useEffect(() => { onTrackEndRef.current = onTrackEnd; }, [onTrackEnd]);
  useEffect(() => { onErrorRef.current = onError; }, [onError]);
  useEffect(() => { onReadyRef.current = onReady; }, [onReady]);

  // ---- Token fetching ----
  const fetchToken = useCallback(async (): Promise<string | null> => {
    const res = await getSpotifyToken();
    if (res.success && res.access_token) {
      return res.access_token;
    }
    return null;
  }, []);

  // ---- Position tracking ----
  const stopPositionUpdates = useCallback(() => {
    if (positionIntervalRef.current !== null) {
      clearInterval(positionIntervalRef.current);
      positionIntervalRef.current = null;
    }
  }, []);

  const startPositionUpdates = useCallback(() => {
    stopPositionUpdates();
    positionIntervalRef.current = window.setInterval(async () => {
      const state = await playerRef.current?.getCurrentState();
      if (state) {
        setCurrentTime(state.position / 1000);
        setDuration(state.duration / 1000);
        setIsPlaying(!state.paused);
      }
    }, 250);
  }, [stopPositionUpdates]);

  // ---- Track end monitoring ----
  const stopTrackEndMonitoring = useCallback(() => {
    if (trackEndCheckRef.current !== null) {
      clearInterval(trackEndCheckRef.current);
      trackEndCheckRef.current = null;
    }
  }, []);

  const startTrackEndMonitoring = useCallback(() => {
    stopTrackEndMonitoring();
    trackEndCheckRef.current = window.setInterval(async () => {
      if (isCrossfadingRef.current) return;

      const state = await playerRef.current?.getCurrentState();
      if (!state || state.paused) return;

      const remainingMs = state.duration - state.position;
      const fadeOutMs = fadeOutDurationRef.current;

      // Trigger when remaining time equals fade-out duration
      if (remainingMs <= fadeOutMs && remainingMs > 0) {
        stopTrackEndMonitoring();
        onTrackEndRef.current?.();
      }
    }, 200);
  }, [stopTrackEndMonitoring]);

  // ---- SDK Initialization ----
  useEffect(() => {
    let cancelled = false;

    const initPlayer = async () => {
      if (!window.Spotify) {
        window.onSpotifyWebPlaybackSDKReady = () => {
          if (!cancelled) initPlayer();
        };
        return;
      }

      const token = await fetchToken();
      if (!token || cancelled) {
        setSdkError('Failed to get Spotify token');
        return;
      }

      const player = new window.Spotify.Player({
        name: 'HeartBeats Player',
        getOAuthToken: async (cb) => {
          const freshToken = await fetchToken();
          if (freshToken) {
            cb(freshToken);
          } else {
            onErrorRef.current?.('Token refresh failed');
          }
        },
        volume: masterVolumeRef.current,
      });

      // Error handlers
      player.addListener('initialization_error', ({ message }) => {
        setSdkError(`Init error: ${message}`);
        onErrorRef.current?.(message);
      });
      player.addListener('authentication_error', ({ message }) => {
        setSdkError(`Auth error: ${message}`);
        onErrorRef.current?.(message);
      });
      player.addListener('account_error', ({ message }) => {
        setSdkError(`Account error (Premium required): ${message}`);
        onErrorRef.current?.(message);
      });
      player.addListener('playback_error', ({ message }) => {
        console.warn('Playback error:', message);
      });

      // Ready
      player.addListener('ready', ({ device_id }) => {
        if (cancelled) return;
        console.log('HeartBeats Player ready, device ID:', device_id);
        deviceIdRef.current = device_id;
        setDeviceId(device_id);
        setIsReady(true);
        setSdkError(null);
        isConnectedRef.current = true;
        onReadyRef.current?.(device_id);
      });

      // Not ready
      player.addListener('not_ready', () => {
        setIsReady(false);
        isConnectedRef.current = false;
      });

      // State changes
      player.addListener('player_state_changed', (state) => {
        if (!state) {
          setIsPlaying(false);
          return;
        }
        setIsPlaying(!state.paused);
        setCurrentTime(state.position / 1000);
        setDuration(state.duration / 1000);
      });

      const connected = await player.connect();
      if (!connected && !cancelled) {
        setSdkError('Failed to connect to Spotify');
      }

      playerRef.current = player;
    };

    if (window.Spotify) {
      initPlayer();
    } else {
      window.onSpotifyWebPlaybackSDKReady = () => {
        if (!cancelled) initPlayer();
      };
    }

    return () => {
      cancelled = true;
      stopPositionUpdates();
      stopTrackEndMonitoring();
      if (crossfadeTimerRef.current) cancelAnimationFrame(crossfadeTimerRef.current);
      if (tokenRefreshTimerRef.current) clearTimeout(tokenRefreshTimerRef.current);
      playerRef.current?.disconnect();
      playerRef.current = null;
      isConnectedRef.current = false;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- Exponential fade curves (matching audioCrossfade.ts) ----
  const fadeVolumeDown = useCallback((): Promise<void> => {
    return new Promise((resolve) => {
      const player = playerRef.current;
      if (!player) { resolve(); return; }

      const startTime = Date.now();
      const fadeMs = fadeOutDurationRef.current;
      const startVolume = masterVolumeRef.current;

      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / fadeMs, 1);
        const vol = Math.pow(1 - progress, 2) * startVolume;
        player.setVolume(Math.max(0, vol));

        if (progress < 1) {
          crossfadeTimerRef.current = requestAnimationFrame(animate);
        } else {
          crossfadeTimerRef.current = null;
          resolve();
        }
      };

      crossfadeTimerRef.current = requestAnimationFrame(animate);
    });
  }, []);

  const fadeVolumeUp = useCallback((): Promise<void> => {
    return new Promise((resolve) => {
      const player = playerRef.current;
      if (!player) { resolve(); return; }

      const startTime = Date.now();
      const fadeMs = fadeInDurationRef.current;
      const targetVolume = masterVolumeRef.current;

      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / fadeMs, 1);
        const vol = Math.pow(progress, 2) * targetVolume;
        player.setVolume(Math.min(targetVolume, vol));

        if (progress < 1) {
          crossfadeTimerRef.current = requestAnimationFrame(animate);
        } else {
          crossfadeTimerRef.current = null;
          resolve();
        }
      };

      crossfadeTimerRef.current = requestAnimationFrame(animate);
    });
  }, []);

  // ---- Start playback via Spotify Web API targeting our SDK device ----
  const startPlaybackOnDevice = useCallback(async (uri: string) => {
    const did = deviceIdRef.current;
    if (!did) throw new Error('Player not ready');

    const token = await fetchToken();
    if (!token) throw new Error('Failed to get token for playback');

    const response = await fetch(
      `https://api.spotify.com/v1/me/player/play?device_id=${did}`,
      {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ uris: [uri] }),
      }
    );

    if (!response.ok && response.status !== 204) {
      const err = await response.json().catch(() => ({ error: { message: response.statusText } }));
      throw new Error(err?.error?.message || `Playback failed: ${response.status}`);
    }
  }, [fetchToken]);

  // ---- Public API ----

  const play = useCallback(async (uri: string) => {
    if (!playerRef.current || !isConnectedRef.current) return;
    stopTrackEndMonitoring();

    try {
      await startPlaybackOnDevice(uri);
      await playerRef.current.setVolume(masterVolumeRef.current);
      setIsPlaying(true);
      startPositionUpdates();
      startTrackEndMonitoring();
    } catch (error) {
      console.error('useSpotifyPlayer: play error:', error);
      onErrorRef.current?.(String(error));
    }
  }, [startPlaybackOnDevice, startPositionUpdates, startTrackEndMonitoring, stopTrackEndMonitoring]);

  const pause = useCallback(() => {
    playerRef.current?.pause();
    setIsPlaying(false);
  }, []);

  const resume = useCallback(async () => {
    if (!playerRef.current) return;
    try {
      await playerRef.current.resume();
      setIsPlaying(true);
      startPositionUpdates();
    } catch (error) {
      console.error('useSpotifyPlayer: resume error:', error);
    }
  }, [startPositionUpdates]);

  const skipTo = useCallback(async (uri: string) => {
    if (crossfadeTimerRef.current) {
      cancelAnimationFrame(crossfadeTimerRef.current);
      crossfadeTimerRef.current = null;
    }
    isCrossfadingRef.current = false;
    setIsCrossfading(false);

    await play(uri);
  }, [play]);

  const crossfadeTo = useCallback(async (uri: string) => {
    if (!playerRef.current || !isConnectedRef.current) {
      return play(uri);
    }

    if (isCrossfadingRef.current) {
      return play(uri);
    }

    isCrossfadingRef.current = true;
    setIsCrossfading(true);
    stopTrackEndMonitoring();

    try {
      // Phase 1: Fade volume down (exponential curve)
      await fadeVolumeDown();

      // Phase 2: Switch track at near-zero volume (inaudible)
      await startPlaybackOnDevice(uri);

      // Phase 3: Fade volume back up (exponential curve)
      await fadeVolumeUp();

      startPositionUpdates();
      startTrackEndMonitoring();
    } catch (error) {
      console.error('useSpotifyPlayer: crossfade error:', error);
      // Fallback: restore volume and try direct play
      await playerRef.current?.setVolume(masterVolumeRef.current);
      try { await startPlaybackOnDevice(uri); } catch { /* ignore */ }
    } finally {
      isCrossfadingRef.current = false;
      setIsCrossfading(false);
    }
  }, [play, fadeVolumeDown, fadeVolumeUp, startPlaybackOnDevice, startPositionUpdates, startTrackEndMonitoring, stopTrackEndMonitoring]);

  const seek = useCallback((time: number) => {
    playerRef.current?.seek(Math.max(0, time * 1000));
    setCurrentTime(time);
  }, []);

  const setVolume = useCallback((vol: number) => {
    const clamped = Math.max(0, Math.min(1, vol));
    masterVolumeRef.current = clamped;
    setVolumeState(clamped);
    if (!isCrossfadingRef.current) {
      playerRef.current?.setVolume(clamped);
    }
  }, []);

  const setCrossfadeDuration = useCallback((ms: number) => {
    const safe = Math.max(0, ms);
    fadeOutDurationRef.current = Math.round(safe * 0.6);
    fadeInDurationRef.current = Math.round(safe * 0.4);
  }, []);

  const cleanup = useCallback(() => {
    stopPositionUpdates();
    stopTrackEndMonitoring();
    if (crossfadeTimerRef.current) cancelAnimationFrame(crossfadeTimerRef.current);
    if (tokenRefreshTimerRef.current) clearTimeout(tokenRefreshTimerRef.current);
    playerRef.current?.disconnect();
    playerRef.current = null;
    isConnectedRef.current = false;
    setIsPlaying(false);
    setIsCrossfading(false);
    setCurrentTime(0);
    setDuration(0);
    setIsReady(false);
    setDeviceId(null);
  }, [stopPositionUpdates, stopTrackEndMonitoring]);

  return {
    play,
    pause,
    resume,
    skipTo,
    crossfadeTo,
    seek,
    isPlaying,
    isCrossfading,
    currentTime,
    duration,
    volume,
    setVolume,
    setCrossfadeDuration,
    isReady,
    deviceId,
    sdkError,
    cleanup,
  };
}

export default useSpotifyPlayer;
