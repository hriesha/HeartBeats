/**
 * React Hook for Apple MusicKit JS Playback with Volume-Based Crossfade
 *
 * Provides in-browser full-track playback via MusicKit JS v3.
 * Same interface as useSpotifyPlayer for drop-in replacement.
 *
 * All Apple Music subscribers get full playback (no Premium distinction).
 */

import { useState, useEffect, useCallback, useRef } from 'react';

export interface UseMusicKitPlayerOptions {
  crossfadeDuration?: number;
  onTrackEnd?: () => void;
  onError?: (error: string) => void;
  onReady?: () => void;
}

export interface UseMusicKitPlayerReturn {
  play: (appleMusicId: string) => Promise<void>;
  pause: () => void;
  resume: () => Promise<void>;
  skipTo: (appleMusicId: string) => Promise<void>;
  crossfadeTo: (appleMusicId: string) => Promise<void>;
  seek: (time: number) => void;

  isPlaying: boolean;
  isCrossfading: boolean;
  currentTime: number;
  duration: number;
  volume: number;

  setVolume: (volume: number) => void;
  setCrossfadeDuration: (ms: number) => void;

  isReady: boolean;
  sdkError: string | null;

  cleanup: () => void;
}

export function useMusicKitPlayer(
  options: UseMusicKitPlayerOptions = {}
): UseMusicKitPlayerReturn {
  const {
    crossfadeDuration: initialCrossfadeDuration = 5000,
    onTrackEnd,
    onError,
    onReady,
  } = options;

  const [isPlaying, setIsPlaying] = useState(false);
  const [isCrossfading, setIsCrossfading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolumeState] = useState(1);
  const [isReady, setIsReady] = useState(false);
  const [sdkError, setSdkError] = useState<string | null>(null);

  const musicRef = useRef<MusicKit.MusicKitInstance | null>(null);
  const onTrackEndRef = useRef(onTrackEnd);
  const onErrorRef = useRef(onError);
  const onReadyRef = useRef(onReady);
  const masterVolumeRef = useRef(1);
  const isCrossfadingRef = useRef(false);
  const crossfadeTimerRef = useRef<number | null>(null);
  const positionIntervalRef = useRef<number | null>(null);
  const trackEndCheckRef = useRef<number | null>(null);
  const fadeOutDurationRef = useRef(Math.round(initialCrossfadeDuration * 0.6));
  const fadeInDurationRef = useRef(Math.round(initialCrossfadeDuration * 0.4));

  useEffect(() => { onTrackEndRef.current = onTrackEnd; }, [onTrackEnd]);
  useEffect(() => { onErrorRef.current = onError; }, [onError]);
  useEffect(() => { onReadyRef.current = onReady; }, [onReady]);

  // Position tracking
  const stopPositionUpdates = useCallback(() => {
    if (positionIntervalRef.current !== null) {
      clearInterval(positionIntervalRef.current);
      positionIntervalRef.current = null;
    }
  }, []);

  const startPositionUpdates = useCallback(() => {
    stopPositionUpdates();
    positionIntervalRef.current = window.setInterval(() => {
      const music = musicRef.current;
      if (!music) return;
      setCurrentTime(music.player.currentPlaybackTime);
      setDuration(music.player.currentPlaybackDuration);
      setIsPlaying(music.player.isPlaying);
    }, 250);
  }, [stopPositionUpdates]);

  // Track end monitoring
  const stopTrackEndMonitoring = useCallback(() => {
    if (trackEndCheckRef.current !== null) {
      clearInterval(trackEndCheckRef.current);
      trackEndCheckRef.current = null;
    }
  }, []);

  const startTrackEndMonitoring = useCallback(() => {
    stopTrackEndMonitoring();
    trackEndCheckRef.current = window.setInterval(() => {
      if (isCrossfadingRef.current) return;
      const music = musicRef.current;
      if (!music || !music.player.isPlaying) return;

      const remaining = music.player.currentPlaybackTimeRemaining;
      const fadeOutMs = fadeOutDurationRef.current;

      if (remaining <= fadeOutMs / 1000 && remaining > 0) {
        stopTrackEndMonitoring();
        onTrackEndRef.current?.();
      }
    }, 200);
  }, [stopTrackEndMonitoring]);

  // MusicKit initialization
  useEffect(() => {
    let cancelled = false;

    const initMusicKit = async () => {
      try {
        if (!window.MusicKit) {
          // Wait for MusicKit to load
          const checkInterval = setInterval(() => {
            if (window.MusicKit && !cancelled) {
              clearInterval(checkInterval);
              initMusicKit();
            }
          }, 100);
          // Timeout after 10s
          setTimeout(() => {
            clearInterval(checkInterval);
            if (!cancelled && !window.MusicKit) {
              setSdkError('MusicKit JS failed to load');
            }
          }, 10000);
          return;
        }

        const developerToken = import.meta.env.VITE_APPLE_MUSIC_DEVELOPER_TOKEN;
        if (!developerToken) {
          setSdkError('Missing Apple Music developer token');
          return;
        }

        const music = await window.MusicKit.configure({
          developerToken,
          app: { name: 'HeartBeats', build: '1.0.0' },
        });

        if (cancelled) return;

        musicRef.current = music;
        music.volume = masterVolumeRef.current;

        // Listen for playback state changes
        music.addEventListener('playbackStateDidChange', (event: any) => {
          const state = event?.state ?? music.playbackState;
          setIsPlaying(state === MusicKit.PlaybackStates.playing);

          if (state === MusicKit.PlaybackStates.completed ||
              state === MusicKit.PlaybackStates.ended) {
            if (!isCrossfadingRef.current) {
              onTrackEndRef.current?.();
            }
          }
        });

        music.addEventListener('nowPlayingItemDidChange', () => {
          const item = music.nowPlayingItem;
          if (item) {
            setDuration(item.duration);
          }
        });

        setIsReady(true);
        setSdkError(null);
        onReadyRef.current?.();
      } catch (err: any) {
        if (!cancelled) {
          const msg = err?.message || String(err);
          setSdkError(`MusicKit init error: ${msg}`);
          onErrorRef.current?.(msg);
        }
      }
    };

    initMusicKit();

    return () => {
      cancelled = true;
      stopPositionUpdates();
      stopTrackEndMonitoring();
      if (crossfadeTimerRef.current) cancelAnimationFrame(crossfadeTimerRef.current);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Start playback of a song by Apple Music ID
  const startPlaybackById = useCallback(async (appleMusicId: string) => {
    const music = musicRef.current;
    if (!music) throw new Error('MusicKit not ready');

    await music.setQueue({ song: appleMusicId });
    await music.play();
  }, []);

  // Smooth crossfade: fade current track down, switch at low volume, fade new track up.
  // The switch happens at ~15% volume so the gap is imperceptible — feels like an overlap.
  const smoothCrossfade = useCallback((appleMusicId: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      const music = musicRef.current;
      if (!music) { reject(new Error('MusicKit not ready')); return; }

      const totalMs = fadeOutDurationRef.current + fadeInDurationRef.current;
      // Switch point: 40% through the total duration (when volume is low)
      const switchAt = 0.4;
      const startVolume = masterVolumeRef.current;
      const startTime = Date.now();
      let switched = false;

      const animate = async () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / totalMs, 1);

        if (progress < switchAt) {
          // Phase 1: Fade out current track
          // Map 0→switchAt to 1→~0.15 with an ease-in curve
          const fadeProgress = progress / switchAt;
          const vol = startVolume * (1 - fadeProgress * 0.85); // down to 15%
          music.volume = Math.max(0.01, vol);
        } else {
          // Phase 2: Switch track once, then fade in
          if (!switched) {
            switched = true;
            try {
              music.volume = 0.01;
              await startPlaybackById(appleMusicId);
              music.volume = startVolume * 0.15;
            } catch (err) {
              reject(err);
              return;
            }
          }
          // Fade from 15% up to full volume
          const fadeProgress = (progress - switchAt) / (1 - switchAt);
          const vol = startVolume * (0.15 + fadeProgress * 0.85);
          music.volume = Math.min(startVolume, vol);
        }

        if (progress < 1) {
          crossfadeTimerRef.current = requestAnimationFrame(animate);
        } else {
          crossfadeTimerRef.current = null;
          music.volume = startVolume;
          resolve();
        }
      };

      crossfadeTimerRef.current = requestAnimationFrame(animate);
    });
  }, [startPlaybackById]);

  // Public API
  const play = useCallback(async (appleMusicId: string) => {
    const music = musicRef.current;
    if (!music) return;
    stopTrackEndMonitoring();

    try {
      await startPlaybackById(appleMusicId);
      music.volume = masterVolumeRef.current;
      setIsPlaying(true);
      startPositionUpdates();
      startTrackEndMonitoring();
    } catch (error: any) {
      console.error('useMusicKitPlayer: play error:', error);
      onErrorRef.current?.(String(error));
    }
  }, [startPlaybackById, startPositionUpdates, startTrackEndMonitoring, stopTrackEndMonitoring]);

  const pause = useCallback(() => {
    musicRef.current?.pause();
    setIsPlaying(false);
  }, []);

  const resume = useCallback(async () => {
    const music = musicRef.current;
    if (!music) return;
    try {
      await music.play();
      setIsPlaying(true);
      startPositionUpdates();
    } catch (error) {
      console.error('useMusicKitPlayer: resume error:', error);
    }
  }, [startPositionUpdates]);

  const skipTo = useCallback(async (appleMusicId: string) => {
    if (crossfadeTimerRef.current) {
      cancelAnimationFrame(crossfadeTimerRef.current);
      crossfadeTimerRef.current = null;
    }
    isCrossfadingRef.current = false;
    setIsCrossfading(false);

    await play(appleMusicId);
  }, [play]);

  const crossfadeTo = useCallback(async (appleMusicId: string) => {
    const music = musicRef.current;
    if (!music) {
      return play(appleMusicId);
    }

    if (isCrossfadingRef.current) {
      return play(appleMusicId);
    }

    isCrossfadingRef.current = true;
    setIsCrossfading(true);
    stopTrackEndMonitoring();

    try {
      await smoothCrossfade(appleMusicId);
      startPositionUpdates();
      startTrackEndMonitoring();
    } catch (error) {
      console.error('useMusicKitPlayer: crossfade error:', error);
      music.volume = masterVolumeRef.current;
      try { await startPlaybackById(appleMusicId); } catch { /* ignore */ }
    } finally {
      isCrossfadingRef.current = false;
      setIsCrossfading(false);
    }
  }, [play, smoothCrossfade, startPlaybackById, startPositionUpdates, startTrackEndMonitoring, stopTrackEndMonitoring]);

  const seek = useCallback((time: number) => {
    musicRef.current?.seekToTime(Math.max(0, time));
    setCurrentTime(time);
  }, []);

  const setVolume = useCallback((vol: number) => {
    const clamped = Math.max(0, Math.min(1, vol));
    masterVolumeRef.current = clamped;
    setVolumeState(clamped);
    if (!isCrossfadingRef.current && musicRef.current) {
      musicRef.current.volume = clamped;
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
    musicRef.current?.stop();
    setIsPlaying(false);
    setIsCrossfading(false);
    setCurrentTime(0);
    setDuration(0);
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
    sdkError,
    cleanup,
  };
}

export default useMusicKitPlayer;
