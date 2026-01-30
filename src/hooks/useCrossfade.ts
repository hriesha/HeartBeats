/**
 * React Hook for Audio Crossfade
 *
 * Provides a React-friendly interface to the AudioCrossfader class.
 * Handles state management and cleanup on unmount.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { AudioCrossfader, CrossfadeOptions } from '../utils/audioCrossfade';

export interface UseCrossfadeOptions {
  crossfadeDuration?: number;
  onTrackEnd?: () => void;
}

export interface UseCrossfadeReturn {
  // Playback controls
  play: (url: string) => Promise<void>;
  pause: () => void;
  resume: () => Promise<void>;
  skipTo: (url: string) => Promise<void>;
  crossfadeTo: (url: string) => Promise<void>;
  seek: (time: number) => void;

  // State
  isPlaying: boolean;
  isCrossfading: boolean;
  currentTime: number;
  duration: number;
  volume: number;

  // Volume control
  setVolume: (volume: number) => void;

  // Configuration
  setCrossfadeDuration: (ms: number) => void;

  // Cleanup
  cleanup: () => void;
}

export function useCrossfade(options: UseCrossfadeOptions = {}): UseCrossfadeReturn {
  const { crossfadeDuration = 5000, onTrackEnd } = options;

  // State
  const [isPlaying, setIsPlaying] = useState(false);
  const [isCrossfading, setIsCrossfading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolumeState] = useState(1);

  // Refs
  const crossfaderRef = useRef<AudioCrossfader | null>(null);
  const timeUpdateIntervalRef = useRef<number | null>(null);
  const onTrackEndRef = useRef(onTrackEnd);

  // Keep onTrackEnd ref updated
  useEffect(() => {
    onTrackEndRef.current = onTrackEnd;
  }, [onTrackEnd]);

  // Initialize crossfader
  useEffect(() => {
    const crossfaderOptions: CrossfadeOptions = {
      crossfadeDuration,
      onTrackStart: () => {
        setIsPlaying(true);
      },
      onTrackEnd: () => {
        onTrackEndRef.current?.();
      },
      onCrossfadeProgress: (progress) => {
        setIsCrossfading(progress < 1);
      },
    };

    crossfaderRef.current = new AudioCrossfader(crossfaderOptions);

    // Cleanup on unmount
    return () => {
      stopTimeUpdates();
      crossfaderRef.current?.cleanup();
      crossfaderRef.current = null;
    };
  }, []); // Only run once on mount

  // Update crossfade duration when it changes
  useEffect(() => {
    crossfaderRef.current?.setCrossfadeDuration(crossfadeDuration);
  }, [crossfadeDuration]);

  // Time update interval
  const startTimeUpdates = useCallback(() => {
    stopTimeUpdates();

    timeUpdateIntervalRef.current = window.setInterval(() => {
      if (crossfaderRef.current) {
        setCurrentTime(crossfaderRef.current.getCurrentTime());
        setDuration(crossfaderRef.current.getDuration());
        setIsPlaying(crossfaderRef.current.isPlaying());
      }
    }, 250); // Update 4 times per second
  }, []);

  const stopTimeUpdates = useCallback(() => {
    if (timeUpdateIntervalRef.current !== null) {
      clearInterval(timeUpdateIntervalRef.current);
      timeUpdateIntervalRef.current = null;
    }
  }, []);

  // Play a track
  const play = useCallback(async (url: string) => {
    if (!crossfaderRef.current) return;

    try {
      await crossfaderRef.current.play(url);
      setIsPlaying(true);
      startTimeUpdates();
    } catch (error) {
      console.error('useCrossfade: Error playing track:', error);
      setIsPlaying(false);
    }
  }, [startTimeUpdates]);

  // Pause playback
  const pause = useCallback(() => {
    crossfaderRef.current?.pause();
    setIsPlaying(false);
  }, []);

  // Resume playback
  const resume = useCallback(async () => {
    if (!crossfaderRef.current) return;

    try {
      await crossfaderRef.current.resume();
      setIsPlaying(true);
      startTimeUpdates();
    } catch (error) {
      console.error('useCrossfade: Error resuming:', error);
    }
  }, [startTimeUpdates]);

  // Skip to next track (instant, no crossfade)
  const skipTo = useCallback(async (url: string) => {
    if (!crossfaderRef.current) return;

    try {
      await crossfaderRef.current.skipTo(url);
      setIsPlaying(true);
      setIsCrossfading(false);
      startTimeUpdates();
    } catch (error) {
      console.error('useCrossfade: Error skipping:', error);
    }
  }, [startTimeUpdates]);

  // Crossfade to next track (smooth transition)
  const crossfadeTo = useCallback(async (url: string) => {
    if (!crossfaderRef.current) return;

    try {
      setIsCrossfading(true);
      await crossfaderRef.current.crossfadeTo(url);
      setIsCrossfading(false);
      startTimeUpdates();
    } catch (error) {
      console.error('useCrossfade: Error crossfading:', error);
      setIsCrossfading(false);
    }
  }, [startTimeUpdates]);

  // Seek to time
  const seek = useCallback((time: number) => {
    crossfaderRef.current?.seek(time);
    setCurrentTime(time);
  }, []);

  // Set volume
  const setVolume = useCallback((vol: number) => {
    const clampedVol = Math.max(0, Math.min(1, vol));
    crossfaderRef.current?.setVolume(clampedVol);
    setVolumeState(clampedVol);
  }, []);

  // Set crossfade duration
  const setCrossfadeDuration = useCallback((ms: number) => {
    crossfaderRef.current?.setCrossfadeDuration(ms);
  }, []);

  // Cleanup
  const cleanup = useCallback(() => {
    stopTimeUpdates();
    crossfaderRef.current?.cleanup();
    setIsPlaying(false);
    setIsCrossfading(false);
    setCurrentTime(0);
    setDuration(0);
  }, [stopTimeUpdates]);

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
    cleanup,
  };
}

export default useCrossfade;
