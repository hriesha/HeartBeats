import { motion } from 'motion/react';
import { ChevronLeft, Play, Pause, Music, SkipForward } from 'lucide-react';
import { VibeType } from '../App';
import { useState, useEffect, useCallback, useRef } from 'react';
import { getClusterTracks, getTracksFromTrack, startPlayback, Track } from '../utils/api';
import { useSpotifyPlayer } from '../hooks/useSpotifyPlayer';

interface VibeDetailProps {
  vibe: VibeType;
  bpm?: number;
  onBack: () => void;
  isPremium?: boolean;
}

function toUri(t: Track): string {
  const id = t.id ?? t.track_id;
  return id?.startsWith('spotify:') ? id : `spotify:track:${id}`;
}

export function VibeDetail({ vibe, bpm = 120, onBack, isPremium = false }: VibeDetailProps) {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [nowPlayingIndex, setNowPlayingIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const clusterId = parseInt(vibe.id.split('-')[1] || '0');
  const tracksRef = useRef<Track[]>([]);
  const nowPlayingIndexRef = useRef(0);

  // Keep refs in sync for use in callbacks
  useEffect(() => { tracksRef.current = tracks; }, [tracks]);
  useEffect(() => { nowPlayingIndexRef.current = nowPlayingIndex; }, [nowPlayingIndex]);

  // Auto-advance handler for SDK player
  const handleAutoAdvance = useCallback(() => {
    const currentIdx = nowPlayingIndexRef.current;
    const currentTracks = tracksRef.current;
    const nextIdx = currentIdx + 1;
    if (nextIdx >= currentTracks.length) return;

    const next = currentTracks[nextIdx];
    setNowPlayingIndex(nextIdx);

    // Extend queue with KNN recommendations
    const tid = next.track_id ?? next.id;
    if (tid) {
      getTracksFromTrack(tid, clusterId, 10).then(from => {
        if (!from?.tracks?.length) return;
        const add = from.tracks.filter(m => (m.track_id ?? m.id) !== tid);
        setTracks(prev => {
          const rest = prev.slice(nextIdx + 1);
          const seen = new Set(rest.map(t => t.track_id ?? t.id));
          const extra = add.filter(m => !seen.has(m.track_id ?? m.id));
          return [...prev.slice(0, nextIdx + 1), ...extra, ...rest];
        });
      });
    }
  }, [clusterId]);

  // SDK player for in-browser playback
  const sdkPlayer = useSpotifyPlayer({
    crossfadeDuration: 5000,
    onTrackEnd: handleAutoAdvance,
  });

  const useSDK = isPremium && sdkPlayer.isReady;

  // Track whether the index change was a manual action (click/skip)
  const isManualActionRef = useRef(false);
  const prevAutoIndexRef = useRef(-1);

  // Crossfade to next track ONLY on auto-advance (track ended), not manual clicks
  useEffect(() => {
    const prev = prevAutoIndexRef.current;
    prevAutoIndexRef.current = nowPlayingIndex;

    // Manual actions handle playback themselves — skip crossfade
    if (isManualActionRef.current) {
      isManualActionRef.current = false;
      return;
    }

    // Only crossfade on auto-advance (index incremented by 1)
    if (useSDK && nowPlayingIndex >= 0 && prev >= 0 && nowPlayingIndex === prev + 1) {
      const track = tracks[nowPlayingIndex];
      if (track) {
        sdkPlayer.crossfadeTo(toUri(track));
      }
    }
  }, [nowPlayingIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  // Play a track by index
  const playTrack = useCallback(async (index: number) => {
    const t = tracks[index];
    if (!t) return;
    if (useSDK) {
      await sdkPlayer.skipTo(toUri(t));
    } else {
      const res = await startPlayback([toUri(t)]);
      if (!res.success) console.warn('startPlayback failed:', res.error);
    }
  }, [tracks, useSDK, sdkPlayer]);

  // Initial data fetch and first track playback
  const initialPlayDone = useRef(false);
  useEffect(() => {
    let cancelled = false;
    initialPlayDone.current = false;

    const run = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const clusterData = await getClusterTracks(clusterId, bpm, 20);
        if (!clusterData?.tracks?.length) {
          setError('No tracks found for this cluster. Try another vibe.');
          return;
        }
        const allClusterTracks = clusterData.tracks;
        if (cancelled) return;

        const randomIdx = Math.floor(Math.random() * allClusterTracks.length);
        const randomTrack = allClusterTracks[randomIdx];
        const otherTracks = allClusterTracks.filter((_, idx) => idx !== randomIdx);
        const ordered = [randomTrack, ...otherTracks];

        setTracks(ordered);
        tracksRef.current = ordered;
        setNowPlayingIndex(0);
        nowPlayingIndexRef.current = 0;

        // Start playback after state is set
        if (!cancelled) {
          initialPlayDone.current = true;
        }
      } catch (err) {
        if (!cancelled) {
          console.error('VibeDetail fetch error:', err);
          setError('Failed to load tracks. Please try again.');
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    run();
    return () => { cancelled = true; };
  }, [clusterId, bpm]);

  // Start playback once SDK is ready and tracks are loaded
  useEffect(() => {
    if (!initialPlayDone.current || tracks.length === 0) return;
    const firstTrack = tracks[0];
    if (!firstTrack) return;

    // If Premium, wait for SDK to be ready — don't fall back to backend
    if (isPremium && !sdkPlayer.isReady) return;

    if (useSDK) {
      sdkPlayer.play(toUri(firstTrack));
    } else {
      startPlayback([toUri(firstTrack)]);
    }
    initialPlayDone.current = false; // Only play once
  }, [useSDK, tracks]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup SDK on unmount
  useEffect(() => {
    return () => { sdkPlayer.cleanup(); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleNext = async () => {
    const nextIdx = nowPlayingIndex + 1;
    if (nextIdx >= tracks.length) return;
    const next = tracks[nextIdx];

    // Manual skip with crossfade, prevent auto-advance double-trigger
    isManualActionRef.current = true;
    setNowPlayingIndex(nextIdx);
    if (useSDK) {
      await sdkPlayer.crossfadeTo(toUri(next));
    } else {
      await playTrack(nextIdx);
    }

    // Extend queue with KNN recommendations
    const tid = next.track_id ?? next.id;
    if (!tid) return;
    const from = await getTracksFromTrack(tid, clusterId, 10);
    if (!from?.tracks?.length) return;
    const add = from.tracks.filter(m => (m.track_id ?? m.id) !== tid);
    setTracks(prev => {
      const rest = prev.slice(nextIdx + 1);
      const seen = new Set(rest.map(t => t.track_id ?? t.id));
      const extra = add.filter(m => !seen.has(m.track_id ?? m.id));
      return [...prev.slice(0, nextIdx + 1), ...extra, ...rest];
    });
  };

  const handlePlayPause = async (track: Track, index: number) => {
    if (!useSDK) {
      // Non-SDK: same behavior as before
      const ok = await startPlayback([toUri(track)]);
      if (ok.success) {
        setNowPlayingIndex(index);
        return;
      }
      if (track.preview_url) window.open(track.preview_url, '_blank');
      else if (track.external_urls) window.open(track.external_urls, '_blank');
      return;
    }

    // SDK mode: play/pause toggle
    if (index === nowPlayingIndex) {
      if (sdkPlayer.isPlaying) {
        sdkPlayer.pause();
      } else {
        sdkPlayer.resume();
      }
    } else {
      isManualActionRef.current = true;
      setNowPlayingIndex(index);
      await sdkPlayer.crossfadeTo(toUri(track));
    }
  };

  const formatDuration = (ms?: number): string => {
    if (!ms) return '0:00';
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatTime = (sec: number): string => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const nowPlaying = tracks[nowPlayingIndex];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      {/* Back Button */}
      <button
        onClick={onBack}
        style={{
          position: 'absolute', top: 20, left: 20, width: 40, height: 40, borderRadius: '50%',
          background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)',
          color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', zIndex: 20,
        }}
      >
        <ChevronLeft style={{ width: 20, height: 20 }} />
      </button>

      {/* Vibe Header */}
      <div style={{ paddingTop: 72, paddingBottom: 16, paddingLeft: 24, paddingRight: 24 }}>
        <motion.div
          style={{
            borderRadius: '16px',
            padding: '20px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'rgba(255, 45, 85, 0.08)',
            border: '1px solid rgba(255, 45, 85, 0.2)',
            boxShadow: '0 0 40px rgba(255, 45, 85, 0.08)',
          }}
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <div>
            <h1 style={{
              fontFamily: 'var(--font-heading)', fontSize: '24px', fontWeight: 200,
              color: '#ffffff', letterSpacing: '0.1em', marginBottom: 4,
            }}>
              {vibe.name}
            </h1>
            {vibe.tags.length > 0 && (
              <p style={{
                fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 300,
                color: 'rgba(255, 255, 255, 0.4)',
              }}>
                {vibe.tags.join(' · ')}
              </p>
            )}
          </div>
          <div style={{
            width: 40, height: 40, borderRadius: '10px',
            background: 'rgba(255, 45, 85, 0.12)', border: '1px solid rgba(255, 45, 85, 0.25)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Music style={{ width: 18, height: 18, color: '#FF2D55' }} />
          </div>
        </motion.div>
      </div>

      {/* Now Playing Bar (SDK mode) */}
      {useSDK && nowPlaying && !isLoading && (
        <div style={{ padding: '0 24px 16px' }}>
          <div style={{
            borderRadius: '14px', padding: '14px 16px',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            background: 'rgba(255, 45, 85, 0.06)',
            border: '1px solid rgba(255, 45, 85, 0.15)',
          }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <p style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: '#FF2D55', opacity: 0.8, fontWeight: 400 }}>
                {sdkPlayer.isCrossfading ? 'crossfading...' : 'now playing'}
              </p>
              <p style={{
                fontFamily: 'var(--font-body)', fontSize: '14px', color: '#ffffff', fontWeight: 500,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {nowPlaying.name}
              </p>
              {/* Progress bar */}
              {sdkPlayer.duration > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{
                    height: 2, borderRadius: 1, overflow: 'hidden',
                    background: 'rgba(255, 255, 255, 0.08)',
                  }}>
                    <div style={{
                      height: '100%', borderRadius: 1, transition: 'width 0.3s',
                      width: `${(sdkPlayer.currentTime / sdkPlayer.duration) * 100}%`,
                      background: '#FF2D55',
                    }} />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                    <span style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'rgba(255,255,255,0.3)' }}>
                      {formatTime(sdkPlayer.currentTime)}
                    </span>
                    <span style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'rgba(255,255,255,0.3)' }}>
                      {formatTime(sdkPlayer.duration)}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Play/Pause */}
            <motion.button
              onClick={() => {
                if (sdkPlayer.isPlaying) sdkPlayer.pause();
                else sdkPlayer.resume();
              }}
              style={{
                marginLeft: 12, padding: 10, borderRadius: '50%',
                background: 'rgba(255, 45, 85, 0.15)', border: 'none',
                color: '#FF2D55', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              {sdkPlayer.isPlaying ? (
                <Pause style={{ width: 18, height: 18 }} />
              ) : (
                <Play style={{ width: 18, height: 18 }} />
              )}
            </motion.button>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 300, padding: '24px' }}>
          <motion.div
            style={{ width: 40, height: 40, border: '2px solid #FF2D55', borderTopColor: 'transparent', borderRadius: '50%' }}
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          />
          <p style={{
            fontFamily: 'var(--font-body)', fontSize: '14px', color: 'rgba(255,255,255,0.4)',
            marginTop: 20, fontWeight: 300,
          }}>
            {isPremium && !sdkPlayer.isReady ? 'connecting to player...' : 'loading your queue...'}
          </p>
        </div>
      )}

      {/* Error State */}
      {error && !isLoading && (
        <div style={{ padding: '0 24px 24px' }}>
          <div style={{
            borderRadius: '14px', padding: '16px',
            background: 'rgba(255, 45, 85, 0.08)',
            border: '1px solid rgba(255, 45, 85, 0.2)',
          }}>
            <p style={{
              fontFamily: 'var(--font-body)', fontSize: '14px', color: 'rgba(255,255,255,0.6)',
              textAlign: 'center', fontWeight: 300,
            }}>
              {error}
            </p>
          </div>
        </div>
      )}

      {/* SDK Error Notice */}
      {isPremium && sdkPlayer.sdkError && !isLoading && (
        <div style={{ padding: '0 24px 12px' }}>
          <div style={{
            borderRadius: '10px', padding: '8px 12px',
            background: 'rgba(255, 165, 0, 0.08)',
            border: '1px solid rgba(255, 165, 0, 0.2)',
          }}>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'rgba(255, 165, 0, 0.7)', textAlign: 'center', fontWeight: 300 }}>
              SDK: {sdkPlayer.sdkError} — using remote playback
            </p>
          </div>
        </div>
      )}

      {/* Track Queue */}
      {!isLoading && !error && tracks.length > 0 && (
        <div style={{ padding: '0 24px 24px', flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <h2 style={{
              fontFamily: 'var(--font-heading)', fontSize: '18px', fontWeight: 200,
              color: '#ffffff', letterSpacing: '0.08em',
            }}>
              your queue
            </h2>
            <motion.button
              onClick={handleNext}
              disabled={nowPlayingIndex >= tracks.length - 1}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '8px 16px', borderRadius: '20px',
                background: '#FF2D55', color: '#ffffff', border: 'none',
                fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 500,
                cursor: 'pointer', opacity: nowPlayingIndex >= tracks.length - 1 ? 0.4 : 1,
              }}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              <SkipForward style={{ width: 14, height: 14 }} />
              next
            </motion.button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {tracks.map((track, index) => {
              const isNowPlaying = index === nowPlayingIndex;
              return (
                <motion.div
                  key={track.track_id || track.id || index}
                  style={{
                    borderRadius: '14px', padding: '14px 16px',
                    display: 'flex', alignItems: 'center', gap: 12,
                    background: isNowPlaying ? 'rgba(255, 45, 85, 0.08)' : 'rgba(255, 255, 255, 0.03)',
                    border: isNowPlaying ? '1px solid rgba(255, 45, 85, 0.2)' : '1px solid rgba(255, 255, 255, 0.06)',
                  }}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.04 }}
                >
                  {/* Album Art or Placeholder */}
                  <div style={{
                    width: 48, height: 48, borderRadius: '10px', flexShrink: 0, overflow: 'hidden',
                    background: track.images && track.images.length > 0
                      ? `url(${track.images[0].url}) center/cover`
                      : 'rgba(255, 45, 85, 0.06)',
                    border: '1px solid rgba(255, 255, 255, 0.06)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    position: 'relative',
                  }}>
                    {!track.images || track.images.length === 0 ? (
                      <Music style={{ width: 20, height: 20, color: 'rgba(255, 45, 85, 0.4)' }} />
                    ) : null}
                    {track.rank && (
                      <div style={{
                        position: 'absolute', top: 0, right: 0,
                        borderBottomLeftRadius: '6px', padding: '1px 5px',
                        background: 'rgba(0, 0, 0, 0.7)',
                        fontFamily: 'var(--font-body)', fontSize: '9px', fontWeight: 600, color: '#FF2D55',
                      }}>
                        #{track.rank}
                      </div>
                    )}
                  </div>

                  {/* Song Info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <h3 style={{
                      fontFamily: 'var(--font-body)', fontSize: '14px', fontWeight: 500,
                      color: '#ffffff', marginBottom: 2,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {track.name || 'Unknown Track'}
                    </h3>
                    <p style={{
                      fontFamily: 'var(--font-body)', fontSize: '12px', fontWeight: 300,
                      color: 'rgba(255, 255, 255, 0.4)',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {track.artist_names || track.artists || 'Unknown Artist'}
                    </p>
                    {track.tempo && (
                      <p style={{
                        fontFamily: 'var(--font-body)', fontSize: '10px', fontWeight: 400,
                        color: '#FF2D55', opacity: 0.7, marginTop: 2,
                      }}>
                        {Math.round(track.tempo)} BPM
                      </p>
                    )}
                  </div>

                  {/* Duration */}
                  {track.duration_ms && (
                    <span style={{
                      fontFamily: 'var(--font-body)', fontSize: '11px', fontWeight: 300,
                      color: 'rgba(255, 255, 255, 0.3)', marginRight: 4,
                    }}>
                      {formatDuration(track.duration_ms)}
                    </span>
                  )}

                  {/* Play / Pause */}
                  <motion.button
                    onClick={() => handlePlayPause(track, index)}
                    style={{
                      padding: 8, borderRadius: '50%', flexShrink: 0, border: 'none', cursor: 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: isNowPlaying && useSDK && sdkPlayer.isPlaying
                        ? '#FF2D55'
                        : 'rgba(255, 45, 85, 0.12)',
                      color: isNowPlaying && useSDK && sdkPlayer.isPlaying ? '#ffffff' : '#FF2D55',
                    }}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    {isNowPlaying && useSDK && sdkPlayer.isPlaying ? (
                      <Pause style={{ width: 14, height: 14 }} />
                    ) : (
                      <Play style={{ width: 14, height: 14 }} />
                    )}
                  </motion.button>
                </motion.div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
