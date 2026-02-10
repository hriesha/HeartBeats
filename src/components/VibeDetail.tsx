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
    <div className="relative w-full h-full overflow-auto" style={{ fontFamily: 'Poppins, sans-serif' }}>
      {/* Background with gradient overlay */}
      <div
        className="absolute inset-0 z-0"
        style={{
          background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
        }}
      />

      {/* Content */}
      <div className="relative z-10 w-full h-full">
        {/* Back Button */}
        <button
          onClick={onBack}
          className="absolute top-4 left-4 p-2 rounded-full transition-all z-20"
          style={{
            backgroundColor: 'rgba(0, 48, 73, 0.8)',
            color: '#FCBF49'
          }}
        >
          <ChevronLeft className="w-6 h-6" />
        </button>

        {/* Vibe Header */}
        <div className="pt-16 pb-6 px-6">
          <motion.div
            className="rounded-2xl p-6 flex items-center justify-between"
            style={{
              backgroundColor: vibe.color,
              boxShadow: `0 8px 24px ${vibe.color}80, inset 0 2px 8px rgba(255, 255, 255, 0.3)`,
            }}
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div>
              <h1
                style={{
                  fontFamily: 'Poppins, sans-serif',
                  fontSize: '24px',
                  fontWeight: 700,
                  color: '#03071E',
                  marginBottom: '4px'
                }}
              >
                {vibe.name}
              </h1>
              {vibe.tags.length > 0 && (
                <p
                  style={{
                    fontFamily: 'Poppins, sans-serif',
                    fontSize: '13px',
                    fontWeight: 600,
                    color: '#370617',
                  }}
                >
                  {vibe.tags.join(' • ')}
                </p>
              )}
            </div>
            <Music className="w-6 h-6" style={{ color: '#03071E' }} />
          </motion.div>
        </div>

        {/* Now Playing Bar (SDK mode) */}
        {useSDK && nowPlaying && !isLoading && (
          <div className="px-6 pb-4">
            <div
              className="rounded-xl p-3 flex items-center justify-between"
              style={{
                backgroundColor: 'rgba(252, 191, 73, 0.15)',
                border: '1px solid rgba(252, 191, 73, 0.3)',
              }}
            >
              <div className="flex-1 min-w-0">
                <p style={{ fontSize: '12px', color: '#FCBF49', opacity: 0.8 }}>
                  {sdkPlayer.isCrossfading ? 'Crossfading...' : 'Now Playing'}
                </p>
                <p
                  className="truncate"
                  style={{ fontSize: '14px', color: '#EAE2B7', fontWeight: 600 }}
                >
                  {nowPlaying.name}
                </p>
                {/* Progress bar */}
                {sdkPlayer.duration > 0 && (
                  <div className="mt-2">
                    <div
                      className="h-1 rounded-full overflow-hidden"
                      style={{ backgroundColor: 'rgba(255, 255, 255, 0.2)' }}
                    >
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${(sdkPlayer.currentTime / sdkPlayer.duration) * 100}%`,
                          backgroundColor: '#FCBF49'
                        }}
                      />
                    </div>
                    <div className="flex justify-between mt-1">
                      <span style={{ fontSize: '10px', color: '#EAE2B7', opacity: 0.6 }}>
                        {formatTime(sdkPlayer.currentTime)}
                      </span>
                      <span style={{ fontSize: '10px', color: '#EAE2B7', opacity: 0.6 }}>
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
                className="ml-3 p-3 rounded-full"
                style={{
                  backgroundColor: 'rgba(252, 191, 73, 0.3)',
                  color: '#FCBF49'
                }}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                {sdkPlayer.isPlaying ? (
                  <Pause className="w-5 h-5" />
                ) : (
                  <Play className="w-5 h-5" />
                )}
              </motion.button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="px-6 pb-6 flex flex-col items-center justify-center" style={{ minHeight: '300px' }}>
            <motion.div
              className="w-12 h-12 border-4 border-#FCBF49 border-t-transparent rounded-full mb-4"
              animate={{ rotate: 360 }}
              transition={{
                duration: 1,
                repeat: Infinity,
                ease: "linear"
              }}
            />
            <p style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '14px',
              color: '#EAE2B7',
              opacity: 0.8
            }}>
              {isPremium && !sdkPlayer.isReady ? 'Connecting to Spotify...' : 'loading your queue...'}
            </p>
          </div>
        )}

        {/* Error State */}
        {error && !isLoading && (
          <div className="px-6 pb-6">
            <div
              className="rounded-xl p-4"
              style={{
                backgroundColor: 'rgba(214, 40, 40, 0.3)',
                border: '1px solid rgba(214, 40, 40, 0.5)'
              }}
            >
              <p style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '14px',
                color: '#EAE2B7',
                textAlign: 'center'
              }}>
                {error}
              </p>
            </div>
          </div>
        )}

        {/* SDK Error Notice */}
        {isPremium && sdkPlayer.sdkError && !isLoading && (
          <div className="px-6 pb-3">
            <div
              className="rounded-lg p-2"
              style={{
                backgroundColor: 'rgba(247, 127, 0, 0.2)',
                border: '1px solid rgba(247, 127, 0, 0.4)'
              }}
            >
              <p style={{ fontSize: '11px', color: '#F77F00', textAlign: 'center' }}>
                SDK: {sdkPlayer.sdkError} — using remote playback
              </p>
            </div>
          </div>
        )}

        {/* Track Queue */}
        {!isLoading && !error && tracks.length > 0 && (
          <div className="px-6 pb-6">
            <div className="flex items-center justify-between mb-3">
              <h2
                style={{
                  fontFamily: 'Poppins, sans-serif',
                  fontSize: '18px',
                  fontWeight: 700,
                  color: '#EAE2B7',
                }}
              >
                Your Queue
              </h2>
              <button
                onClick={handleNext}
                disabled={nowPlayingIndex >= tracks.length - 1}
                className="flex items-center gap-2 px-4 py-2 rounded-full transition-all disabled:opacity-50"
                style={{
                  fontFamily: 'Poppins, sans-serif',
                  background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)',
                  color: 'white',
                  fontSize: '14px',
                  fontWeight: 600,
                }}
              >
                <SkipForward className="w-4 h-4" />
                Next
              </button>
            </div>
            <div className="space-y-3">
              {tracks.map((track, index) => (
                <motion.div
                  key={track.track_id || track.id || index}
                  className="rounded-xl p-4 flex items-center gap-3"
                  style={{
                    backgroundColor: index === nowPlayingIndex
                      ? 'rgba(252, 191, 73, 0.2)'
                      : 'rgba(0, 48, 73, 0.6)',
                    border: index === nowPlayingIndex
                      ? '1px solid rgba(252, 191, 73, 0.5)'
                      : '1px solid rgba(252, 191, 73, 0.2)',
                  }}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  {/* Album Art or Placeholder */}
                  <div
                    className="rounded-lg flex-shrink-0 relative"
                    style={{
                      width: '50px',
                      height: '50px',
                      background: track.images && track.images.length > 0
                        ? `url(${track.images[0].url}) center/cover`
                        : `linear-gradient(135deg, ${vibe.color}80 0%, ${vibe.color}40 100%)`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      overflow: 'hidden'
                    }}
                  >
                    {!track.images || track.images.length === 0 ? (
                      <Music className="w-6 h-6" style={{ color: '#03071E', opacity: 0.5 }} />
                    ) : null}
                    {track.rank && (
                      <div
                        className="absolute top-0 right-0 rounded-bl-lg px-1.5 py-0.5"
                        style={{
                          backgroundColor: 'rgba(0, 0, 0, 0.7)',
                          fontSize: '10px',
                          fontWeight: 700,
                          color: '#FCBF49',
                          fontFamily: 'Poppins, sans-serif'
                        }}
                      >
                        #{track.rank}
                      </div>
                    )}
                  </div>

                  {/* Song Info */}
                  <div className="flex-1 min-w-0">
                    <h3
                      className="truncate"
                      style={{
                        fontFamily: 'Poppins, sans-serif',
                        fontSize: '14px',
                        fontWeight: 600,
                        color: '#EAE2B7',
                        marginBottom: '2px'
                      }}
                    >
                      {track.name || 'Unknown Track'}
                    </h3>
                    <p
                      className="truncate"
                      style={{
                        fontFamily: 'Poppins, sans-serif',
                        fontSize: '12px',
                        color: '#EAE2B7',
                        opacity: 0.7
                      }}
                    >
                      {track.artist_names || track.artists || 'Unknown Artist'}
                    </p>
                    {track.tempo && (
                      <p
                        style={{
                          fontFamily: 'Poppins, sans-serif',
                          fontSize: '10px',
                          color: '#FCBF49',
                          opacity: 0.8,
                          marginTop: '2px'
                        }}
                      >
                        {Math.round(track.tempo)} BPM
                      </p>
                    )}
                  </div>

                  {/* Duration */}
                  {track.duration_ms && (
                    <span
                      style={{
                        fontFamily: 'Poppins, sans-serif',
                        fontSize: '12px',
                        color: '#EAE2B7',
                        marginRight: '8px',
                        opacity: 0.7
                      }}
                    >
                      {formatDuration(track.duration_ms)}
                    </span>
                  )}

                  {/* Play / Pause */}
                  <motion.button
                    className="rounded-full p-2 flex-shrink-0 transition-all"
                    style={{
                      background: index === nowPlayingIndex && useSDK && sdkPlayer.isPlaying
                        ? 'linear-gradient(135deg, #F77F00 0%, #D62828 100%)'
                        : 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)',
                    }}
                    onClick={() => handlePlayPause(track, index)}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    {index === nowPlayingIndex && useSDK && sdkPlayer.isPlaying ? (
                      <Pause className="w-4 h-4 text-white fill-white" />
                    ) : (
                      <Play className="w-4 h-4 text-white fill-white" />
                    )}
                  </motion.button>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
