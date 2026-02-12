import React, { useCallback } from 'react';
import { motion } from 'motion/react';
import { Play, Pause, Music, ExternalLink, SkipForward, ChevronLeft } from 'lucide-react';
import { Track } from '../utils/api';
import { useCrossfade } from '../hooks/useCrossfade';
import { useSpotifyPlayer } from '../hooks/useSpotifyPlayer';

interface SongQueueProps {
  tracks: Track[];
  clusterId: number;
  bpm: number;
  onBack: () => void;
  isPremium?: boolean;
}

function toUri(track: Track): string {
  const id = track.id ?? track.track_id;
  return id?.startsWith('spotify:') ? id : `spotify:track:${id}`;
}

export function SongQueue({ tracks, clusterId, bpm, onBack, isPremium = false }: SongQueueProps) {
  const [currentTrackIndex, setCurrentTrackIndex] = React.useState<number>(-1);

  const currentTrack = currentTrackIndex >= 0 ? tracks[currentTrackIndex] : null;
  const playingTrackId = currentTrack?.track_id || null;

  // Handle track end - advance to next track
  const handleTrackEnd = useCallback(() => {
    setCurrentTrackIndex((prevIndex) => {
      const nextIndex = prevIndex + 1;
      if (nextIndex < tracks.length) {
        // SDK can play any track; preview mode needs preview_url
        if (isPremium || tracks[nextIndex]?.preview_url) {
          return nextIndex;
        }
      }
      return -1;
    });
  }, [tracks, isPremium]);

  // SDK player for Premium users
  const sdkPlayer = useSpotifyPlayer({
    crossfadeDuration: 5000,
    onTrackEnd: isPremium ? handleTrackEnd : undefined,
  });

  // Preview player for free-tier users
  const previewPlayer = useCrossfade({
    crossfadeDuration: 5000,
    onTrackEnd: !isPremium ? handleTrackEnd : undefined,
  });

  // Pick active player based on Premium status + SDK readiness
  const useSDK = isPremium && sdkPlayer.isReady;
  const player = useSDK ? sdkPlayer : previewPlayer;

  // Get the playback target for a track (URI for SDK, preview URL for free tier)
  const getPlaybackTarget = (track: Track): string | null => {
    if (useSDK) return toUri(track);
    return track.preview_url || null;
  };

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      sdkPlayer.cleanup();
      previewPlayer.cleanup();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Track whether the index change was a manual action (click/skip button)
  const isManualActionRef = React.useRef(false);
  const prevIndexRef = React.useRef<number>(-1);

  // Handle crossfade ONLY for auto-advance (track ended), not manual clicks
  React.useEffect(() => {
    const prevIndex = prevIndexRef.current;
    prevIndexRef.current = currentTrackIndex;

    // Manual actions (click, skip button) handle playback themselves — skip crossfade
    if (isManualActionRef.current) {
      isManualActionRef.current = false;
      return;
    }

    // Auto-advance: crossfade to the next track
    if (
      currentTrackIndex >= 0 &&
      prevIndex >= 0 &&
      currentTrackIndex === prevIndex + 1
    ) {
      const target = getPlaybackTarget(tracks[currentTrackIndex]);
      if (target) player.crossfadeTo(target);
    }
  }, [currentTrackIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  const handlePlayPause = (track: Track, index: number) => {
    if (playingTrackId === track.track_id) {
      // Pause/resume current track
      if (player.isPlaying) {
        player.pause();
      } else {
        player.resume();
      }
    } else {
      // Play different track with crossfade
      const target = getPlaybackTarget(track);
      if (target) {
        isManualActionRef.current = true;
        setCurrentTrackIndex(index);
        player.crossfadeTo(target).catch((err) => {
          console.error('Error playing track:', err);
        });
      }
    }
  };

  const handleSkipNext = () => {
    const nextIndex = currentTrackIndex + 1;
    if (nextIndex < tracks.length) {
      const target = getPlaybackTarget(tracks[nextIndex]);
      if (target) {
        isManualActionRef.current = true;
        setCurrentTrackIndex(nextIndex);
        player.crossfadeTo(target).catch((err) => {
          console.error('Error skipping to next:', err);
        });
      }
    }
  };

  const formatDuration = (ms?: number) => {
    if (!ms) return '--:--';
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Check if a track is playable (SDK can play all, preview needs preview_url)
  const isPlayable = (track: Track): boolean => {
    if (useSDK) return true;
    return !!track.preview_url;
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      {/* Header */}
      <div style={{ padding: '24px', paddingBottom: 0 }}>
        {/* Back Button */}
        <button
          onClick={onBack}
          style={{
            width: 40, height: 40, borderRadius: '50%',
            background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)',
            color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer', marginBottom: 20,
          }}
        >
          <ChevronLeft style={{ width: 20, height: 20 }} />
        </button>

        <h1 style={{
          fontFamily: 'var(--font-heading)', fontWeight: 200, fontSize: '28px',
          color: '#ffffff', letterSpacing: '0.1em', marginBottom: 8,
        }}>
          your queue
        </h1>
        <p style={{
          fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 300,
          color: 'rgba(255, 255, 255, 0.35)',
        }}>
          cluster {clusterId} · {bpm} BPM · {tracks.length} songs
          {useSDK && ' · full tracks'}
        </p>

        {/* Now Playing & Skip Controls */}
        {currentTrack && (
          <div style={{
            marginTop: 16, padding: '14px 16px', borderRadius: '14px',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            background: 'rgba(255, 45, 85, 0.06)',
            border: '1px solid rgba(255, 45, 85, 0.15)',
          }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <p style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: '#FF2D55', opacity: 0.8, fontWeight: 400 }}>
                {player.isCrossfading ? 'crossfading...' : 'now playing'}
              </p>
              <p style={{
                fontFamily: 'var(--font-body)', fontSize: '14px', color: '#ffffff', fontWeight: 500,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {currentTrack.name}
              </p>
              {/* Progress bar */}
              {player.duration > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{
                    height: 2, borderRadius: 1, overflow: 'hidden',
                    background: 'rgba(255, 255, 255, 0.08)',
                  }}>
                    <div style={{
                      height: '100%', borderRadius: 1, transition: 'width 0.3s',
                      width: `${(player.currentTime / player.duration) * 100}%`,
                      background: '#FF2D55',
                    }} />
                  </div>
                </div>
              )}
            </div>

            {/* Skip Next Button */}
            <motion.button
              onClick={handleSkipNext}
              disabled={currentTrackIndex >= tracks.length - 1}
              style={{
                marginLeft: 12, padding: 10, borderRadius: '50%', border: 'none',
                background: 'rgba(255, 45, 85, 0.15)', color: '#FF2D55',
                cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                opacity: currentTrackIndex >= tracks.length - 1 ? 0.4 : 1,
              }}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <SkipForward style={{ width: 18, height: 18 }} />
            </motion.button>
          </div>
        )}
      </div>

      {/* Song List */}
      <div style={{ flex: 1, overflow: 'auto', padding: '20px 24px 24px', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {tracks.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px 0' }}>
            <Music style={{ width: 48, height: 48, color: 'rgba(255, 45, 85, 0.2)', margin: '0 auto 16px' }} />
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '14px', color: 'rgba(255,255,255,0.35)', fontWeight: 300 }}>
              no tracks found
            </p>
          </div>
        ) : (
          tracks.map((track, index) => {
            const isNowPlaying = index === currentTrackIndex;
            return (
              <motion.div
                key={track.track_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.04 }}
                style={{
                  borderRadius: '14px', padding: '14px 16px',
                  display: 'flex', alignItems: 'center', gap: 14,
                  background: isNowPlaying ? 'rgba(255, 45, 85, 0.08)' : 'rgba(255, 255, 255, 0.03)',
                  border: isNowPlaying ? '1px solid rgba(255, 45, 85, 0.2)' : '1px solid rgba(255, 255, 255, 0.06)',
                }}
              >
                {/* Album Art */}
                <div style={{
                  flexShrink: 0, width: 56, height: 56, borderRadius: '10px', overflow: 'hidden',
                  background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.06)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {track.images && track.images.length > 0 ? (
                    <img
                      src={track.images[track.images.length - 1].url}
                      alt={track.album || track.name}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  ) : (
                    <Music style={{ width: 24, height: 24, color: 'rgba(255, 45, 85, 0.3)' }} />
                  )}
                </div>

                {/* Track Info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <h3 style={{
                    fontFamily: 'var(--font-body)', fontSize: '15px', fontWeight: 500,
                    color: '#ffffff', marginBottom: 2,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {track.name}
                  </h3>
                  <p style={{
                    fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 300,
                    color: 'rgba(255, 255, 255, 0.4)', marginBottom: 2,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {track.artist_names || track.artists}
                  </p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: '#FF2D55', opacity: 0.6, fontWeight: 400 }}>
                      {Math.round(track.tempo ?? 0)} BPM
                    </span>
                    <span style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'rgba(255,255,255,0.2)' }}>·</span>
                    <span style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontWeight: 300 }}>
                      {formatDuration(track.duration_ms)}
                    </span>
                    {track.rank && (
                      <>
                        <span style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'rgba(255,255,255,0.2)' }}>·</span>
                        <span style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: '#FF2D55', opacity: 0.5, fontWeight: 400 }}>
                          #{track.rank}
                        </span>
                      </>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  {isPlayable(track) && (
                    <motion.button
                      onClick={() => handlePlayPause(track, index)}
                      style={{
                        padding: 10, borderRadius: '50%', border: 'none', cursor: 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        background: playingTrackId === track.track_id && player.isPlaying
                          ? '#FF2D55'
                          : 'rgba(255, 45, 85, 0.12)',
                        color: playingTrackId === track.track_id && player.isPlaying
                          ? '#ffffff'
                          : '#FF2D55',
                      }}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      {playingTrackId === track.track_id && player.isPlaying ? (
                        <Pause style={{ width: 16, height: 16 }} />
                      ) : (
                        <Play style={{ width: 16, height: 16 }} />
                      )}
                    </motion.button>
                  )}
                  {track.external_urls && (
                    <motion.a
                      href={track.external_urls}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        padding: 10, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        background: 'rgba(255, 255, 255, 0.05)', color: 'rgba(255, 255, 255, 0.3)',
                        textDecoration: 'none',
                      }}
                      whileHover={{ scale: 1.1, color: '#FF2D55' }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <ExternalLink style={{ width: 16, height: 16 }} />
                    </motion.a>
                  )}
                </div>
              </motion.div>
            );
          })
        )}
      </div>
    </div>
  );
}
