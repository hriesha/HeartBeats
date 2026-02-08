import React, { useCallback } from 'react';
import { motion } from 'motion/react';
import { Play, Pause, Music, ExternalLink, SkipForward } from 'lucide-react';
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

  // Handle crossfade when track index changes due to auto-advance
  const prevIndexRef = React.useRef<number>(-1);
  React.useEffect(() => {
    const prevIndex = prevIndexRef.current;
    prevIndexRef.current = currentTrackIndex;

    // Only crossfade if this was an auto-advance (track ended)
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
      // Play different track - instant skip (no crossfade)
      const target = getPlaybackTarget(track);
      if (target) {
        player.skipTo(target).then(() => {
          setCurrentTrackIndex(index);
        }).catch((err) => {
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
        // Manual skip - instant switch, no crossfade
        player.skipTo(target).then(() => {
          setCurrentTrackIndex(nextIndex);
        }).catch((err) => {
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
    <div className="relative w-full h-full overflow-auto" style={{ fontFamily: 'Poppins, sans-serif' }}>
      {/* Background with gradient overlay */}
      <div
        className="absolute inset-0 z-0"
        style={{
          background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
        }}
      />

      {/* Content */}
      <div className="relative z-10 w-full h-full flex flex-col px-6 py-8">
        {/* Header */}
        <div className="mb-6">
          <button
            onClick={onBack}
            className="mb-4 p-2 rounded-full transition-all"
            style={{
              backgroundColor: 'rgba(0, 48, 73, 0.8)',
              color: '#FCBF49'
            }}
          >
            ← Back
          </button>

          <h1
            className="mb-2"
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontWeight: 700,
              fontSize: '28px',
              color: '#EAE2B7',
              textShadow: '0 2px 8px rgba(252, 191, 73, 0.4)'
            }}
          >
            Your Queue
          </h1>
          <p
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '14px',
              color: '#FCBF49',
              fontWeight: 500
            }}
          >
            Cluster {clusterId} • {bpm} BPM • {tracks.length} songs
            {useSDK && ' • Full tracks'}
          </p>

          {/* Now Playing & Skip Controls */}
          {currentTrack && (
            <div
              className="mt-4 p-3 rounded-xl flex items-center justify-between"
              style={{
                backgroundColor: 'rgba(252, 191, 73, 0.15)',
                border: '1px solid rgba(252, 191, 73, 0.3)',
              }}
            >
              <div className="flex-1 min-w-0">
                <p style={{ fontSize: '12px', color: '#FCBF49', opacity: 0.8 }}>
                  {player.isCrossfading ? 'Crossfading to...' : 'Now Playing'}
                </p>
                <p
                  className="truncate"
                  style={{ fontSize: '14px', color: '#EAE2B7', fontWeight: 600 }}
                >
                  {currentTrack.name}
                </p>
                {/* Progress bar */}
                {player.duration > 0 && (
                  <div
                    className="mt-2 h-1 rounded-full overflow-hidden"
                    style={{ backgroundColor: 'rgba(255, 255, 255, 0.2)' }}
                  >
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${(player.currentTime / player.duration) * 100}%`,
                        backgroundColor: '#FCBF49'
                      }}
                    />
                  </div>
                )}
              </div>

              {/* Skip Next Button */}
              <motion.button
                onClick={handleSkipNext}
                disabled={currentTrackIndex >= tracks.length - 1}
                className="ml-4 p-3 rounded-full disabled:opacity-40"
                style={{
                  backgroundColor: 'rgba(252, 191, 73, 0.3)',
                  color: '#FCBF49'
                }}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <SkipForward className="w-5 h-5" />
              </motion.button>
            </div>
          )}
        </div>

        {/* Song List */}
        <div className="flex-1 overflow-auto space-y-3">
          {tracks.length === 0 ? (
            <div className="text-center py-12">
              <Music className="w-16 h-16 mx-auto mb-4" style={{ color: '#EAE2B7', opacity: 0.5 }} />
              <p style={{ color: '#EAE2B7', opacity: 0.7 }}>No tracks found</p>
            </div>
          ) : (
            tracks.map((track, index) => (
              <motion.div
                key={track.track_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="rounded-xl p-4 flex items-center gap-4"
                style={{
                  backgroundColor: index === currentTrackIndex
                    ? 'rgba(252, 191, 73, 0.2)'
                    : 'rgba(0, 48, 73, 0.6)',
                  border: index === currentTrackIndex
                    ? '1px solid rgba(252, 191, 73, 0.5)'
                    : '1px solid rgba(252, 191, 73, 0.2)',
                  backdropFilter: 'blur(10px)'
                }}
              >
                {/* Album Art */}
                <div className="flex-shrink-0 w-16 h-16 rounded-lg overflow-hidden bg-gray-800 flex items-center justify-center">
                  {track.images && track.images.length > 0 ? (
                    <img
                      src={track.images[track.images.length - 1].url}
                      alt={track.album || track.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <Music className="w-8 h-8" style={{ color: '#FCBF49', opacity: 0.5 }} />
                  )}
                </div>

                {/* Track Info */}
                <div className="flex-1 min-w-0">
                  <h3
                    className="truncate mb-1"
                    style={{
                      fontFamily: 'Poppins, sans-serif',
                      fontSize: '16px',
                      fontWeight: 600,
                      color: '#EAE2B7'
                    }}
                  >
                    {track.name}
                  </h3>
                  <p
                    className="truncate mb-1"
                    style={{
                      fontFamily: 'Poppins, sans-serif',
                      fontSize: '14px',
                      color: '#EAE2B7',
                      opacity: 0.8
                    }}
                  >
                    {track.artist_names || track.artists}
                  </p>
                  <div className="flex items-center gap-3 text-xs" style={{ color: '#FCBF49', opacity: 0.7 }}>
                    <span>{Math.round(track.tempo)} BPM</span>
                    <span>•</span>
                    <span>{formatDuration(track.duration_ms)}</span>
                    {track.rank && <span>•</span>}
                    {track.rank && <span>#{track.rank}</span>}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2">
                  {isPlayable(track) && (
                    <motion.button
                      onClick={() => handlePlayPause(track, index)}
                      className="p-3 rounded-full"
                      style={{
                        backgroundColor: playingTrackId === track.track_id ? '#FCBF49' : 'rgba(252, 191, 73, 0.2)',
                        color: playingTrackId === track.track_id ? '#03071E' : '#FCBF49'
                      }}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      {playingTrackId === track.track_id && player.isPlaying ? (
                        <Pause className="w-5 h-5" />
                      ) : (
                        <Play className="w-5 h-5" />
                      )}
                    </motion.button>
                  )}
                  {track.external_urls && (
                    <motion.a
                      href={track.external_urls}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-3 rounded-full"
                      style={{
                        backgroundColor: 'rgba(252, 191, 73, 0.2)',
                        color: '#FCBF49'
                      }}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <ExternalLink className="w-5 h-5" />
                    </motion.a>
                  )}
                </div>
              </motion.div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
