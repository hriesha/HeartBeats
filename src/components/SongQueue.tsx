import React, { useCallback } from 'react';
import { motion } from 'motion/react';
import { Play, Pause, Music, ExternalLink, SkipForward } from 'lucide-react';
import { Track } from '../services/heartbeatsApi';
import { useCrossfade } from '../hooks/useCrossfade';

interface SongQueueProps {
  tracks: Track[];
  clusterId: number;
  bpm: number;
  onBack: () => void;
}

export function SongQueue({ tracks, clusterId, bpm, onBack }: SongQueueProps) {
  const [currentTrackIndex, setCurrentTrackIndex] = React.useState<number>(-1);

  // Get the current track based on index
  const currentTrack = currentTrackIndex >= 0 ? tracks[currentTrackIndex] : null;
  const playingTrackId = currentTrack?.track_id || null;

  // Handle track end - crossfade to next track
  const handleTrackEnd = useCallback(() => {
    setCurrentTrackIndex((prevIndex) => {
      const nextIndex = prevIndex + 1;
      if (nextIndex < tracks.length && tracks[nextIndex]?.preview_url) {
        // Will trigger crossfade via useEffect below
        return nextIndex;
      }
      // No more tracks with preview URLs
      return -1;
    });
  }, [tracks]);

  // Initialize crossfade hook
  const {
    play,
    pause,
    resume,
    skipTo,
    crossfadeTo,
    isPlaying,
    isCrossfading,
    currentTime,
    duration,
    cleanup,
  } = useCrossfade({
    crossfadeDuration: 5000, // 5 second crossfade
    onTrackEnd: handleTrackEnd,
  });

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  // Handle crossfade when track index changes due to auto-advance
  const prevIndexRef = React.useRef<number>(-1);
  React.useEffect(() => {
    const prevIndex = prevIndexRef.current;
    prevIndexRef.current = currentTrackIndex;

    // Only crossfade if this was an auto-advance (track ended)
    if (
      currentTrackIndex >= 0 &&
      prevIndex >= 0 &&
      currentTrackIndex === prevIndex + 1 &&
      tracks[currentTrackIndex]?.preview_url
    ) {
      crossfadeTo(tracks[currentTrackIndex].preview_url!);
    }
  }, [currentTrackIndex, tracks, crossfadeTo]);

  const handlePlayPause = (track: Track, index: number) => {
    if (playingTrackId === track.track_id) {
      // Pause/resume current track
      if (isPlaying) {
        pause();
      } else {
        resume();
      }
    } else {
      // Play different track - instant skip (no crossfade)
      if (track.preview_url) {
        skipTo(track.preview_url).then(() => {
          setCurrentTrackIndex(index);
        }).catch((err) => {
          console.error('Error playing preview:', err);
        });
      }
    }
  };

  const handleSkipNext = () => {
    const nextIndex = currentTrackIndex + 1;
    if (nextIndex < tracks.length && tracks[nextIndex]?.preview_url) {
      // Manual skip - instant switch, no crossfade
      skipTo(tracks[nextIndex].preview_url!).then(() => {
        setCurrentTrackIndex(nextIndex);
      }).catch((err) => {
        console.error('Error skipping to next:', err);
      });
    }
  };

  const formatDuration = (ms?: number) => {
    if (!ms) return '--:--';
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
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
                  {isCrossfading ? 'Crossfading to...' : 'Now Playing'}
                </p>
                <p
                  className="truncate"
                  style={{ fontSize: '14px', color: '#EAE2B7', fontWeight: 600 }}
                >
                  {currentTrack.name}
                </p>
                {/* Progress bar */}
                {duration > 0 && (
                  <div
                    className="mt-2 h-1 rounded-full overflow-hidden"
                    style={{ backgroundColor: 'rgba(255, 255, 255, 0.2)' }}
                  >
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${(currentTime / duration) * 100}%`,
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
                  {track.preview_url && (
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
                      {playingTrackId === track.track_id && isPlaying ? (
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
