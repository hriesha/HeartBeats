import React from 'react';
import { motion } from 'motion/react';
import { Play, Pause, Music, ExternalLink } from 'lucide-react';
import { Track } from '../services/heartbeatsApi';

interface SongQueueProps {
  tracks: Track[];
  clusterId: number;
  bpm: number;
  onBack: () => void;
}

export function SongQueue({ tracks, clusterId, bpm, onBack }: SongQueueProps) {
  const [playingTrackId, setPlayingTrackId] = React.useState<string | null>(null);
  const [audio, setAudio] = React.useState<HTMLAudioElement | null>(null);

  React.useEffect(() => {
    return () => {
      // Cleanup audio on unmount
      if (audio) {
        audio.pause();
        audio.src = '';
      }
    };
  }, [audio]);

  const handlePlayPause = (track: Track) => {
    if (playingTrackId === track.track_id) {
      // Pause current track
      if (audio) {
        audio.pause();
        setAudio(null);
      }
      setPlayingTrackId(null);
    } else {
      // Play new track
      if (audio) {
        audio.pause();
      }

      if (track.preview_url) {
        const newAudio = new Audio(track.preview_url);
        newAudio.play().catch((err) => {
          console.error('Error playing preview:', err);
        });
        setAudio(newAudio);
        setPlayingTrackId(track.track_id);

        newAudio.onended = () => {
          setPlayingTrackId(null);
          setAudio(null);
        };
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
                  backgroundColor: 'rgba(0, 48, 73, 0.6)',
                  border: '1px solid rgba(252, 191, 73, 0.2)',
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
                      onClick={() => handlePlayPause(track)}
                      className="p-3 rounded-full"
                      style={{
                        backgroundColor: playingTrackId === track.track_id ? '#FCBF49' : 'rgba(252, 191, 73, 0.2)',
                        color: playingTrackId === track.track_id ? '#03071E' : '#FCBF49'
                      }}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      {playingTrackId === track.track_id ? (
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
