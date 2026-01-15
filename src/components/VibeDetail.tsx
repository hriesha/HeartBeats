import { motion } from 'motion/react';
import { ChevronLeft, Play, Music } from 'lucide-react';
import { VibeType } from '../App';
import { useState, useEffect } from 'react';
import { getClusterTracks, getTrackDetails, Track } from '../utils/api';

interface VibeDetailProps {
  vibe: VibeType;
  bpm?: number;
  onBack: () => void;
}

export function VibeDetail({ vibe, bpm = 120, onBack }: VibeDetailProps) {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Extract cluster ID from vibe ID (format: "cluster-0")
  const clusterId = parseInt(vibe.id.split('-')[1] || '0');

  useEffect(() => {
    const fetchTracks = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Get KNN-matched tracks for this cluster
        const clusterData = await getClusterTracks(clusterId, bpm, 10);

        if (clusterData && clusterData.tracks && clusterData.tracks.length > 0) {
          // Extract track IDs
          const trackIds = clusterData.tracks.map(t => t.track_id).filter(Boolean);

          // Get detailed track information from Spotify
          const detailsResponse = await getTrackDetails(trackIds);

          if (detailsResponse && detailsResponse.tracks) {
            // Merge KNN data with Spotify details
            const mergedTracks = clusterData.tracks.map(knnTrack => {
              const spotifyTrack = detailsResponse.tracks.find(
                st => st.id === knnTrack.track_id || st.track_id === knnTrack.track_id
              );

              return {
                ...knnTrack,
                ...spotifyTrack,
                // Keep KNN metadata
                rank: knnTrack.rank,
                distance: knnTrack.distance,
                tempo: knnTrack.tempo,
              };
            });

            setTracks(mergedTracks);
          } else {
            // Fallback: use KNN tracks without Spotify details
            setTracks(clusterData.tracks);
          }
        } else {
          setError('No tracks found for this cluster. Please try another vibe.');
        }
      } catch (err) {
        console.error('Error fetching tracks:', err);
        setError('Failed to load tracks. Please try again.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchTracks();
  }, [clusterId, bpm]);

  const formatDuration = (ms?: number): string => {
    if (!ms) return '0:00';
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
            </div>
            <Music className="w-6 h-6" style={{ color: '#03071E' }} />
          </motion.div>
        </div>

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
              loading your queue...
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

        {/* Track Queue */}
        {!isLoading && !error && tracks.length > 0 && (
          <div className="px-6 pb-6">
            <h2
              className="mb-3"
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '18px',
                fontWeight: 700,
                color: '#EAE2B7',
              }}
            >
              Your Queue
            </h2>
            <div className="space-y-3">
              {tracks.map((track, index) => (
                <motion.div
                  key={track.track_id || track.id || index}
                  className="rounded-xl p-4 flex items-center gap-3"
                  style={{
                    backgroundColor: 'rgba(0, 48, 73, 0.6)',
                    border: '1px solid rgba(252, 191, 73, 0.2)',
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

                  {/* Play Button */}
                  <button
                    className="rounded-full p-2 flex-shrink-0 transition-all"
                    style={{
                      background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)',
                    }}
                    onClick={() => {
                      if (track.preview_url) {
                        window.open(track.preview_url, '_blank');
                      } else if (track.external_urls) {
                        window.open(track.external_urls, '_blank');
                      }
                    }}
                  >
                    <Play className="w-4 h-4 text-white fill-white" />
                  </button>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
