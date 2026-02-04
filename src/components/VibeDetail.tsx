import { motion } from 'motion/react';
import { ChevronLeft, Play, Music, SkipForward } from 'lucide-react';
import { VibeType } from '../App';
import { useState, useEffect, useCallback } from 'react';
import { getClusterTracks, getTrackDetails, getTracksFromTrack, startPlayback, Track } from '../utils/api';

interface VibeDetailProps {
  vibe: VibeType;
  bpm?: number;
  onBack: () => void;
}

function toUri(t: Track): string {
  const id = t.id ?? t.track_id;
  return id?.startsWith('spotify:') ? id : `spotify:track:${id}`;
}

function mergeWithDetails(knnTracks: Track[], details: Track[]): Track[] {
  return knnTracks.map(knn => {
    const spotify = details.find(
      s => (s.id ?? s.track_id) === (knn.id ?? knn.track_id)
    );
    return { ...knn, ...spotify, rank: knn.rank, distance: knn.distance, tempo: knn.tempo };
  });
}

export function VibeDetail({ vibe, bpm = 120, onBack }: VibeDetailProps) {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [nowPlayingIndex, setNowPlayingIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const clusterId = parseInt(vibe.id.split('-')[1] || '0');

  const playTrack = useCallback(async (index: number) => {
    const t = tracks[index];
    if (!t) return;
    const res = await startPlayback([toUri(t)]);
    if (!res.success) console.warn('startPlayback failed:', res.error);
  }, [tracks]);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // Limit to 20 tracks per cluster to avoid Spotify API rate limits
        const clusterData = await getClusterTracks(clusterId, bpm, 20);
        if (!clusterData?.tracks?.length) {
          setError('No tracks found for this cluster. Try another vibe.');
          return;
        }
        // Tracks already include name and artists from dataset, no need for getTrackDetails
        const allClusterTracks = clusterData.tracks;

        if (cancelled) return;

        // Pick a random track to start playing
        const randomIdx = Math.floor(Math.random() * allClusterTracks.length);
        const randomTrack = allClusterTracks[randomIdx];
        const tid = randomTrack.track_id ?? randomTrack.id;
        if (tid) await startPlayback([toUri(randomTrack)]);

        // Use all cluster tracks, but put the random track first
        // Remove the random track from its original position and put it at the start
        const otherTracks = allClusterTracks.filter(
          (t, idx) => idx !== randomIdx
        );
        setTracks([randomTrack, ...otherTracks]);
        setNowPlayingIndex(0);
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

  const handleNext = async () => {
    const nextIdx = nowPlayingIndex + 1;
    if (nextIdx >= tracks.length) return;
    const next = tracks[nextIdx];
    const tid = next.track_id ?? next.id;
    await playTrack(nextIdx);
    setNowPlayingIndex(nextIdx);
    if (!tid) return;
    const from = await getTracksFromTrack(tid, clusterId, 10);
    if (!from?.tracks?.length) return;
    // Tracks already include metadata, no need for getTrackDetails
    const add = from.tracks.filter(m => (m.track_id ?? m.id) !== tid);
    setTracks(prev => {
      const rest = prev.slice(nextIdx + 1);
      const seen = new Set(rest.map(t => t.track_id ?? t.id));
      const extra = add.filter(m => !seen.has(m.track_id ?? m.id));
      return [...prev.slice(0, nextIdx + 1), ...extra, ...rest];
    });
  };

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

                  {/* Play / Open */}
                  <button
                    className="rounded-full p-2 flex-shrink-0 transition-all"
                    style={{
                      background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)',
                    }}
                    onClick={async () => {
                      const ok = await startPlayback([toUri(track)]);
                      if (ok.success) return;
                      if (track.preview_url) window.open(track.preview_url, '_blank');
                      else if (track.external_urls) window.open(track.external_urls, '_blank');
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
