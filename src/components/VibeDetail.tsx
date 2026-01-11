import { motion } from 'motion/react';
import { ChevronLeft, Music } from 'lucide-react';
import { useState, useEffect } from 'react';
import { VibeType } from '../App';
import { heartbeatsApi, Track } from '../services/heartbeatsApi';
import { SongQueue } from './SongQueue';

interface VibeDetailProps {
  vibe: VibeType;
  bpm: number;
  onBack: () => void;
}

export function VibeDetail({ vibe, bpm, onBack }: VibeDetailProps) {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showQueue, setShowQueue] = useState(false);

  useEffect(() => {
    // Fetch tracks for this cluster when component mounts
    const fetchTracks = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // First get track IDs using KNN
        const tracksResponse = await heartbeatsApi.getTracks(bpm, vibe.clusterId, 10);
        
        if (tracksResponse.success && tracksResponse.data) {
          const trackIds = tracksResponse.data.tracks.map(t => t.track_id);
          
          // Then get full track details from Spotify
          const detailsResponse = await heartbeatsApi.getTrackDetails(trackIds);
          
          if (detailsResponse.success && detailsResponse.data) {
            // Merge metadata
            const tracksMap = new Map(tracksResponse.data.tracks.map(t => [t.track_id, t]));
            const mergedTracks = detailsResponse.data.tracks.map(detail => ({
              ...tracksMap.get(detail.id || '') || {},
              ...detail,
            })) as Track[];
            
            setTracks(mergedTracks);
            setShowQueue(true);
          } else {
            // Fallback to basic track info if Spotify API fails
            setTracks(tracksResponse.data.tracks);
            setShowQueue(true);
          }
        } else {
          setError(tracksResponse.error || 'Failed to load tracks');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchTracks();
  }, [vibe.clusterId, bpm]);

  if (showQueue && tracks.length > 0) {
    return (
      <SongQueue 
        tracks={tracks} 
        clusterId={vibe.clusterId} 
        bpm={bpm}
        onBack={onBack}
      />
    );
  }

  return (
    <div className="relative w-full h-full overflow-auto flex items-center justify-center" style={{ fontFamily: 'Poppins, sans-serif' }}>
      {/* Background with gradient overlay */}
      <div 
        className="absolute inset-0 z-0"
        style={{
          background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
        }}
      />
      
      {/* Content */}
      <div className="relative z-10 w-full h-full flex items-center justify-center px-6">
        {loading ? (
          <div className="text-center">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              className="w-16 h-16 mx-auto mb-4"
              style={{
                border: '4px solid rgba(252, 191, 73, 0.3)',
                borderTopColor: '#FCBF49',
                borderRadius: '50%'
              }}
            />
            <p style={{ color: '#EAE2B7', fontSize: '16px' }}>Loading your queue...</p>
          </div>
        ) : error ? (
          <div className="text-center">
            <p style={{ color: '#D62828', fontSize: '16px', marginBottom: '16px' }}>Error: {error}</p>
            <button
              onClick={onBack}
              className="px-6 py-3 rounded-xl"
              style={{
                backgroundColor: '#FCBF49',
                color: '#03071E',
                fontFamily: 'Poppins, sans-serif',
                fontWeight: 600
              }}
            >
              Go Back
            </button>
          </div>
        ) : (
          <div className="text-center">
            <Music className="w-16 h-16 mx-auto mb-4" style={{ color: '#EAE2B7', opacity: 0.5 }} />
            <p style={{ color: '#EAE2B7', opacity: 0.7 }}>No tracks found</p>
          </div>
        )}
      </div>
    </div>
  );
}
