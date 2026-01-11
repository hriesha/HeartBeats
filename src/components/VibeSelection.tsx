import { motion } from 'motion/react';
import { ChevronLeft } from 'lucide-react';
import { useState, useEffect } from 'react';
import { heartbeatsApi, Cluster } from '../services/heartbeatsApi';
import { VibeType } from '../App';

interface VibeSelectionProps {
  bpm: number;
  onVibeSelect: (vibe: VibeType) => void;
  onBack: () => void;
}

// Color palette for clusters
const CLUSTER_COLORS = ['#EAE2B7', '#FCBF49', '#F77F00', '#D62828', '#003049', '#D62828'];

// Tag mappings based on tempo and energy
const getTagsForCluster = (meanTempo: number, meanEnergy: number): string[] => {
  if (meanTempo < 100) {
    return meanEnergy < 0.5 ? ['calm', 'relaxing'] : ['chill', 'ambient'];
  } else if (meanTempo < 130) {
    return meanEnergy < 0.6 ? ['focused', 'steady'] : ['upbeat', 'motivating'];
  } else {
    return meanEnergy < 0.7 ? ['energetic', 'pumping'] : ['intense', 'high-energy'];
  }
};

const getClusterName = (meanTempo: number, meanEnergy: number, index: number): string => {
  if (meanTempo < 100) {
    return meanEnergy < 0.5 ? 'Chill Flow' : 'Ambient Vibes';
  } else if (meanTempo < 130) {
    return meanEnergy < 0.6 ? 'Focus Pulse' : 'Steady Groove';
  } else {
    return meanEnergy < 0.7 ? 'Energy Rush' : 'Intense Beats';
  }
};

export function VibeSelection({ bpm, onVibeSelect, onBack }: VibeSelectionProps) {
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [selectedVibe, setSelectedVibe] = useState<VibeType | null>(null);
  const [clusters, setClusters] = useState<VibeType[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch clusters when component mounts
    const fetchClusters = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await heartbeatsApi.getClusters(undefined, 4);
        
        if (response.success && response.data) {
          const vibeClusters: VibeType[] = response.data.clusters.map((cluster: Cluster, index: number) => ({
            id: `cluster-${cluster.cluster_id}`,
            name: getClusterName(cluster.mean_tempo, cluster.mean_energy, cluster.cluster_id),
            color: CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length],
            tags: getTagsForCluster(cluster.mean_tempo, cluster.mean_energy),
            clusterId: cluster.cluster_id,
            meanTempo: cluster.mean_tempo,
          }));
          setClusters(vibeClusters);
        } else {
          setError(response.error || 'Failed to load clusters');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchClusters();
  }, []);

  const handleVibeClick = (vibe: VibeType) => {
    setSelectedVibe(vibe);
    setIsTransitioning(true);
    // Wait for transition before navigating
    setTimeout(() => {
      onVibeSelect(vibe);
    }, 800);
  };

  // Show loading state
  if (loading) {
    return (
      <div className="relative w-full h-full overflow-auto flex items-center justify-center" style={{ fontFamily: 'Poppins, sans-serif' }}>
        <div 
          className="absolute inset-0 z-0"
          style={{
            background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
          }}
        />
        <div className="relative z-10 text-center">
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
          <p style={{ color: '#EAE2B7', fontSize: '16px' }}>Loading your vibes...</p>
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="relative w-full h-full overflow-auto flex items-center justify-center" style={{ fontFamily: 'Poppins, sans-serif' }}>
        <div 
          className="absolute inset-0 z-0"
          style={{
            background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
          }}
        />
        <div className="relative z-10 text-center px-6">
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
      </div>
    );
  }

  return (
    <div className="relative w-full h-full overflow-auto" style={{ fontFamily: 'Poppins, sans-serif' }}>
      {/* Background with gradient overlay */}
      <div 
        className="absolute inset-0 z-0"
        style={{
          background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
        }}
      />
      
      {/* Zoom Transition Overlay */}
      {isTransitioning && selectedVibe && (
        <motion.div
          className="absolute inset-0 z-50 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <motion.div
            className="rounded-full"
            style={{
              backgroundColor: selectedVibe.color,
              width: '155px',
              height: '155px'
            }}
            initial={{ scale: 1 }}
            animate={{ scale: 20 }}
            transition={{ duration: 0.8, ease: "easeInOut" }}
          />
        </motion.div>
      )}
      
      {/* Content */}
      <div className="relative z-10 w-full h-full flex flex-col px-6 py-12">
        {/* Back Button */}
        <button
          onClick={onBack}
          className="absolute top-4 left-4 p-2 rounded-full transition-all"
          style={{
            backgroundColor: 'rgba(0, 48, 73, 0.8)',
            color: '#FCBF49'
          }}
        >
          <ChevronLeft className="w-6 h-6" />
        </button>

        {/* Header */}
        <div className="mt-8 mb-8">
          <p 
            className="mb-2"
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '14px',
              color: '#FCBF49',
              fontWeight: 500
            }}
          >
            based on {bpm} bpm
          </p>
          <h1 
            className="mb-2"
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontWeight: 700,
              fontSize: '32px',
              color: '#EAE2B7',
              textShadow: '0 2px 8px rgba(252, 191, 73, 0.4)'
            }}
          >
            Choose your vibe
          </h1>
          <p 
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '16px',
              color: '#EAE2B7',
              fontWeight: 400,
              opacity: 0.8,
              lineHeight: 1.5
            }}
          >
            we've clustered your tracks into {clusters.length} moods. pick how you want to feel.
          </p>
        </div>

        {/* Vibe Bubbles */}
        <div className="flex-1 flex items-center justify-center">
          <div className="relative w-full" style={{ height: '400px' }}>
            {clusters.map((vibe, index) => {
              const positions = [
                { top: '10%', left: '5%' },
                { top: '15%', right: '8%' },
                { bottom: '20%', left: '12%' },
                { bottom: '15%', right: '5%' },
              ];
              const position = positions[index % positions.length];
              const delays = [0, 0.5, 1, 1.5];
              
              return (
                <motion.button
                  key={vibe.id}
                  onClick={() => handleVibeClick(vibe)}
                  className="absolute"
                  style={{
                    ...position,
                    width: '140px',
                    height: '140px',
                    borderRadius: '50%',
                    backgroundColor: vibe.color,
                    boxShadow: `0 8px 24px ${vibe.color}80, inset 0 2px 8px rgba(255, 255, 255, 0.3)`,
                    border: 'none',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '16px'
                  }}
                  animate={{
                    y: [0, index % 2 === 0 ? -15 : 20, 0],
                    x: [0, index % 2 === 0 ? 10 : -8, 0],
                  }}
                  transition={{
                    duration: 4 + index * 0.5,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: delays[index] || 0
                  }}
                  whileHover={{ 
                    scale: 1.1,
                    boxShadow: `0 12px 32px ${vibe.color}CC, inset 0 2px 8px rgba(255, 255, 255, 0.3)`
                  }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span style={{ fontFamily: 'Poppins, sans-serif', fontSize: '16px', fontWeight: 700, color: '#03071E', textAlign: 'center', lineHeight: 1.2 }}>
                    {vibe.name}
                  </span>
                  <span style={{ fontFamily: 'Poppins, sans-serif', fontSize: '11px', fontWeight: 500, color: '#370617', marginTop: '4px' }}>
                    {vibe.tags.join(' • ')}
                  </span>
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Bottom Hint */}
        <div className="text-center">
          <p 
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '14px',
              color: '#FFDAB9',
              fontWeight: 500,
              opacity: 0.7
            }}
          >
            tap a vibe bubble to dive in
          </p>
        </div>
      </div>
    </div>
  );
}
