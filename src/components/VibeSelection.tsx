import { motion } from 'motion/react';
import { ChevronLeft } from 'lucide-react';
import { VibeType } from '../App';
import { useState, useEffect } from 'react';
import { runClustering, Cluster } from '../utils/api';

interface VibeSelectionProps {
  paceValue: number;
  paceUnit: 'min/mile' | 'min/km';
  bpm?: number;
  onVibeSelect: (vibe: VibeType) => void;
  onBack: () => void;
}

// Bubble layout positions — arranged so they don't overlap, centered in viewport
const bubbleLayouts: { x: string; y: string; size: number }[][] = [
  // 1 cluster
  [{ x: '50%', y: '50%', size: 170 }],
  // 2 clusters
  [
    { x: '40%', y: '44%', size: 155 },
    { x: '62%', y: '56%', size: 145 },
  ],
  // 3 clusters
  [
    { x: '50%', y: '36%', size: 145 },
    { x: '35%', y: '58%', size: 135 },
    { x: '65%', y: '60%', size: 130 },
  ],
  // 4 clusters
  [
    { x: '40%', y: '36%', size: 135 },
    { x: '62%', y: '34%', size: 125 },
    { x: '36%', y: '60%', size: 130 },
    { x: '64%', y: '62%', size: 120 },
  ],
  // 5+ clusters
  [
    { x: '50%', y: '32%', size: 125 },
    { x: '32%', y: '46%', size: 115 },
    { x: '68%', y: '44%', size: 110 },
    { x: '38%', y: '64%', size: 115 },
    { x: '62%', y: '66%', size: 105 },
  ],
];

// Color palettes for each bubble
const vibeColors = [
  { bg: 'rgba(255, 45, 85, 0.06)', border: 'rgba(255, 45, 85, 0.25)', glow: 'rgba(255, 45, 85, 0.12)', text: '#FF6B8A', accent: '255, 45, 85' },
  { bg: 'rgba(100, 210, 255, 0.05)', border: 'rgba(100, 210, 255, 0.2)', glow: 'rgba(100, 210, 255, 0.1)', text: '#64D2FF', accent: '100, 210, 255' },
  { bg: 'rgba(48, 209, 88, 0.05)', border: 'rgba(48, 209, 88, 0.2)', glow: 'rgba(48, 209, 88, 0.1)', text: '#30D158', accent: '48, 209, 88' },
  { bg: 'rgba(255, 214, 10, 0.05)', border: 'rgba(255, 214, 10, 0.2)', glow: 'rgba(255, 214, 10, 0.1)', text: '#FFD60A', accent: '255, 214, 10' },
  { bg: 'rgba(191, 90, 242, 0.05)', border: 'rgba(191, 90, 242, 0.2)', glow: 'rgba(191, 90, 242, 0.1)', text: '#BF5AF2', accent: '191, 90, 242' },
];

export function VibeSelection({ paceValue, paceUnit, bpm, onVibeSelect, onBack }: VibeSelectionProps) {
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [selectedVibe, setSelectedVibe] = useState<VibeType | null>(null);
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchClusters = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const result = await runClustering(paceValue, paceUnit);
        console.log('Clustering result:', result);

        if (result && result.clusters) {
          if (result.clusters.length === 0) {
            setError(`No tracks found for this pace. Try a different pace.`);
          } else {
            setClusters(result.clusters);
          }
        } else {
          setError(result?.message || 'Failed to load clusters.');
        }
      } catch (err: any) {
        console.error('Error fetching clusters:', err);
        const errorMessage = err?.message || String(err);
        if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
          setError('Connection failed. Please make sure the server is running.');
        } else {
          setError(`Failed to load clusters: ${errorMessage}`);
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchClusters();
  }, [paceValue, paceUnit]);

  const handleVibeClick = (cluster: Cluster) => {
    const vibe: VibeType = {
      id: `cluster-${cluster.id}`,
      name: cluster.name,
      color: cluster.color,
      tags: cluster.tags,
    };

    setSelectedVibe(vibe);
    setIsTransitioning(true);
    setTimeout(() => {
      onVibeSelect(vibe);
    }, 600);
  };

  // Loading state
  if (isLoading) {
    return (
      <div style={{ minHeight: '100dvh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 'calc(48px + var(--safe-area-top)) 24px calc(48px + var(--safe-area-bottom))' }}>
        <motion.div
          style={{ width: 40, height: 40, border: '2px solid #FF2D55', borderTopColor: 'transparent', borderRadius: '50%' }}
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '14px', color: 'rgba(255,255,255,0.4)', marginTop: 20, fontWeight: 300 }}>
          finding your vibes...
        </p>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div style={{ minHeight: '100dvh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 'calc(48px + var(--safe-area-top)) 24px calc(48px + var(--safe-area-bottom))', position: 'relative' }}>
        <button
          onClick={onBack}
          style={{
            position: 'absolute', top: 'calc(20px + var(--safe-area-top))', left: 20, width: 44, height: 44, borderRadius: '50%',
            background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)',
            color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
          }}
        >
          <ChevronLeft style={{ width: 20, height: 20 }} />
        </button>
        <div style={{ textAlign: 'center', maxWidth: 300 }}>
          <p style={{ fontFamily: 'var(--font-body)', fontSize: '15px', color: 'rgba(255,255,255,0.6)', marginBottom: 20 }}>
            {error}
          </p>
          <button
            onClick={() => window.location.reload()}
            style={{
              padding: '10px 24px', borderRadius: '10px', background: '#FF2D55', color: '#ffffff',
              fontFamily: 'var(--font-body)', fontSize: '14px', fontWeight: 500, border: 'none', cursor: 'pointer',
            }}
          >
            try again
          </button>
        </div>
      </div>
    );
  }

  // Empty state
  if (clusters.length === 0) {
    return (
      <div style={{ minHeight: '100dvh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '15px', color: 'rgba(255,255,255,0.4)' }}>
          no vibes found
        </p>
      </div>
    );
  }

  // Get the right layout for the number of clusters
  const layoutIndex = Math.min(clusters.length, 5) - 1;
  const layout = bubbleLayouts[layoutIndex];

  return (
    <div style={{ minHeight: '100dvh', display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
      {/* Zoom Transition */}
      {isTransitioning && selectedVibe && (
        <motion.div
          style={{ position: 'fixed', inset: 0, zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <motion.div
            style={{ width: 120, height: 120, borderRadius: '50%', background: '#0a0a0a' }}
            initial={{ scale: 1, opacity: 0.8 }}
            animate={{ scale: 20, opacity: 1 }}
            transition={{ duration: 0.6, ease: 'easeInOut' }}
          />
        </motion.div>
      )}

      {/* Back Button */}
      <button
        onClick={onBack}
        style={{
          position: 'absolute', top: 'calc(20px + var(--safe-area-top))', left: 20, width: 44, height: 44, borderRadius: '50%',
          background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)',
          color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', zIndex: 10,
        }}
      >
        <ChevronLeft style={{ width: 20, height: 20 }} />
      </button>

      {/* Header */}
      <div style={{ padding: 'calc(72px + var(--safe-area-top)) 24px 0', zIndex: 2 }}>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '12px', color: '#FF2D55', fontWeight: 400, marginBottom: 8, letterSpacing: '0.05em' }}>
          {paceValue ? `${Math.floor(paceValue)}:${Math.round((paceValue % 1) * 60).toString().padStart(2, '0')} ${paceUnit}` : `${bpm} bpm`}
        </p>
        <h1 style={{ fontFamily: 'var(--font-heading)', fontWeight: 200, fontSize: '28px', color: '#ffffff', letterSpacing: '0.1em', marginBottom: 8 }}>
          choose your vibe
        </h1>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '14px', color: 'rgba(255,255,255,0.4)', fontWeight: 300 }}>
          tap a bubble to start your session
        </p>
      </div>

      {/* Floating Bubbles Area */}
      <div style={{ flex: 1, position: 'relative', minHeight: '380px' }}>
        {clusters.slice(0, 5).map((cluster, index) => {
          const colorSet = vibeColors[index % vibeColors.length];
          const pos = layout[index] || layout[layout.length - 1];
          const size = pos.size;

          return (
            <motion.button
              key={cluster.id}
              onClick={() => handleVibeClick(cluster)}
              style={{
                position: 'absolute',
                left: pos.x,
                top: pos.y,
                width: size,
                height: size,
                marginLeft: -size / 2,
                marginTop: -size / 2,
                borderRadius: '50%',
                background: `radial-gradient(circle at 35% 35%, rgba(${colorSet.accent}, 0.12) 0%, rgba(${colorSet.accent}, 0.04) 50%, rgba(${colorSet.accent}, 0.01) 100%)`,
                border: `1.5px solid ${colorSet.border}`,
                boxShadow: `0 0 40px ${colorSet.glow}, inset 0 0 30px rgba(${colorSet.accent}, 0.03)`,
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6,
                padding: 16,
                overflow: 'hidden',
              }}
              initial={{ scale: 0, opacity: 0 }}
              animate={{
                scale: 1,
                opacity: 1,
                y: [0, -6, 0, 4, 0],
              }}
              transition={{
                scale: { delay: index * 0.12, duration: 0.5, ease: 'backOut' },
                opacity: { delay: index * 0.12, duration: 0.4 },
                y: { delay: index * 0.12 + 0.5, duration: 4 + index * 0.5, repeat: Infinity, ease: 'easeInOut' },
              }}
              whileHover={{
                scale: 1.08,
                boxShadow: `0 0 60px rgba(${colorSet.accent}, 0.2), 0 0 100px rgba(${colorSet.accent}, 0.08), inset 0 0 40px rgba(${colorSet.accent}, 0.06)`,
              }}
              whileTap={{ scale: 0.95 }}
            >
              {/* Cluster name */}
              <span style={{
                fontFamily: 'var(--font-body)', fontSize: size > 130 ? '15px' : '13px',
                fontWeight: 500, color: '#ffffff', textAlign: 'center',
                lineHeight: 1.2,
              }}>
                {cluster.name}
              </span>

              {/* Tags */}
              <span style={{
                fontFamily: 'var(--font-body)', fontSize: '10px', fontWeight: 300,
                color: `rgba(255, 255, 255, 0.35)`, textAlign: 'center',
                lineHeight: 1.3, maxWidth: size - 40,
                overflow: 'hidden', display: '-webkit-box',
                WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
              }}>
                {cluster.tags.slice(0, 3).join(' · ')}
              </span>

              {/* Track count */}
              <span style={{
                fontFamily: 'var(--font-body)', fontSize: '11px', fontWeight: 400,
                color: colorSet.text, opacity: 0.8, marginTop: 2,
              }}>
                {cluster.track_count} tracks
              </span>
            </motion.button>
          );
        })}
      </div>

      {/* Bottom hint */}
      <div style={{ textAlign: 'center', padding: '0 24px calc(24px + var(--safe-area-bottom))', zIndex: 2 }}>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '12px', color: 'rgba(255,255,255,0.2)', fontWeight: 300 }}>
          {clusters.length} moods matched to your pace
        </p>
      </div>

      {/* Float animation keyframes */}
      <style>{`
        @keyframes bubbleFloat {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-8px); }
        }
      `}</style>
    </div>
  );
}
