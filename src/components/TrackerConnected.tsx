import { motion } from 'motion/react';
import { Watch, Heart, ChevronLeft } from 'lucide-react';
import { useEffect, useState, useRef } from 'react';
import { runClustering, Cluster } from '../utils/api';
import { VibeType } from '../App';

interface TrackerConnectedProps {
  watchBpm: number | null;
  watchStatus: 'idle' | 'requesting' | 'waiting' | 'reading' | 'denied';
  onComplete: (bpm: number, vibe: VibeType) => void;
  onBack: () => void;
}

const bubbleLayouts: { x: string; y: string; size: number }[][] = [
  [{ x: '50%', y: '50%', size: 140 }],
  [{ x: '38%', y: '44%', size: 128 }, { x: '62%', y: '56%', size: 118 }],
  [{ x: '50%', y: '33%', size: 118 }, { x: '32%', y: '62%', size: 108 }, { x: '68%', y: '64%', size: 104 }],
  [{ x: '38%', y: '34%', size: 112 }, { x: '62%', y: '32%', size: 104 }, { x: '34%', y: '64%', size: 108 }, { x: '65%', y: '66%', size: 100 }],
  [{ x: '50%', y: '28%', size: 100 }, { x: '30%', y: '50%', size: 94 }, { x: '70%', y: '48%', size: 90 }, { x: '36%', y: '72%', size: 94 }, { x: '65%', y: '74%', size: 88 }],
];

const vibeColors = [
  { border: 'rgba(255, 45, 85, 0.25)', glow: 'rgba(255, 45, 85, 0.12)', text: '#FF6B8A', accent: '255, 45, 85' },
  { border: 'rgba(100, 210, 255, 0.2)', glow: 'rgba(100, 210, 255, 0.1)', text: '#64D2FF', accent: '100, 210, 255' },
  { border: 'rgba(48, 209, 88, 0.2)', glow: 'rgba(48, 209, 88, 0.1)', text: '#30D158', accent: '48, 209, 88' },
  { border: 'rgba(255, 214, 10, 0.2)', glow: 'rgba(255, 214, 10, 0.1)', text: '#FFD60A', accent: '255, 214, 10' },
  { border: 'rgba(191, 90, 242, 0.2)', glow: 'rgba(191, 90, 242, 0.1)', text: '#BF5AF2', accent: '191, 90, 242' },
];

export function TrackerConnected({ watchBpm, watchStatus, onComplete, onBack }: TrackerConnectedProps) {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [clustersLoading, setClustersLoading] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const bpmRef = useRef<number | null>(null);
  const clustersFetched = useRef(false);

  useEffect(() => { bpmRef.current = watchBpm; }, [watchBpm]);

  // Fetch clusters once when first reading arrives
  useEffect(() => {
    if (watchStatus !== 'reading' || watchBpm === null || clustersFetched.current) return;
    clustersFetched.current = true;
    setClustersLoading(true);
    const musicBpm = watchBpm < 120 ? Math.min(watchBpm * 2, 200) : watchBpm;
    runClustering(0, 'min/mile', musicBpm).then(result => {
      if (result?.clusters?.length) setClusters(result.clusters);
      setClustersLoading(false);
    }).catch(() => setClustersLoading(false));
  }, [watchStatus, watchBpm]);

  const handleVibeClick = (cluster: Cluster) => {
    const vibe: VibeType = {
      id: `cluster-${cluster.id}`,
      name: cluster.name,
      color: cluster.color,
      tags: cluster.tags,
      topArtists: cluster.top_artists || [],
    };
    setIsTransitioning(true);
    setTimeout(() => {
      const rawBpm = bpmRef.current ?? watchBpm ?? 120;
      const musicBpm = rawBpm < 120 ? Math.min(rawBpm * 2, 200) : rawBpm;
      onComplete(musicBpm, vibe);
    }, 600);
  };

  const layoutIndex = Math.min(clusters.length, 5) - 1;
  const layout = clusters.length > 0 ? (bubbleLayouts[layoutIndex] || bubbleLayouts[bubbleLayouts.length - 1]) : [];

  return (
    <div style={{ minHeight: '100dvh', display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
      {/* Zoom transition */}
      {isTransitioning && (
        <motion.div
          style={{ position: 'fixed', inset: 0, zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        >
          <motion.div
            style={{ width: 120, height: 120, borderRadius: '50%', background: '#0a0a0a' }}
            initial={{ scale: 1, opacity: 0.8 }}
            animate={{ scale: 20, opacity: 1 }}
            transition={{ duration: 0.6, ease: 'easeInOut' }}
          />
        </motion.div>
      )}

      {/* Back button */}
      <button
        onClick={onBack}
        style={{
          position: 'absolute', top: 'calc(20px + var(--safe-area-top))', left: 20,
          width: 44, height: 44, borderRadius: '50%', zIndex: 10,
          background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)',
          color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
        }}
      >
        <ChevronLeft style={{ width: 20, height: 20 }} />
      </button>

      {/* Pre-reading: centered */}
      {watchStatus !== 'reading' && (
        <div style={{
          flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          padding: 'calc(72px + var(--safe-area-top)) 24px calc(48px + var(--safe-area-bottom))',
        }}>
          <motion.div
            style={{ marginBottom: 40 }}
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          >
            <div style={{
              width: 88, height: 88, borderRadius: '50%',
              background: 'rgba(255, 45, 85, 0.1)', border: '1.5px solid rgba(255, 45, 85, 0.3)',
              boxShadow: '0 0 40px rgba(255, 45, 85, 0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Watch style={{ width: 36, height: 36, color: '#FF2D55' }} />
            </div>
          </motion.div>

          {(watchStatus === 'idle' || watchStatus === 'requesting') && (
            <p style={{ fontFamily: 'var(--font-heading)', fontSize: '18px', color: 'rgba(255,255,255,0.6)', letterSpacing: '0.1em' }}>
              requesting access...
            </p>
          )}
          {watchStatus === 'waiting' && (
            <>
              <p style={{ fontFamily: 'var(--font-heading)', fontSize: '18px', color: 'rgba(255,255,255,0.6)', letterSpacing: '0.1em', marginBottom: 8 }}>
                waiting for reading...
              </p>
              <p style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'rgba(255,255,255,0.3)', fontWeight: 300 }}>
                make sure your Apple Watch is on your wrist
              </p>
              <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 24 }}>
                {[0, 1, 2].map(i => (
                  <motion.div key={i}
                    style={{ width: 6, height: 6, borderRadius: '50%', background: '#FF2D55' }}
                    animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1.2, 0.8] }}
                    transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                  />
                ))}
              </div>
            </>
          )}
          {watchStatus === 'denied' && (
            <div style={{ textAlign: 'center', maxWidth: 280 }}>
              <p style={{ fontFamily: 'var(--font-heading)', fontSize: '18px', color: 'rgba(255,255,255,0.6)', letterSpacing: '0.08em', marginBottom: 8 }}>
                access denied
              </p>
              <p style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'rgba(255,255,255,0.3)', fontWeight: 300, lineHeight: 1.5 }}>
                enable Health access in Settings → Privacy → Health → HeartBeats
              </p>
            </div>
          )}
        </div>
      )}

      {/* Reading: BPM header + vibe bubbles */}
      {watchStatus === 'reading' && watchBpm !== null && (
        <>
          {/* BPM header — always visible at top */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            style={{ paddingTop: 'calc(72px + var(--safe-area-top))', paddingLeft: 24, paddingRight: 24, paddingBottom: 12, zIndex: 2 }}
          >
            <div style={{
              borderRadius: '16px', padding: '14px 20px',
              background: 'rgba(255, 45, 85, 0.06)', border: '1px solid rgba(255, 45, 85, 0.18)',
              display: 'flex', alignItems: 'center', gap: 14,
            }}>
              <motion.div
                animate={{ scale: [1, 1.3, 1] }}
                transition={{ duration: 0.6, repeat: Infinity, ease: 'easeInOut' }}
              >
                <Heart style={{ width: 18, height: 18, color: '#FF2D55', fill: '#FF2D55' }} />
              </motion.div>
              <div>
                <p style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'rgba(255,255,255,0.3)', fontWeight: 300, marginBottom: 2 }}>
                  live from apple watch
                </p>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                  <motion.span
                    key={watchBpm}
                    initial={{ scale: 1.15, color: '#FF2D55' }}
                    animate={{ scale: 1, color: '#ffffff' }}
                    transition={{ duration: 0.25 }}
                    style={{ fontFamily: 'var(--font-heading)', fontSize: '34px', fontWeight: 200, lineHeight: 1 }}
                  >
                    {watchBpm}
                  </motion.span>
                  <span style={{ fontFamily: 'var(--font-body)', fontSize: '12px', color: 'rgba(255,255,255,0.35)', fontWeight: 300 }}>bpm</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Vibe section */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
            <div style={{ paddingLeft: 24, paddingRight: 24, paddingBottom: 4, zIndex: 2 }}>
              <h1 style={{ fontFamily: 'var(--font-heading)', fontWeight: 200, fontSize: '22px', color: '#ffffff', letterSpacing: '0.1em' }}>
                choose your vibe
              </h1>
            </div>

            {clustersLoading && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
                <motion.div
                  style={{ width: 32, height: 32, border: '2px solid #FF2D55', borderTopColor: 'transparent', borderRadius: '50%' }}
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                />
                <p style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'rgba(255,255,255,0.35)', marginTop: 16, fontWeight: 300 }}>
                  finding your vibes...
                </p>
              </div>
            )}

            {!clustersLoading && clusters.length === 0 && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, padding: '0 24px' }}>
                <p style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'rgba(255,255,255,0.3)', textAlign: 'center', fontWeight: 300 }}>
                  couldn't load vibes — make sure the server is running
                </p>
              </div>
            )}

            {!clustersLoading && clusters.length > 0 && (
              <div style={{ flex: 1, position: 'relative', minHeight: 280 }}>
                {clusters.slice(0, 5).map((cluster, index) => {
                  const colorSet = vibeColors[index % vibeColors.length];
                  const pos = layout[index] || layout[layout.length - 1];
                  const size = pos.size;
                  return (
                    <motion.button
                      key={cluster.id}
                      onClick={() => handleVibeClick(cluster)}
                      style={{
                        position: 'absolute', left: pos.x, top: pos.y,
                        width: size, height: size, marginLeft: -size / 2, marginTop: -size / 2,
                        borderRadius: '50%',
                        background: `radial-gradient(circle at 35% 35%, rgba(${colorSet.accent}, 0.12) 0%, rgba(${colorSet.accent}, 0.04) 50%, rgba(${colorSet.accent}, 0.01) 100%)`,
                        border: `1.5px solid ${colorSet.border}`,
                        boxShadow: `0 0 40px ${colorSet.glow}, inset 0 0 30px rgba(${colorSet.accent}, 0.03)`,
                        cursor: 'pointer', display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center', gap: 4, padding: 12,
                      }}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1, y: [0, -5, 0, 4, 0] }}
                      transition={{
                        scale: { delay: index * 0.1, duration: 0.4, ease: 'backOut' },
                        opacity: { delay: index * 0.1, duration: 0.3 },
                        y: { delay: index * 0.1 + 0.4, duration: 4 + index * 0.5, repeat: Infinity, ease: 'easeInOut' },
                      }}
                      whileHover={{ scale: 1.08, boxShadow: `0 0 60px rgba(${colorSet.accent}, 0.2), 0 0 100px rgba(${colorSet.accent}, 0.08)` }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <span style={{ fontFamily: 'var(--font-body)', fontSize: size > 110 ? '14px' : '12px', fontWeight: 500, color: '#ffffff', textAlign: 'center', lineHeight: 1.2 }}>
                        {cluster.name}
                      </span>
                      <span style={{ fontFamily: 'var(--font-body)', fontSize: '9px', fontWeight: 300, color: 'rgba(255,255,255,0.35)', textAlign: 'center', lineHeight: 1.3, maxWidth: size - 32, overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
                        {cluster.tags.slice(0, 3).join(' · ')}
                      </span>
                      <span style={{ fontFamily: 'var(--font-body)', fontSize: '10px', fontWeight: 400, color: colorSet.text, opacity: 0.8 }}>
                        {cluster.track_count} tracks
                      </span>
                    </motion.button>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
