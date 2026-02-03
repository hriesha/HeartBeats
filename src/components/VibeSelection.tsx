import { motion } from 'motion/react';
import { ChevronLeft, ChevronDown, ChevronUp, Beaker } from 'lucide-react';
import { VibeType } from '../App';
import { useState, useEffect } from 'react';
import { runClustering, Cluster, getRecsCoverage, runClusteringWithRecs, getTracksFromTrack } from '../utils/api';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';

interface VibeSelectionProps {
  paceValue: number;
  paceUnit: 'min/mile' | 'min/km';
  bpm?: number; // For backward compatibility with workouts
  onVibeSelect: (vibe: VibeType) => void;
  onBack: () => void;
}

export function VibeSelection({ paceValue, paceUnit, bpm, onVibeSelect, onBack }: VibeSelectionProps) {
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [selectedVibe, setSelectedVibe] = useState<VibeType | null>(null);
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [recsOpen, setRecsOpen] = useState(false);
  const [recsLoading, setRecsLoading] = useState(false);
  const [recsResult, setRecsResult] = useState<{
    coverage?: { total_saved: number; in_lookup: number; coverage_pct: number; by_cluster: Record<string, number> };
    clusters?: { name: string; count: number }[];
    fromTrack?: { source: string; tracks: { name?: string; artist_names?: string }[] };
    error?: string;
  } | null>(null);

  useEffect(() => {
    // Fetch clusters using recs model with pace
    const fetchClusters = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Use recs model with pace (converts pace to BPM automatically)
        const result = await runClustering(paceValue, paceUnit, null);
        console.log('Clustering result:', result);
        
        if (result && result.clusters) {
          if (result.clusters.length === 0) {
            setError(`No tracks found for pace ${paceValue} ${paceUnit}. Try a different pace, or make sure you have tracks in your Spotify library that match this tempo.`);
          } else {
            setClusters(result.clusters);
            console.log(`Clusters loaded for pace ${paceValue} ${paceUnit}:`, result.clusters);

            // Also fetch coverage info to show user
            try {
              const coverage = await getRecsCoverage();
              if (coverage?.success) {
                setRecsResult({
                  coverage: {
                    total_saved: coverage.total_saved ?? 0,
                    in_lookup: coverage.in_lookup ?? 0,
                    coverage_pct: coverage.coverage_pct ?? 0,
                    by_cluster: coverage.by_cluster ?? {},
                  },
                });
              }
            } catch (e) {
              console.warn('Could not fetch coverage:', e);
            }
          }
        } else {
          const errorMsg = result?.message || 'Failed to load clusters. Please make sure you have saved tracks in your Spotify library.';
          setError(errorMsg);
        }
      } catch (err: any) {
        console.error('Error fetching clusters:', err);
        const errorMessage = err?.message || String(err);
        if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError') || errorMessage.includes('connection')) {
          setError('Connection failed: API server is not running. Please start it with: python3 api/heartbeats_api.py');
        } else if (errorMessage.includes('Spotify not connected')) {
          setError('Spotify not connected. Please connect Spotify first.');
        } else {
          setError(`Failed to load clusters: ${errorMessage}`);
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchClusters();
  }, [paceValue, paceUnit]); // Re-fetch when pace changes

  const handleVibeClick = (cluster: Cluster) => {
    // Convert Cluster to VibeType
    const vibe: VibeType = {
      id: `cluster-${cluster.id}`,
      name: cluster.name,
      color: cluster.color,
      tags: cluster.tags,
    };

    setSelectedVibe(vibe);
    setIsTransitioning(true);
    // Wait for transition before navigating
    setTimeout(() => {
      onVibeSelect(vibe);
    }, 800);
  };

  const runRecsTest = async () => {
    setRecsLoading(true);
    setRecsResult(null);
    try {
      const coverage = await getRecsCoverage();
      if (!coverage?.success) {
        setRecsResult({ error: coverage?.error ?? 'Recs coverage failed. Connect Spotify first.' });
        return;
      }
      const clustersRes = await runClusteringWithRecs(bpm);
      const sampleId = coverage.sample_in_lookup?.[0];
      let fromTrackRes: { source?: string; tracks?: { name?: string; artist_names?: string }[] } = {};
      if (sampleId) {
        const ft = await getTracksFromTrack(sampleId, undefined, 5);
        if (ft) fromTrackRes = { source: 'recs', tracks: ft.tracks };
      }
      setRecsResult({
        coverage: {
          total_saved: coverage.total_saved ?? 0,
          in_lookup: coverage.in_lookup ?? 0,
          coverage_pct: coverage.coverage_pct ?? 0,
          by_cluster: coverage.by_cluster ?? {},
        },
        clusters: clustersRes?.clusters?.map(c => ({ name: c.name, count: c.track_count })) ?? [],
        fromTrack: fromTrackRes.tracks?.length ? fromTrackRes : undefined,
      });
    } catch (e) {
      setRecsResult({ error: String(e) });
    } finally {
      setRecsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="relative w-full h-full overflow-auto" style={{ fontFamily: 'Poppins, sans-serif' }}>
        <div
          className="absolute inset-0 z-0"
          style={{
            background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
          }}
        />
        <div className="relative z-10 w-full h-full flex flex-col items-center justify-center px-6">
          <div className="flex flex-col items-center gap-4">
            <motion.div
              className="w-16 h-16 border-4 border-#FCBF49 border-t-transparent rounded-full"
              animate={{ rotate: 360 }}
              transition={{
                duration: 1,
                repeat: Infinity,
                ease: "linear"
              }}
            />
            <p style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '16px',
              color: '#EAE2B7',
              opacity: 0.8
            }}>
              analyzing your music library...
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="relative w-full h-full overflow-auto" style={{ fontFamily: 'Poppins, sans-serif' }}>
        <div
          className="absolute inset-0 z-0"
          style={{
            background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
          }}
        />
        <div className="relative z-10 w-full h-full flex flex-col items-center justify-center px-6">
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
          <div className="text-center max-w-sm">
            <p style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '18px',
              color: '#EAE2B7',
              marginBottom: '12px'
            }}>
              {error}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-2 rounded-full"
              style={{
                background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)',
                color: 'white',
                fontFamily: 'Poppins, sans-serif',
                fontSize: '14px',
                fontWeight: 600
              }}
            >
              try again
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (clusters.length === 0) {
    return (
      <div className="relative w-full h-full overflow-auto" style={{ fontFamily: 'Poppins, sans-serif' }}>
        <div
          className="absolute inset-0 z-0"
          style={{
            background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
          }}
        />
        <div className="relative z-10 w-full h-full flex flex-col items-center justify-center px-6">
          <p style={{
            fontFamily: 'Poppins, sans-serif',
            fontSize: '16px',
            color: '#EAE2B7',
            opacity: 0.8
          }}>
            no clusters found
          </p>
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
            {paceValue ? `pace: ${Math.floor(paceValue)}:${Math.round((paceValue % 1) * 60).toString().padStart(2, '0')} ${paceUnit}` : `based on ${bpm} bpm`}
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
          {recsResult?.coverage && (
            <div
              className="mt-3 p-2 rounded-lg text-xs"
              style={{
                fontFamily: 'Poppins, sans-serif',
                backgroundColor: recsResult.coverage.coverage_pct >= 50 ? 'rgba(46, 204, 113, 0.2)' : 'rgba(241, 196, 15, 0.2)',
                color: '#EAE2B7',
                border: `1px solid ${recsResult.coverage.coverage_pct >= 50 ? 'rgba(46, 204, 113, 0.4)' : 'rgba(241, 196, 15, 0.4)'}`,
              }}
            >
              📊 {recsResult.coverage.in_lookup} / {recsResult.coverage.total_saved} tracks in model ({recsResult.coverage.coverage_pct}% coverage)
              {recsResult.coverage.coverage_pct < 50 && (
                <span className="block mt-1 opacity-80">Some tracks may not have recommendations.</span>
              )}
            </div>
          )}
        </div>

        {/* Vibe Bubbles - dynamically positioned */}
        <div className="flex-1 flex items-center justify-center">
          <div className="relative w-full" style={{ height: '400px' }}>
            {clusters.map((cluster, index) => {
              // Position clusters in a 2x2 grid pattern
              const positions = [
                { top: '10%', left: '5%' },
                { top: '15%', right: '8%' },
                { bottom: '20%', left: '12%' },
                { bottom: '15%', right: '5%' }
              ];
              const position = positions[index] || { top: '50%', left: '50%' };
              const animations = [
                { y: [0, -15, 0], x: [0, 10, 0], duration: 4 },
                { y: [0, 20, 0], x: [0, -8, 0], duration: 5, delay: 0.5 },
                { y: [0, -18, 0], x: [0, 12, 0], duration: 4.5, delay: 1 },
                { y: [0, 22, 0], x: [0, -15, 0], duration: 5.5, delay: 1.5 }
              ];
              const animation = animations[index] || { y: [0, 0, 0], x: [0, 0, 0], duration: 4 };
              const sizes = [140, 150, 145, 155];
              const size = sizes[index] || 140;

              return (
                <motion.button
                  key={cluster.id}
                  onClick={() => handleVibeClick(cluster)}
                  className="absolute"
                  style={{
                    ...position,
                    width: `${size}px`,
                    height: `${size}px`,
                    borderRadius: '50%',
                    backgroundColor: cluster.color,
                    boxShadow: `0 8px 24px ${cluster.color}80, inset 0 2px 8px rgba(255, 255, 255, 0.3)`,
                    border: 'none',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '16px'
                  }}
                  animate={animation}
                  transition={{
                    duration: animation.duration || 4,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: animation.delay || 0
                  }}
                  whileHover={{
                    scale: 1.1,
                    boxShadow: `0 12px 32px ${cluster.color}B3, inset 0 2px 8px rgba(255, 255, 255, 0.3)`
                  }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span style={{
                    fontFamily: 'Poppins, sans-serif',
                    fontSize: '16px',
                    fontWeight: 700,
                    color: '#03071E',
                    textAlign: 'center',
                    lineHeight: 1.2
                  }}>
                    {cluster.name}
                  </span>
                  <span style={{
                    fontFamily: 'Poppins, sans-serif',
                    fontSize: '11px',
                    fontWeight: 500,
                    color: '#370617',
                    marginTop: '4px'
                  }}>
                    {cluster.tags.join(' • ')}
                  </span>
                  <span style={{
                    fontFamily: 'Poppins, sans-serif',
                    fontSize: '10px',
                    fontWeight: 400,
                    color: '#6A040F',
                    marginTop: '4px',
                    opacity: 0.8
                  }}>
                    {cluster.track_count} tracks
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

        {/* Test recs model (collapsible) */}
        <Collapsible open={recsOpen} onOpenChange={setRecsOpen} className="mt-4">
          <CollapsibleTrigger
            className="flex items-center justify-center gap-2 w-full py-2 rounded-lg text-sm font-medium transition-colors"
            style={{
              fontFamily: 'Poppins, sans-serif',
              color: '#FCBF49',
              backgroundColor: 'rgba(0, 48, 73, 0.6)',
            }}
          >
            <Beaker className="w-4 h-4" />
            Test recs model
            {recsOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div
              className="mt-2 p-3 rounded-lg text-left space-y-2"
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '12px',
                color: '#EAE2B7',
                backgroundColor: 'rgba(0, 48, 73, 0.5)',
              }}
            >
              <p className="opacity-80">See how well the trained model covers your Spotify library and gets recommendations.</p>
              <button
                type="button"
                onClick={runRecsTest}
                disabled={recsLoading}
                className="px-3 py-1.5 rounded-md text-xs font-semibold disabled:opacity-50"
                style={{ backgroundColor: '#F77F00', color: '#fff' }}
              >
                {recsLoading ? 'Running...' : 'Run full test'}
              </button>
              {recsResult?.error && (
                <p className="text-red-300 text-xs">{recsResult.error}</p>
              )}
              {recsResult?.coverage && (
                <div className="space-y-1 text-xs">
                  <p>Coverage: {recsResult.coverage.in_lookup} / {recsResult.coverage.total_saved} saved tracks ({recsResult.coverage.coverage_pct}%) in recs lookup.</p>
                  {Object.keys(recsResult.coverage.by_cluster).length > 0 && (
                    <p>By cluster: {JSON.stringify(recsResult.coverage.by_cluster)}</p>
                  )}
                </div>
              )}
              {recsResult?.clusters && recsResult.clusters.length > 0 && (
                <p className="text-xs">Recs clusters: {recsResult.clusters.map(c => `${c.name} (${c.count})`).join(', ')}</p>
              )}
              {recsResult?.fromTrack?.tracks && recsResult.fromTrack.tracks.length > 0 && (
                <div className="text-xs">
                  <p className="font-semibold mb-1">Sample recommendations (from one of your tracks):</p>
                  <ul className="list-disc pl-4 space-y-0.5">
                    {recsResult.fromTrack.tracks.slice(0, 5).map((t, i) => (
                      <li key={i}>{t.name ?? t.artist_names ?? '—'} {t.artist_names ? `– ${t.artist_names}` : ''}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    </div>
  );
}
