import { useState, useEffect, useRef } from 'react';
import { Capacitor } from '@capacitor/core';
import { StatusBar, Style } from '@capacitor/status-bar';
import { GlowingArcs } from './components/GlowingArcs';
import { CursorTrail } from './components/CursorTrail';
import { AppleMusicConnect } from './components/AppleMusicConnect';
import { ControlOptions } from './components/ControlOptions';
import { PaceSelection } from './components/PaceSelection';
import { WorkoutSelection } from './components/WorkoutSelection';
import { VibeSelection } from './components/VibeSelection';
import { VibeDetail } from './components/VibeDetail';
import { TrackerConnected } from './components/TrackerConnected';
import { ArtistSelection } from './components/ArtistSelection';
import HealthKit from './plugins/HealthKit';

export type VibeType = {
  id: string;
  name: string;
  color: string;
  tags: string[];
  clusterId?: number;
  meanTempo?: number;
  topArtists?: string[];
};

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<'loading' | 'connect' | 'controlOptions' | 'bpm' | 'workout' | 'vibe' | 'artistSelection' | 'detail' | 'trackerConnected'>('loading');
  const [selectedBPM, setSelectedBPM] = useState(120);
  const [paceValue, setPaceValue] = useState(10.0);
  const [paceUnit, setPaceUnit] = useState<'min/mile' | 'min/km'>('min/mile');
  const [selectedVibe, setSelectedVibe] = useState<VibeType | null>(null);
  const [selectedArtists, setSelectedArtists] = useState<string[]>([]);

  // Watch / HealthKit state — lives here so BPM persists across screens
  const [watchBpm, setWatchBpm] = useState<number | null>(null);
  const [watchStatus, setWatchStatus] = useState<'idle' | 'requesting' | 'waiting' | 'reading' | 'denied'>('idle');
  const [isWatchMode, setIsWatchMode] = useState(false);
  const watchListenerRef = useRef<{ remove: () => void } | null>(null);

  // Configure native status bar
  useEffect(() => {
    if (Capacitor.isNativePlatform()) {
      StatusBar.setStyle({ style: Style.Dark });
      StatusBar.setBackgroundColor({ color: '#0a0a0a' });
    }
  }, []);

  // On mount: Check if MusicKit is already authorized
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const waitForMusicKit = (): Promise<void> => {
          return new Promise((resolve) => {
            if (window.MusicKit) { resolve(); return; }
            const check = setInterval(() => {
              if (window.MusicKit) { clearInterval(check); resolve(); }
            }, 100);
            setTimeout(() => { clearInterval(check); resolve(); }, 5000);
          });
        };

        await waitForMusicKit();

        if (!window.MusicKit) { setCurrentScreen('connect'); return; }

        const developerToken = import.meta.env.VITE_APPLE_MUSIC_DEVELOPER_TOKEN;
        if (!developerToken) { setCurrentScreen('connect'); return; }

        const music = await window.MusicKit.configure({
          developerToken,
          app: { name: 'HeartBeats', build: '1.0.0' },
        });

        if (music.isAuthorized) {
          setCurrentScreen('controlOptions');
        } else {
          setCurrentScreen('connect');
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        setCurrentScreen('connect');
      }
    };
    checkAuth();
  }, []);

  const startWatchMonitoring = async () => {
    if (!Capacitor.isNativePlatform()) {
      setWatchStatus('reading');
      setWatchBpm(142);
      return;
    }
    setWatchStatus('requesting');
    try {
      const { granted } = await HealthKit.requestPermission();
      if (!granted) { setWatchStatus('denied'); return; }
      setWatchStatus('waiting');
      await HealthKit.startHeartRateMonitoring();
      const handle = await HealthKit.addListener('heartRateUpdate', (data) => {
        setWatchBpm(data.bpm);
        setWatchStatus('reading');
      });
      watchListenerRef.current = handle;
    } catch (err) {
      console.error('HealthKit error:', err);
      setWatchStatus('denied');
    }
  };

  const stopWatchMonitoring = () => {
    watchListenerRef.current?.remove();
    watchListenerRef.current = null;
    HealthKit.stopHeartRateMonitoring().catch(() => {});
    HealthKit.removeAllListeners().catch(() => {});
    setWatchBpm(null);
    setWatchStatus('idle');
    setIsWatchMode(false);
  };

  const handleConnected = () => setCurrentScreen('controlOptions');
  const handleSkip = () => setCurrentScreen('controlOptions');

  const handleDisconnect = async () => {
    try {
      const music = window.MusicKit?.getInstance?.();
      if (music) await music.unauthorize();
    } catch (err) {
      console.warn('Disconnect error:', err);
    }
    setCurrentScreen('connect');
  };

  const handleSelectCustomControls = () => {
    setIsWatchMode(false);
    setCurrentScreen('bpm');
  };

  const handleSelectWatch = () => {
    setIsWatchMode(true);
    setCurrentScreen('trackerConnected');
    startWatchMonitoring();
  };

  const handleTrackerConnected = (bpm: number, vibe: VibeType) => {
    setSelectedBPM(bpm);
    setSelectedVibe(vibe);
    setCurrentScreen('artistSelection');
  };

  const handlePaceSubmit = (value: number, unit: 'min/mile' | 'min/km') => {
    setPaceValue(value);
    setPaceUnit(unit);
    // Mirror the backend BPM formula so VibeDetail fetches at the right BPM
    const KM_PER_MILE = 1.609344;
    let speedMph = 60 / value;
    if (unit === 'min/km') speedMph /= KM_PER_MILE;
    const bpm = Math.max(140, Math.min(200, Math.round(125 + 5.5 * speedMph)));
    setSelectedBPM(bpm);
    setCurrentScreen('vibe');
  };

  const handleChooseWorkout = () => setCurrentScreen('workout');

  const handleWorkoutSelect = (workout: string) => {
    const workoutBPMs: { [key: string]: number } = {
      jogging: 130, cycling: 120, strength: 110,
      swimming: 140, hiit: 160, yoga: 90,
    };
    setSelectedBPM(workoutBPMs[workout] || 120);
    setCurrentScreen('vibe');
  };

  const handleVibeSelect = (vibe: VibeType) => {
    setSelectedVibe(vibe);
    setCurrentScreen('artistSelection');
  };

  const handleArtistsSelected = (artists: string[]) => {
    setSelectedArtists(artists);
    setCurrentScreen('detail');
  };

  const handleBack = () => {
    if (currentScreen === 'detail') {
      setCurrentScreen('artistSelection');
    } else if (currentScreen === 'artistSelection') {
      if (isWatchMode) {
        setCurrentScreen('trackerConnected');
      } else {
        setCurrentScreen('vibe');
      }
    } else if (currentScreen === 'trackerConnected') {
      stopWatchMonitoring();
      setCurrentScreen('controlOptions');
    } else if (currentScreen === 'vibe') {
      setCurrentScreen('bpm');
    } else if (currentScreen === 'workout') {
      setCurrentScreen('bpm');
    } else if (currentScreen === 'bpm') {
      setCurrentScreen('controlOptions');
    } else if (currentScreen === 'controlOptions') {
      setCurrentScreen('connect');
    }
  };

  return (
    <div style={{ minHeight: '100dvh', background: '#0a0a0a', position: 'relative', overflow: 'hidden' }}>
      <GlowingArcs />
      {!Capacitor.isNativePlatform() && <CursorTrail />}
      <div style={{ position: 'relative', zIndex: 1, minHeight: '100dvh' }}>
        {currentScreen === 'loading' && (
          <div className="w-full h-full flex items-center justify-center" style={{ minHeight: '100vh' }}>
            <div className="text-center">
              <div style={{ width: 40, height: 40, border: '2px solid #FF2D55', borderTopColor: 'transparent', borderRadius: '50%', margin: '0 auto 16px' }} className="animate-spin" />
              <p style={{ fontFamily: 'var(--font-heading)', color: 'rgba(255,255,255,0.6)', fontSize: '14px', letterSpacing: '0.15em', textTransform: 'uppercase' }}>Loading</p>
            </div>
          </div>
        )}
        {currentScreen === 'connect' && <AppleMusicConnect onConnected={handleConnected} onSkip={handleSkip} onDisconnect={handleDisconnect} />}
        {currentScreen === 'controlOptions' && <ControlOptions onSelectCustom={handleSelectCustomControls} onSelectWatch={handleSelectWatch} onBack={handleBack} onDisconnect={handleDisconnect} />}
        {currentScreen === 'bpm' && <PaceSelection onSubmit={handlePaceSubmit} onChooseWorkout={handleChooseWorkout} onBack={handleBack} />}
        {currentScreen === 'workout' && <WorkoutSelection onWorkoutSelect={handleWorkoutSelect} onBack={handleBack} />}
        {currentScreen === 'vibe' && <VibeSelection paceValue={paceValue} paceUnit={paceUnit} bpm={selectedBPM} onVibeSelect={handleVibeSelect} onBack={handleBack} />}
        {currentScreen === 'artistSelection' && selectedVibe && (
          <ArtistSelection
            vibe={selectedVibe}
            bpm={selectedBPM}
            topArtists={selectedVibe.topArtists || []}
            onComplete={handleArtistsSelected}
            onBack={handleBack}
          />
        )}
        {currentScreen === 'detail' && selectedVibe && <VibeDetail vibe={selectedVibe} bpm={selectedBPM} watchBpm={isWatchMode ? watchBpm : null} artistNames={selectedArtists} onBack={handleBack} />}
        {currentScreen === 'trackerConnected' && <TrackerConnected watchBpm={watchBpm} watchStatus={watchStatus} onComplete={handleTrackerConnected} onBack={handleBack} />}
      </div>
    </div>
  );
}
