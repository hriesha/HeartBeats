import { useState, useEffect } from 'react';
import { GlowingArcs } from './components/GlowingArcs';
import { CursorTrail } from './components/CursorTrail';
import { AppleMusicConnect } from './components/AppleMusicConnect';
import { ControlOptions } from './components/ControlOptions';
import { PaceSelection } from './components/PaceSelection';
import { WorkoutSelection } from './components/WorkoutSelection';
import { VibeSelection } from './components/VibeSelection';
import { VibeDetail } from './components/VibeDetail';
import { TrackerConnected } from './components/TrackerConnected';

export type VibeType = {
  id: string;
  name: string;
  color: string;
  tags: string[];
  clusterId?: number;
  meanTempo?: number;
};

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<'loading' | 'connect' | 'controlOptions' | 'bpm' | 'workout' | 'vibe' | 'detail' | 'trackerConnected'>('loading');
  const [selectedBPM, setSelectedBPM] = useState(120);
  const [paceValue, setPaceValue] = useState(10.0);
  const [paceUnit, setPaceUnit] = useState<'min/mile' | 'min/km'>('min/mile');
  const [selectedVibe, setSelectedVibe] = useState<VibeType | null>(null);

  // On mount: Check if MusicKit is already authorized
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Wait for MusicKit to be available
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

        if (!window.MusicKit) {
          setCurrentScreen('connect');
          return;
        }

        const developerToken = import.meta.env.VITE_APPLE_MUSIC_DEVELOPER_TOKEN;
        if (!developerToken) {
          setCurrentScreen('connect');
          return;
        }

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

  const handleConnected = () => {
    setCurrentScreen('controlOptions');
  };

  const handleSkip = () => {
    setCurrentScreen('controlOptions');
  };

  const handleSelectCustomControls = () => {
    setCurrentScreen('bpm');
  };

  const handleSelectWatch = () => {
    setCurrentScreen('trackerConnected');
  };

  const handleTrackerConnected = () => {
    setSelectedBPM(125);
    setCurrentScreen('vibe');
  };

  const handlePaceSubmit = (value: number, unit: 'min/mile' | 'min/km') => {
    setPaceValue(value);
    setPaceUnit(unit);
    setCurrentScreen('vibe');
  };

  const handleChooseWorkout = () => {
    setCurrentScreen('workout');
  };

  const handleWorkoutSelect = (workout: string) => {
    const workoutBPMs: { [key: string]: number } = {
      jogging: 130,
      cycling: 120,
      strength: 110,
      swimming: 140,
      hiit: 160,
      yoga: 90
    };
    setSelectedBPM(workoutBPMs[workout] || 120);
    setCurrentScreen('vibe');
  };

  const handleVibeSelect = (vibe: VibeType) => {
    setSelectedVibe(vibe);
    setCurrentScreen('detail');
  };

  const handleBack = () => {
    if (currentScreen === 'detail') {
      setCurrentScreen('vibe');
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
    <div style={{ minHeight: '100vh', background: '#0a0a0a', position: 'relative', overflow: 'hidden' }}>
      <GlowingArcs />
      <CursorTrail />
      <div style={{ position: 'relative', zIndex: 1, minHeight: '100vh' }}>
        {currentScreen === 'loading' && (
          <div className="w-full h-full flex items-center justify-center" style={{ minHeight: '100vh' }}>
            <div className="text-center">
              <div style={{ width: 40, height: 40, border: '2px solid #FF2D55', borderTopColor: 'transparent', borderRadius: '50%', margin: '0 auto 16px' }} className="animate-spin" />
              <p style={{ fontFamily: 'var(--font-heading)', color: 'rgba(255,255,255,0.6)', fontSize: '14px', letterSpacing: '0.15em', textTransform: 'uppercase' }}>Loading</p>
            </div>
          </div>
        )}
        {currentScreen === 'connect' && <AppleMusicConnect onConnected={handleConnected} onSkip={handleSkip} />}
        {currentScreen === 'controlOptions' && <ControlOptions onSelectCustom={handleSelectCustomControls} onSelectWatch={handleSelectWatch} onBack={handleBack} />}
        {currentScreen === 'bpm' && <PaceSelection onSubmit={handlePaceSubmit} onChooseWorkout={handleChooseWorkout} onBack={handleBack} />}
        {currentScreen === 'workout' && <WorkoutSelection onWorkoutSelect={handleWorkoutSelect} onBack={handleBack} />}
        {currentScreen === 'vibe' && <VibeSelection paceValue={paceValue} paceUnit={paceUnit} bpm={selectedBPM} onVibeSelect={handleVibeSelect} onBack={handleBack} />}
        {currentScreen === 'detail' && selectedVibe && <VibeDetail vibe={selectedVibe} bpm={selectedBPM} onBack={handleBack} />}
        {currentScreen === 'trackerConnected' && <TrackerConnected onComplete={handleTrackerConnected} />}
      </div>
    </div>
  );
}
