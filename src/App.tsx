import { useState, useEffect } from 'react';
import { PhoneFrame } from './components/PhoneFrame';
import { SpotifyConnect } from './components/SpotifyConnect';
import { ControlOptions } from './components/ControlOptions';
import { PaceSelection } from './components/PaceSelection';
import { WorkoutSelection } from './components/WorkoutSelection';
import { VibeSelection } from './components/VibeSelection';
import { VibeDetail } from './components/VibeDetail';
import { TrackerConnected } from './components/TrackerConnected';
import { checkSpotifyStatus, SpotifyUser } from './utils/api';

export type VibeType = {
  id: string;
  name: string;
  color: string;
  tags: string[];
  clusterId?: number;
  meanTempo?: number;
};

export default function App() {
  // Screen states: 'loading' -> check auth -> 'spotify' (if not auth) or 'controlOptions' (if auth)
  const [currentScreen, setCurrentScreen] = useState<'loading' | 'spotify' | 'controlOptions' | 'bpm' | 'workout' | 'vibe' | 'detail' | 'trackerConnected'>('loading');
  const [selectedBPM, setSelectedBPM] = useState(120); // Keep for backward compatibility with workouts
  const [paceValue, setPaceValue] = useState(10.0);
  const [paceUnit, setPaceUnit] = useState<'min/mile' | 'min/km'>('min/mile');
  const [selectedVibe, setSelectedVibe] = useState<VibeType | null>(null);
  const [spotifyUser, setSpotifyUser] = useState<SpotifyUser | null>(null);

  // On mount: Check if user is already authenticated with Spotify
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const status = await checkSpotifyStatus();
        if (status.connected && status.user) {
          // User is already logged in - skip to main app
          setSpotifyUser(status.user);
          setCurrentScreen('controlOptions');
        } else {
          // User needs to log in
          setCurrentScreen('spotify');
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        setCurrentScreen('spotify');
      }
    };
    checkAuth();
  }, []);

  // Handle OAuth callback URL parameters (after redirect from Spotify)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const spotifyConnected = params.get('spotify_connected');
    const spotifyError = params.get('spotify_error');

    if (spotifyConnected === 'true') {
      // Clear URL params (clean up the URL)
      window.history.replaceState({}, '', window.location.pathname);

      // Refresh auth status to get user info
      checkSpotifyStatus().then(status => {
        if (status.connected && status.user) {
          setSpotifyUser(status.user);
          setCurrentScreen('controlOptions');
        }
      });
    } else if (spotifyError) {
      // Clear URL params
      window.history.replaceState({}, '', window.location.pathname);
      console.error('Spotify OAuth error:', spotifyError);
      // Stay on spotify screen so user can try again
      setCurrentScreen('spotify');
    }
  }, []);

  const handleSpotifyConnected = () => {
    // This is called after successful OAuth (user is redirected back)
    // The useEffect above handles the actual auth check
    setCurrentScreen('controlOptions');
  };

  const handleSpotifySkip = () => {
    // Allow skipping Spotify login (limited functionality)
    setCurrentScreen('controlOptions');
  };

  const handleSelectCustomControls = () => {
    setCurrentScreen('bpm');
  };

  const handleSelectWatch = () => {
    // Show tracker connected screen
    setCurrentScreen('trackerConnected');
  };

  const handleTrackerConnected = () => {
    // Set a default BPM from watch simulation
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
    // Different workouts have different BPM ranges
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
      // Go back to Spotify login (allows re-auth or logging into different account)
      setCurrentScreen('spotify');
    }
    // No back from 'spotify' - it's the first screen now
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <PhoneFrame>
        {/* Loading state while checking auth */}
        {currentScreen === 'loading' && (
          <div className="w-full h-full flex items-center justify-center" style={{ background: 'linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)' }}>
            <div className="text-center">
              <div className="w-12 h-12 border-4 border-[#FCBF49] border-t-transparent rounded-full animate-spin mx-auto mb-4" />
              <p style={{ fontFamily: 'Poppins, sans-serif', color: '#EAE2B7', fontSize: '16px' }}>Loading...</p>
            </div>
          </div>
        )}
        {currentScreen === 'spotify' && <SpotifyConnect onConnected={handleSpotifyConnected} onSkip={handleSpotifySkip} />}
        {currentScreen === 'controlOptions' && <ControlOptions onSelectCustom={handleSelectCustomControls} onSelectWatch={handleSelectWatch} onBack={handleBack} />}
        {currentScreen === 'bpm' && <PaceSelection onSubmit={handlePaceSubmit} onChooseWorkout={handleChooseWorkout} onBack={handleBack} />}
        {currentScreen === 'workout' && <WorkoutSelection onWorkoutSelect={handleWorkoutSelect} onBack={handleBack} />}
        {currentScreen === 'vibe' && <VibeSelection paceValue={paceValue} paceUnit={paceUnit} bpm={selectedBPM} onVibeSelect={handleVibeSelect} onBack={handleBack} />}
        {currentScreen === 'detail' && selectedVibe && <VibeDetail vibe={selectedVibe} bpm={selectedBPM} onBack={handleBack} />}
        {currentScreen === 'trackerConnected' && <TrackerConnected onComplete={handleTrackerConnected} />}
      </PhoneFrame>
    </div>
  );
}