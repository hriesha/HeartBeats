import { useState } from 'react';
import { PhoneFrame } from './components/PhoneFrame';
import { LoginScreen } from './components/LoginScreen';
import { ControlOptions } from './components/ControlOptions';
import { BPMSelection } from './components/BPMSelection';
import { WorkoutSelection } from './components/WorkoutSelection';
import { VibeSelection } from './components/VibeSelection';
import { VibeDetail } from './components/VibeDetail';
import { TrackerConnected } from './components/TrackerConnected';

export type VibeType = {
  id: string;
  name: string;
  color: string;
  tags: string[];
};

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<'login' | 'controlOptions' | 'bpm' | 'workout' | 'vibe' | 'detail' | 'trackerConnected'>('login');
  const [selectedBPM, setSelectedBPM] = useState(120);
  const [selectedVibe, setSelectedVibe] = useState<VibeType | null>(null);

  const handleLogin = () => {
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

  const handleBPMSubmit = (bpm: number) => {
    setSelectedBPM(bpm);
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
      setCurrentScreen('login');
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <PhoneFrame>
        {currentScreen === 'login' && <LoginScreen onLogin={handleLogin} />}
        {currentScreen === 'controlOptions' && <ControlOptions onSelectCustom={handleSelectCustomControls} onSelectWatch={handleSelectWatch} onBack={handleBack} />}
        {currentScreen === 'bpm' && <BPMSelection onSubmit={handleBPMSubmit} onChooseWorkout={handleChooseWorkout} onBack={handleBack} />}
        {currentScreen === 'workout' && <WorkoutSelection onWorkoutSelect={handleWorkoutSelect} onBack={handleBack} />}
        {currentScreen === 'vibe' && <VibeSelection bpm={selectedBPM} onVibeSelect={handleVibeSelect} onBack={handleBack} />}
        {currentScreen === 'detail' && selectedVibe && <VibeDetail vibe={selectedVibe} onBack={handleBack} />}
        {currentScreen === 'trackerConnected' && <TrackerConnected onComplete={handleTrackerConnected} />}
      </PhoneFrame>
    </div>
  );
}