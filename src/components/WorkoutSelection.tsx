import { motion } from 'motion/react';
import { ChevronLeft, Bike, Footprints, Dumbbell, Waves, Zap, Leaf } from 'lucide-react';

interface WorkoutSelectionProps {
  onWorkoutSelect: (workout: string) => void;
  onBack: () => void;
}

const workouts = [
  { id: 'jogging', name: 'jogging', icon: Footprints, bpm: '120-140' },
  { id: 'cycling', name: 'cycling', icon: Bike, bpm: '110-130' },
  { id: 'strength', name: 'strength', icon: Dumbbell, bpm: '100-120' },
  { id: 'swimming', name: 'swimming', icon: Waves, bpm: '130-150' },
  { id: 'hiit', name: 'HIIT', icon: Zap, bpm: '150-170' },
  { id: 'yoga', name: 'yoga', icon: Leaf, bpm: '80-100' },
];

export function WorkoutSelection({ onWorkoutSelect, onBack }: WorkoutSelectionProps) {
  return (
    <div style={{ minHeight: '100dvh', display: 'flex', flexDirection: 'column', padding: 'calc(24px + var(--safe-area-top)) 24px calc(24px + var(--safe-area-bottom))', position: 'relative' }}>
      {/* Back Button */}
      <button
        onClick={onBack}
        style={{
          position: 'absolute',
          top: 'calc(20px + var(--safe-area-top))',
          left: 20,
          width: 44,
          height: 44,
          borderRadius: '50%',
          background: 'rgba(255, 255, 255, 0.05)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          color: '#ffffff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
        }}
      >
        <ChevronLeft style={{ width: 20, height: 20 }} />
      </button>

      {/* Header */}
      <div style={{ marginTop: 40, marginBottom: 32 }}>
        <h1
          style={{
            fontFamily: 'var(--font-heading)',
            fontWeight: 200,
            fontSize: '28px',
            color: '#ffffff',
            letterSpacing: '0.1em',
            marginBottom: 8,
          }}
        >
          pick your workout
        </h1>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '14px', color: 'rgba(255,255,255,0.4)', fontWeight: 300 }}>
          each workout has its own optimal tempo
        </p>
      </div>

      {/* Workout Grid */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, width: '100%', maxWidth: 340 }}>
          {workouts.map((workout, index) => {
            const Icon = workout.icon;
            return (
              <motion.button
                key={workout.id}
                onClick={() => onWorkoutSelect(workout.id)}
                style={{
                  padding: '24px 16px',
                  borderRadius: '16px',
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 12,
                }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.08, duration: 0.4 }}
                whileHover={{ scale: 1.03, borderColor: 'rgba(255, 45, 85, 0.3)', background: 'rgba(255, 45, 85, 0.05)' }}
                whileTap={{ scale: 0.97 }}
              >
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: '12px',
                    background: 'rgba(255, 45, 85, 0.08)',
                    border: '1px solid rgba(255, 45, 85, 0.15)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Icon style={{ width: 20, height: 20, color: '#FF2D55' }} />
                </div>
                <div style={{ textAlign: 'center' }}>
                  <span
                    style={{
                      fontFamily: 'var(--font-body)',
                      fontSize: '14px',
                      fontWeight: 500,
                      color: '#ffffff',
                      display: 'block',
                      marginBottom: 4,
                    }}
                  >
                    {workout.name}
                  </span>
                  <span
                    style={{
                      fontFamily: 'var(--font-body)',
                      fontSize: '11px',
                      fontWeight: 300,
                      color: 'rgba(255,255,255,0.35)',
                    }}
                  >
                    {workout.bpm} bpm
                  </span>
                </div>
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Bottom hint */}
      <div style={{ textAlign: 'center', paddingBottom: 16 }}>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '12px', color: 'rgba(255,255,255,0.25)', fontWeight: 300 }}>
          tap a workout to match your activity
        </p>
      </div>
    </div>
  );
}
