import { motion } from 'motion/react';
import { ChevronLeft, Bike, Footprints, Dumbbell, Waves, Zap } from 'lucide-react';

interface WorkoutSelectionProps {
  onWorkoutSelect: (workout: string) => void;
  onBack: () => void;
}

const workouts = [
  { id: 'jogging', name: 'jogging', icon: Footprints, color: '#FCBF49', bpm: '120-140' },
  { id: 'cycling', name: 'cycling', icon: Bike, color: '#F77F00', bpm: '110-130' },
  { id: 'strength', name: 'muscle training', icon: Dumbbell, color: '#D62828', bpm: '100-120' },
  { id: 'swimming', name: 'swimming', icon: Waves, color: '#003049', bpm: '130-150' },
  { id: 'hiit', name: 'HIIT', icon: Zap, color: '#D62828', bpm: '150-170' },
  { id: 'yoga', name: 'yoga', icon: Footprints, color: '#EAE2B7', bpm: '80-100' }
];

export function WorkoutSelection({ onWorkoutSelect, onBack }: WorkoutSelectionProps) {
  return (
    <div className="relative w-full h-full overflow-auto" style={{ fontFamily: 'Poppins, sans-serif' }}>
      {/* Background with gradient overlay */}
      <div 
        className="absolute inset-0 z-0"
        style={{
          background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
        }}
      />
      
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
            pick your workout
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
            each workout has its own optimal BPM range
          </p>
        </div>

        {/* Workout Circles */}
        <div className="flex-1 flex items-center justify-center">
          <div className="grid grid-cols-2 gap-x-8 gap-y-6 w-full max-w-md">
            {workouts.map((workout, index) => {
              const Icon = workout.icon;
              
              return (
                <div 
                  key={workout.id}
                  className="relative flex items-center justify-center"
                  style={{ height: '140px' }}
                >
                  <motion.button
                    onClick={() => onWorkoutSelect(workout.id)}
                    className="absolute"
                    style={{
                      width: '130px',
                      height: '130px',
                      borderRadius: '50%',
                      backgroundColor: workout.color,
                      boxShadow: `0 8px 24px ${workout.color}80, inset 0 2px 8px rgba(255, 255, 255, 0.3)`,
                      border: 'none',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      padding: '12px'
                    }}
                    animate={{
                      y: [0, (index % 2 === 0 ? -10 : 10), 0],
                      x: [0, (index % 3 === 0 ? 6 : -6), 0],
                    }}
                    transition={{
                      duration: 3 + index * 0.5,
                      repeat: Infinity,
                      ease: "easeInOut",
                      delay: index * 0.3
                    }}
                    whileHover={{ 
                      scale: 1.15,
                      boxShadow: `0 12px 32px ${workout.color}CC, inset 0 2px 8px rgba(255, 255, 255, 0.3)`
                    }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Icon className="w-8 h-8 mb-2" style={{ color: '#4A3419' }} />
                    <span style={{ 
                      fontFamily: 'Poppins, sans-serif', 
                      fontSize: '13px', 
                      fontWeight: 700, 
                      color: '#4A3419', 
                      textAlign: 'center', 
                      lineHeight: 1.1,
                      marginBottom: '4px'
                    }}>
                      {workout.name}
                    </span>
                    <span style={{ 
                      fontFamily: 'Poppins, sans-serif', 
                      fontSize: '10px', 
                      fontWeight: 500, 
                      color: '#6B4423'
                    }}>
                      {workout.bpm} bpm
                    </span>
                  </motion.button>
                </div>
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
            tap a workout to match your activity
          </p>
        </div>
      </div>
    </div>
  );
}