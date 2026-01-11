import { useState } from 'react';
import { motion } from 'motion/react';
import { ChevronLeft } from 'lucide-react';

interface BPMSelectionProps {
  onSubmit: (bpm: number) => void;
  onChooseWorkout: () => void;
  onBack: () => void;
}

const presetBPMs = [90, 110, 130, 150];

export function BPMSelection({ onSubmit, onChooseWorkout, onBack }: BPMSelectionProps) {
  const [bpm, setBpm] = useState(120);

  const handleSubmit = () => {
    onSubmit(bpm);
  };

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
        <div className="mt-8 mb-12">
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
            Pick your BPM
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
            choose your target heart rate to match your music's energy.
          </p>
        </div>

        {/* BPM Display Card */}
        <div className="flex-1 flex flex-col items-center justify-center">
          <motion.div
            className="relative mb-12"
            style={{
              width: '240px',
              height: '240px',
              borderRadius: '50%',
              background: 'rgba(0, 48, 73, 0.6)',
              border: '3px solid #FCBF49',
              boxShadow: '0 8px 32px rgba(252, 191, 73, 0.4), inset 0 4px 16px rgba(252, 191, 73, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column'
            }}
            animate={{
              boxShadow: [
                '0 8px 32px rgba(252, 191, 73, 0.4), inset 0 4px 16px rgba(252, 191, 73, 0.1)',
                '0 8px 48px rgba(252, 191, 73, 0.6), inset 0 4px 16px rgba(252, 191, 73, 0.2)',
                '0 8px 32px rgba(252, 191, 73, 0.4), inset 0 4px 16px rgba(252, 191, 73, 0.1)'
              ]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <span 
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '72px',
                fontWeight: 700,
                color: '#FCBF49',
                lineHeight: 1
              }}
            >
              {bpm}
            </span>
            <span 
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '24px',
                fontWeight: 600,
                color: '#F77F00',
                marginTop: '8px'
              }}
            >
              BPM
            </span>
          </motion.div>

          {/* Slider */}
          <div className="w-full mb-8">
            <input
              type="range"
              min="80"
              max="180"
              value={bpm}
              onChange={(e) => setBpm(Number(e.target.value))}
              className="w-full h-2 rounded-full appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #FCBF49 0%, #F77F00 ${((bpm - 80) / 100) * 100}%, rgba(234, 226, 183, 0.3) ${((bpm - 80) / 100) * 100}%, rgba(234, 226, 183, 0.3) 100%)`,
                outline: 'none'
              }}
            />
            <style>{`
              input[type="range"]::-webkit-slider-thumb {
                appearance: none;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                background: linear-gradient(135deg, #FCBF49 0%, #F77F00 100%);
                cursor: pointer;
                box-shadow: 0 0 0 4px rgba(252, 191, 73, 0.3), 0 2px 8px rgba(0, 0, 0, 0.3);
              }
              input[type="range"]::-moz-range-thumb {
                width: 24px;
                height: 24px;
                border-radius: 50%;
                background: linear-gradient(135deg, #FCBF49 0%, #F77F00 100%);
                cursor: pointer;
                border: none;
                box-shadow: 0 0 0 4px rgba(252, 191, 73, 0.3), 0 2px 8px rgba(0, 0, 0, 0.3);
              }
            `}</style>
          </div>

          {/* Preset BPM Chips */}
          <div className="flex gap-3 mb-8">
            {presetBPMs.map((preset) => (
              <button
                key={preset}
                onClick={() => setBpm(preset)}
                className="px-6 py-2 rounded-full transition-all"
                style={{
                  fontFamily: 'Poppins, sans-serif',
                  backgroundColor: bpm === preset ? '#F77F00' : 'rgba(0, 48, 73, 0.6)',
                  color: bpm === preset ? 'white' : '#EAE2B7',
                  fontSize: '14px',
                  fontWeight: 600,
                  border: bpm === preset ? 'none' : '1px solid rgba(252, 191, 73, 0.4)',
                  boxShadow: bpm === preset ? '0 4px 16px rgba(247, 127, 0, 0.4)' : 'none'
                }}
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        {/* Submit Button */}
        <motion.button
          onClick={handleSubmit}
          className="w-full py-4 rounded-xl transition-all flex items-center justify-center mb-3"
          style={{
            background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)',
            color: 'white',
            fontFamily: 'Poppins, sans-serif',
            fontSize: '18px',
            fontWeight: 700,
            boxShadow: '0 4px 16px rgba(247, 127, 0, 0.4)',
            border: 'none',
            cursor: 'pointer'
          }}
          whileHover={{ 
            scale: 1.02,
            boxShadow: '0 6px 24px rgba(247, 127, 0, 0.6)'
          }}
          whileTap={{ scale: 0.98 }}
        >
          find my vibes
        </motion.button>

        {/* Or Text */}
        <div className="text-center mb-3">
          <span style={{ 
            fontFamily: 'Poppins, sans-serif', 
            color: '#EAE2B7', 
            fontSize: '14px',
            fontWeight: 400,
            opacity: 0.8
          }}>
            or
          </span>
        </div>

        {/* Choose Workout Button */}
        <motion.button
          onClick={onChooseWorkout}
          className="w-full py-4 rounded-xl transition-all flex items-center justify-center"
          style={{
            background: 'rgba(0, 48, 73, 0.6)',
            color: '#EAE2B7',
            fontFamily: 'Poppins, sans-serif',
            fontSize: '18px',
            fontWeight: 700,
            border: '1px solid rgba(252, 191, 73, 0.4)',
            cursor: 'pointer'
          }}
          whileHover={{ 
            scale: 1.02,
            backgroundColor: 'rgba(0, 48, 73, 0.8)'
          }}
          whileTap={{ scale: 0.98 }}
        >
          choose your workout
        </motion.button>
      </div>
    </div>
  );
}