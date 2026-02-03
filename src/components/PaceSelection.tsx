import { useState } from 'react';
import { motion } from 'motion/react';
import { ChevronLeft } from 'lucide-react';

interface PaceSelectionProps {
  onSubmit: (paceValue: number, paceUnit: 'min/mile' | 'min/km') => void;
  onChooseWorkout: () => void;
  onBack: () => void;
}

// Common pace presets (in min/mile)
const presetPaces = [
  { value: 7.0, label: '7:00' },
  { value: 8.0, label: '8:00' },
  { value: 9.0, label: '9:00' },
  { value: 10.0, label: '10:00' },
];

export function PaceSelection({ onSubmit, onChooseWorkout, onBack }: PaceSelectionProps) {
  const [paceValue, setPaceValue] = useState(10.0);
  const [paceUnit, setPaceUnit] = useState<'min/mile' | 'min/km'>('min/mile');

  const handleSubmit = () => {
    onSubmit(paceValue, paceUnit);
  };

  const formatPace = (value: number): string => {
    const minutes = Math.floor(value);
    const seconds = Math.round((value - minutes) * 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const minValue = paceUnit === 'min/mile' ? 4.0 : 2.5; // ~4 min/mile = ~2.5 min/km
  const maxValue = paceUnit === 'min/mile' ? 15.0 : 9.3; // ~15 min/mile = ~9.3 min/km

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
            Set your pace
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
            choose your running pace to match your music's tempo.
          </p>
        </div>

        {/* Unit Toggle */}
        <div className="flex gap-2 mb-6 justify-center">
          <button
            onClick={() => {
              if (paceUnit === 'min/km') {
                // Convert min/km to min/mile (multiply by 1.60934)
                setPaceValue(paceValue * 1.60934);
                setPaceUnit('min/mile');
              }
            }}
            className={`px-4 py-2 rounded-full transition-all text-sm font-semibold ${
              paceUnit === 'min/mile'
                ? 'bg-[#F77F00] text-white'
                : 'bg-[rgba(0,48,73,0.6)] text-[#EAE2B7] border border-[rgba(252,191,73,0.4)]'
            }`}
            style={{ fontFamily: 'Poppins, sans-serif' }}
          >
            min/mile
          </button>
          <button
            onClick={() => {
              if (paceUnit === 'min/mile') {
                // Convert min/mile to min/km (divide by 1.60934)
                setPaceValue(paceValue / 1.60934);
                setPaceUnit('min/km');
              }
            }}
            className={`px-4 py-2 rounded-full transition-all text-sm font-semibold ${
              paceUnit === 'min/km'
                ? 'bg-[#F77F00] text-white'
                : 'bg-[rgba(0,48,73,0.6)] text-[#EAE2B7] border border-[rgba(252,191,73,0.4)]'
            }`}
            style={{ fontFamily: 'Poppins, sans-serif' }}
          >
            min/km
          </button>
        </div>

        {/* Pace Display Card */}
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
                fontSize: '64px',
                fontWeight: 700,
                color: '#FCBF49',
                lineHeight: 1
              }}
            >
              {formatPace(paceValue)}
            </span>
            <span
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '18px',
                fontWeight: 600,
                color: '#F77F00',
                marginTop: '8px'
              }}
            >
              {paceUnit}
            </span>
          </motion.div>

          {/* Slider */}
          <div className="w-full mb-8">
            <input
              type="range"
              min={minValue}
              max={maxValue}
              step={0.1}
              value={paceValue}
              onChange={(e) => setPaceValue(Number(e.target.value))}
              className="w-full h-2 rounded-full appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #FCBF49 0%, #F77F00 ${((paceValue - minValue) / (maxValue - minValue)) * 100}%, rgba(234, 226, 183, 0.3) ${((paceValue - minValue) / (maxValue - minValue)) * 100}%, rgba(234, 226, 183, 0.3) 100%)`,
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

          {/* Preset Pace Chips */}
          <div className="flex gap-3 mb-8 flex-wrap justify-center">
            {presetPaces.map((preset) => {
              const displayValue = paceUnit === 'min/mile' ? preset.value : preset.value / 1.60934;
              return (
                <button
                  key={preset.value}
                  onClick={() => setPaceValue(displayValue)}
                  className="px-6 py-2 rounded-full transition-all"
                  style={{
                    fontFamily: 'Poppins, sans-serif',
                    backgroundColor: Math.abs(paceValue - displayValue) < 0.1 ? '#F77F00' : 'rgba(0, 48, 73, 0.6)',
                    color: Math.abs(paceValue - displayValue) < 0.1 ? 'white' : '#EAE2B7',
                    fontSize: '14px',
                    fontWeight: 600,
                    border: Math.abs(paceValue - displayValue) < 0.1 ? 'none' : '1px solid rgba(252, 191, 73, 0.4)',
                    boxShadow: Math.abs(paceValue - displayValue) < 0.1 ? '0 4px 16px rgba(247, 127, 0, 0.4)' : 'none'
                  }}
                >
                  {formatPace(displayValue)}
                </button>
              );
            })}
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
