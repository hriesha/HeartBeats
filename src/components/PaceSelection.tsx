import { useState } from 'react';
import { motion } from 'motion/react';
import { ChevronLeft } from 'lucide-react';

interface PaceSelectionProps {
  onSubmit: (paceValue: number, paceUnit: 'min/mile' | 'min/km') => void;
  onChooseWorkout: () => void;
  onBack: () => void;
}

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

  const minValue = paceUnit === 'min/mile' ? 4.0 : 2.5;
  const maxValue = paceUnit === 'min/mile' ? 15.0 : 9.3;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', padding: '24px', position: 'relative' }}>
      {/* Back Button */}
      <button
        onClick={onBack}
        style={{
          position: 'absolute',
          top: 20,
          left: 20,
          width: 40,
          height: 40,
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
      <div style={{ marginTop: 56, marginBottom: 32 }}>
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
          set your pace
        </h1>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '14px', color: 'rgba(255,255,255,0.4)', fontWeight: 300 }}>
          match your music to your running tempo
        </p>
      </div>

      {/* Unit Toggle */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 32, justifyContent: 'center' }}>
        {(['min/mile', 'min/km'] as const).map((unit) => (
          <button
            key={unit}
            onClick={() => {
              if (unit !== paceUnit) {
                const factor = unit === 'min/km' ? 1 / 1.60934 : 1.60934;
                setPaceValue(paceValue * factor);
                setPaceUnit(unit);
              }
            }}
            style={{
              padding: '8px 20px',
              borderRadius: '20px',
              fontFamily: 'var(--font-body)',
              fontSize: '13px',
              fontWeight: 400,
              cursor: 'pointer',
              transition: 'all 0.2s',
              background: paceUnit === unit ? '#FF2D55' : 'rgba(255, 255, 255, 0.05)',
              color: paceUnit === unit ? '#ffffff' : 'rgba(255,255,255,0.5)',
              border: paceUnit === unit ? '1px solid transparent' : '1px solid rgba(255, 255, 255, 0.1)',
            }}
          >
            {unit}
          </button>
        ))}
      </div>

      {/* Pace Display */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <motion.div
          style={{
            width: 200,
            height: 200,
            borderRadius: '50%',
            border: '1.5px solid rgba(255, 45, 85, 0.25)',
            background: 'rgba(255, 45, 85, 0.03)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: 40,
          }}
          animate={{
            boxShadow: [
              '0 0 30px rgba(255, 45, 85, 0.08)',
              '0 0 50px rgba(255, 45, 85, 0.15)',
              '0 0 30px rgba(255, 45, 85, 0.08)',
            ],
          }}
          transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        >
          <span
            style={{
              fontFamily: 'var(--font-heading)',
              fontSize: '52px',
              fontWeight: 200,
              color: '#ffffff',
              lineHeight: 1,
              letterSpacing: '0.05em',
            }}
          >
            {formatPace(paceValue)}
          </span>
          <span
            style={{
              fontFamily: 'var(--font-body)',
              fontSize: '13px',
              fontWeight: 300,
              color: '#FF2D55',
              marginTop: 8,
            }}
          >
            {paceUnit}
          </span>
        </motion.div>

        {/* Slider */}
        <div style={{ width: '100%', maxWidth: 320, marginBottom: 24 }}>
          <input
            type="range"
            min={minValue}
            max={maxValue}
            step={0.1}
            value={paceValue}
            onChange={(e) => setPaceValue(Number(e.target.value))}
            style={{
              width: '100%',
              height: 2,
              borderRadius: 1,
              appearance: 'none',
              cursor: 'pointer',
              background: `linear-gradient(to right, #FF2D55 0%, #FF2D55 ${((paceValue - minValue) / (maxValue - minValue)) * 100}%, rgba(255, 255, 255, 0.1) ${((paceValue - minValue) / (maxValue - minValue)) * 100}%, rgba(255, 255, 255, 0.1) 100%)`,
              outline: 'none',
            }}
          />
          <style>{`
            input[type="range"]::-webkit-slider-thumb {
              appearance: none;
              width: 20px;
              height: 20px;
              border-radius: 50%;
              background: #FF2D55;
              cursor: pointer;
              box-shadow: 0 0 0 4px rgba(255, 45, 85, 0.15), 0 0 20px rgba(255, 45, 85, 0.3);
            }
            input[type="range"]::-moz-range-thumb {
              width: 20px;
              height: 20px;
              border-radius: 50%;
              background: #FF2D55;
              cursor: pointer;
              border: none;
              box-shadow: 0 0 0 4px rgba(255, 45, 85, 0.15), 0 0 20px rgba(255, 45, 85, 0.3);
            }
          `}</style>
        </div>

        {/* Preset Chips */}
        <div style={{ display: 'flex', gap: 10, marginBottom: 32, flexWrap: 'wrap', justifyContent: 'center' }}>
          {presetPaces.map((preset) => {
            const displayValue = paceUnit === 'min/mile' ? preset.value : preset.value / 1.60934;
            const isActive = Math.abs(paceValue - displayValue) < 0.1;
            return (
              <button
                key={preset.value}
                onClick={() => setPaceValue(displayValue)}
                style={{
                  padding: '8px 20px',
                  borderRadius: '20px',
                  fontFamily: 'var(--font-body)',
                  fontSize: '13px',
                  fontWeight: 400,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  background: isActive ? 'rgba(255, 45, 85, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                  color: isActive ? '#FF2D55' : 'rgba(255,255,255,0.4)',
                  border: isActive ? '1px solid rgba(255, 45, 85, 0.3)' : '1px solid rgba(255, 255, 255, 0.08)',
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
        style={{
          width: '100%',
          padding: '16px',
          borderRadius: '12px',
          background: '#FF2D55',
          color: '#ffffff',
          fontFamily: 'var(--font-body)',
          fontSize: '15px',
          fontWeight: 500,
          border: '1px solid transparent',
          cursor: 'pointer',
          marginBottom: 12,
          letterSpacing: '0.03em',
        }}
        whileHover={{ scale: 1.02, boxShadow: '0 0 30px rgba(255, 45, 85, 0.3)' }}
        whileTap={{ scale: 0.98 }}
      >
        find my vibes
      </motion.button>

      {/* Or divider */}
      <div style={{ textAlign: 'center', marginBottom: 12 }}>
        <span style={{ fontFamily: 'var(--font-body)', color: 'rgba(255,255,255,0.25)', fontSize: '13px', fontWeight: 300 }}>
          or
        </span>
      </div>

      {/* Choose Workout Button */}
      <motion.button
        onClick={onChooseWorkout}
        style={{
          width: '100%',
          padding: '16px',
          borderRadius: '12px',
          background: 'rgba(255, 255, 255, 0.03)',
          color: 'rgba(255,255,255,0.6)',
          fontFamily: 'var(--font-body)',
          fontSize: '15px',
          fontWeight: 400,
          border: '1px solid rgba(255, 255, 255, 0.08)',
          cursor: 'pointer',
          letterSpacing: '0.03em',
        }}
        whileHover={{ scale: 1.02, borderColor: 'rgba(255, 255, 255, 0.15)' }}
        whileTap={{ scale: 0.98 }}
      >
        choose a workout instead
      </motion.button>
    </div>
  );
}
