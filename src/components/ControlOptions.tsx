import { motion } from 'motion/react';
import { Sliders, Watch, ChevronLeft } from 'lucide-react';

interface ControlOptionsProps {
  onSelectCustom: () => void;
  onSelectWatch: () => void;
  onBack?: () => void;
}

export function ControlOptions({ onSelectCustom, onSelectWatch, onBack }: ControlOptionsProps) {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '48px 24px', position: 'relative' }}>
      {/* Back Button */}
      {onBack && (
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
      )}

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: 48 }}>
        <h1
          style={{
            fontFamily: 'var(--font-heading)',
            fontWeight: 200,
            fontSize: '28px',
            color: '#ffffff',
            letterSpacing: '0.1em',
            marginBottom: 12,
          }}
        >
          how do you track?
        </h1>
        <p
          style={{
            fontFamily: 'var(--font-body)',
            fontSize: '14px',
            color: 'rgba(255,255,255,0.4)',
            fontWeight: 300,
          }}
        >
          choose your preferred method
        </p>
      </div>

      {/* Options */}
      <div style={{ width: '100%', maxWidth: '340px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {/* Custom Controls */}
        <motion.button
          onClick={onSelectCustom}
          style={{
            width: '100%',
            padding: '28px 24px',
            borderRadius: '16px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '20px',
            textAlign: 'left',
          }}
          whileHover={{ scale: 1.02, borderColor: 'rgba(255, 45, 85, 0.3)', background: 'rgba(255, 45, 85, 0.05)' }}
          whileTap={{ scale: 0.98 }}
        >
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: '12px',
              background: 'rgba(255, 45, 85, 0.1)',
              border: '1px solid rgba(255, 45, 85, 0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <Sliders style={{ width: 22, height: 22, color: '#FF2D55' }} />
          </div>
          <div>
            <h2 style={{ fontFamily: 'var(--font-body)', fontSize: '16px', fontWeight: 500, color: '#ffffff', marginBottom: 4 }}>
              set your pace
            </h2>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'rgba(255,255,255,0.4)', fontWeight: 300 }}>
              manually set your BPM and preferences
            </p>
          </div>
        </motion.button>

        {/* Connect Watch */}
        <motion.button
          onClick={onSelectWatch}
          style={{
            width: '100%',
            padding: '28px 24px',
            borderRadius: '16px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '20px',
            textAlign: 'left',
          }}
          whileHover={{ scale: 1.02, borderColor: 'rgba(255, 45, 85, 0.3)', background: 'rgba(255, 45, 85, 0.05)' }}
          whileTap={{ scale: 0.98 }}
        >
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: '12px',
              background: 'rgba(255, 45, 85, 0.1)',
              border: '1px solid rgba(255, 45, 85, 0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <Watch style={{ width: 22, height: 22, color: '#FF2D55' }} />
          </div>
          <div>
            <h2 style={{ fontFamily: 'var(--font-body)', fontSize: '16px', fontWeight: 500, color: '#ffffff', marginBottom: 4 }}>
              connect your watch
            </h2>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'rgba(255,255,255,0.4)', fontWeight: 300 }}>
              sync with your fitness tracker
            </p>
          </div>
        </motion.button>
      </div>
    </div>
  );
}
