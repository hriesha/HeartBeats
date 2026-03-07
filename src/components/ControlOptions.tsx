import { motion, AnimatePresence } from 'motion/react';
import { Sliders, Watch, ChevronLeft, MessageCircle, Settings2, X, LogOut } from 'lucide-react';
import { useState } from 'react';
import { FeedbackModal } from './FeedbackModal';

interface ControlOptionsProps {
  onSelectCustom: () => void;
  onSelectWatch: () => void;
  onBack?: () => void;
  onDisconnect: () => void;
}

export function ControlOptions({ onSelectCustom, onSelectWatch, onBack, onDisconnect }: ControlOptionsProps) {
  const [showFeedback, setShowFeedback] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  return (
    <>
    <FeedbackModal isOpen={showFeedback} onClose={() => setShowFeedback(false)} />

    {/* Settings Modal */}
    <AnimatePresence>
      {showSettings && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          style={{
            position: 'fixed', inset: 0, zIndex: 100,
            background: 'rgba(0,0,0,0.7)',
            display: 'flex', alignItems: 'flex-end', justifyContent: 'center',
          }}
          onClick={() => setShowSettings(false)}
        >
          <motion.div
            initial={{ y: 80, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 80, opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            onClick={(e) => e.stopPropagation()}
            style={{
              width: '100%', maxWidth: 480,
              background: '#141414', borderRadius: '24px 24px 0 0',
              padding: 'calc(24px + var(--safe-area-bottom)) 24px',
              border: '1px solid rgba(255,255,255,0.08)', borderBottom: 'none',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
              <h2 style={{ fontFamily: 'var(--font-heading)', fontWeight: 200, fontSize: '20px', color: '#ffffff', letterSpacing: '0.08em' }}>
                settings
              </h2>
              <button
                onClick={() => setShowSettings(false)}
                style={{ width: 36, height: 36, borderRadius: '50%', background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
              >
                <X style={{ width: 16, height: 16 }} />
              </button>
            </div>

            {/* Disconnect Apple Music */}
            <motion.button
              onClick={() => { setShowSettings(false); onDisconnect(); }}
              style={{
                width: '100%', padding: '16px 20px', borderRadius: '14px', marginBottom: 12,
                background: 'rgba(255, 45, 85, 0.06)', border: '1px solid rgba(255, 45, 85, 0.2)',
                display: 'flex', alignItems: 'center', gap: 14, cursor: 'pointer', textAlign: 'left',
              }}
              whileHover={{ background: 'rgba(255, 45, 85, 0.1)' }}
              whileTap={{ scale: 0.98 }}
            >
              <LogOut style={{ width: 18, height: 18, color: '#FF2D55', flexShrink: 0 }} />
              <div>
                <p style={{ fontFamily: 'var(--font-body)', fontSize: '14px', fontWeight: 500, color: '#ffffff', marginBottom: 2 }}>
                  disconnect apple music
                </p>
                <p style={{ fontFamily: 'var(--font-body)', fontSize: '12px', fontWeight: 300, color: 'rgba(255,255,255,0.4)' }}>
                  sign out and re-authorize
                </p>
              </div>
            </motion.button>

            {/* Health Permissions */}
            <div
              style={{
                width: '100%', padding: '16px 20px', borderRadius: '14px',
                background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255,255,255,0.08)',
                display: 'flex', alignItems: 'flex-start', gap: 14,
              }}
            >
              <Settings2 style={{ width: 18, height: 18, color: 'rgba(255,255,255,0.4)', flexShrink: 0, marginTop: 2 }} />
              <div>
                <p style={{ fontFamily: 'var(--font-body)', fontSize: '14px', fontWeight: 500, color: '#ffffff', marginBottom: 4 }}>
                  reset health access
                </p>
                <p style={{ fontFamily: 'var(--font-body)', fontSize: '12px', fontWeight: 300, color: 'rgba(255,255,255,0.4)', lineHeight: 1.5 }}>
                  Settings → Privacy & Security → Health → HeartBeats
                </p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>

    <div style={{ minHeight: '100dvh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 'calc(48px + var(--safe-area-top)) 24px calc(48px + var(--safe-area-bottom))', position: 'relative' }}>
      {/* Back Button */}
      {onBack && (
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
      )}

      {/* Settings Button */}
      <button
        onClick={() => setShowSettings(true)}
        style={{
          position: 'absolute',
          top: 'calc(20px + var(--safe-area-top))',
          right: 20,
          width: 44,
          height: 44,
          borderRadius: '50%',
          background: 'rgba(255, 255, 255, 0.05)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          color: 'rgba(255,255,255,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
        }}
      >
        <Settings2 style={{ width: 18, height: 18 }} />
      </button>

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
              sync heart rate from Apple Watch
            </p>
          </div>
        </motion.button>
      </div>

      {/* Feedback Button */}
      <motion.button
        onClick={() => setShowFeedback(true)}
        style={{
          position: 'absolute',
          bottom: 'calc(24px + var(--safe-area-bottom))',
          right: 24,
          width: 48,
          height: 48,
          borderRadius: '50%',
          background: 'rgba(255, 45, 85, 0.12)',
          border: '1px solid rgba(255, 45, 85, 0.25)',
          color: '#FF2D55',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
        }}
        whileHover={{ scale: 1.1, boxShadow: '0 0 20px rgba(255, 45, 85, 0.2)' }}
        whileTap={{ scale: 0.9 }}
      >
        <MessageCircle style={{ width: 20, height: 20 }} />
      </motion.button>
    </div>
    </>
  );
}
