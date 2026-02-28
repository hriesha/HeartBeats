import { motion } from 'motion/react';
import { Watch, ChevronLeft, Heart } from 'lucide-react';
import { useHeartRate } from '../hooks/useHeartRate';
import { useEffect } from 'react';

interface AppleWatchConnectProps {
  onConnected: (initialBpm: number) => void;
  onBack: () => void;
}

export function AppleWatchConnect({ onConnected, onBack }: AppleWatchConnectProps) {
  const { bpm, status, requestPermission, startPolling } = useHeartRate();

  const handleConnect = async () => {
    const authorized = await requestPermission();
    if (authorized) {
      startPolling();
    }
  };

  // Once we have a BPM reading, allow user to proceed
  const isReady = status === 'polling' && bpm !== null;

  useEffect(() => {
    // If already authorized and polling yields a reading, auto-advance after a brief moment
    if (isReady) {
      const t = setTimeout(() => onConnected(bpm!), 1200);
      return () => clearTimeout(t);
    }
  }, [isReady, bpm, onConnected]);

  return (
    <div style={{
      minHeight: '100dvh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 'calc(48px + var(--safe-area-top)) 24px calc(48px + var(--safe-area-bottom))',
      position: 'relative',
    }}>
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

      {/* Watch Icon with pulse ring */}
      <motion.div style={{ position: 'relative', marginBottom: 40 }}>
        {/* Pulse rings when polling */}
        {(status === 'polling' || status === 'requesting') && (
          <>
            {[0, 1].map((i) => (
              <motion.div
                key={i}
                style={{
                  position: 'absolute',
                  inset: 0,
                  borderRadius: '50%',
                  border: '1px solid rgba(255, 45, 85, 0.4)',
                }}
                animate={{ scale: [1, 2.2], opacity: [0.6, 0] }}
                transition={{ duration: 2, repeat: Infinity, delay: i * 1, ease: 'easeOut' }}
              />
            ))}
          </>
        )}
        <div style={{
          width: 96,
          height: 96,
          borderRadius: '50%',
          background: 'rgba(255, 45, 85, 0.1)',
          border: '1.5px solid rgba(255, 45, 85, 0.3)',
          boxShadow: '0 0 40px rgba(255, 45, 85, 0.15)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <Watch style={{ width: 40, height: 40, color: '#FF2D55' }} />
        </div>
      </motion.div>

      {/* Title */}
      <motion.h1
        style={{
          fontFamily: 'var(--font-heading)',
          fontWeight: 200,
          fontSize: '26px',
          color: '#ffffff',
          letterSpacing: '0.1em',
          textAlign: 'center',
          marginBottom: 12,
        }}
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        connect apple watch
      </motion.h1>

      {/* Subtitle / state description */}
      <motion.p
        key={status}
        style={{
          fontFamily: 'var(--font-body)',
          fontSize: '14px',
          color: 'rgba(255,255,255,0.45)',
          fontWeight: 300,
          textAlign: 'center',
          maxWidth: 280,
          marginBottom: 48,
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4 }}
      >
        {status === 'idle' && 'HeartBeats will read your heart rate to sync music with your intensity.'}
        {status === 'requesting' && 'Allow access to Apple Health in the prompt…'}
        {status === 'polling' && bpm === null && 'Reading your heart rate from Apple Watch…'}
        {status === 'polling' && bpm !== null && 'Heart rate detected — starting your session.'}
        {status === 'error' && 'Could not access Apple Health. Please allow access in Settings.'}
        {status === 'unsupported' && 'Apple Watch is not available on this device.'}
      </motion.p>

      {/* Live BPM display */}
      {status === 'polling' && bpm !== null && (
        <motion.div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            marginBottom: 40,
          }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4 }}
        >
          <Heart style={{ width: 20, height: 20, color: '#FF2D55' }} />
          <span style={{
            fontFamily: 'var(--font-heading)',
            fontSize: '48px',
            fontWeight: 200,
            color: '#ffffff',
            letterSpacing: '0.05em',
          }}>
            {bpm}
          </span>
          <span style={{
            fontFamily: 'var(--font-body)',
            fontSize: '13px',
            color: 'rgba(255,255,255,0.4)',
            fontWeight: 300,
            alignSelf: 'flex-end',
            paddingBottom: 8,
          }}>
            bpm
          </span>
        </motion.div>
      )}

      {/* CTA Button */}
      {(status === 'idle' || status === 'error') && (
        <motion.button
          onClick={handleConnect}
          style={{
            padding: '16px 40px',
            borderRadius: '100px',
            background: '#FF2D55',
            border: 'none',
            color: '#ffffff',
            fontFamily: 'var(--font-body)',
            fontSize: '15px',
            fontWeight: 500,
            letterSpacing: '0.05em',
            cursor: 'pointer',
          }}
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.96 }}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.4 }}
        >
          {status === 'error' ? 'try again' : 'authorise health access'}
        </motion.button>
      )}

      {/* Loading dots while requesting/polling without reading */}
      {(status === 'requesting' || (status === 'polling' && bpm === null)) && (
        <div style={{ display: 'flex', gap: 8 }}>
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              style={{ width: 6, height: 6, borderRadius: '50%', background: '#FF2D55' }}
              animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1.2, 0.8] }}
              transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
            />
          ))}
        </div>
      )}

      {/* Not supported message */}
      {status === 'unsupported' && (
        <motion.button
          onClick={onBack}
          style={{
            padding: '16px 40px',
            borderRadius: '100px',
            background: 'rgba(255,255,255,0.06)',
            border: '1px solid rgba(255,255,255,0.1)',
            color: 'rgba(255,255,255,0.6)',
            fontFamily: 'var(--font-body)',
            fontSize: '15px',
            fontWeight: 400,
            cursor: 'pointer',
          }}
          whileTap={{ scale: 0.96 }}
        >
          go back
        </motion.button>
      )}
    </div>
  );
}
