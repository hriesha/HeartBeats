import { motion } from 'motion/react';
import { Watch, Check } from 'lucide-react';
import { useEffect } from 'react';

interface TrackerConnectedProps {
  onComplete: () => void;
}

export function TrackerConnected({ onComplete }: TrackerConnectedProps) {
  useEffect(() => {
    const timer = setTimeout(() => {
      onComplete();
    }, 1500);
    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '48px 24px' }}>
      {/* Success Icon */}
      <motion.div
        style={{ position: 'relative', marginBottom: 32 }}
        initial={{ scale: 0, rotate: -180 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      >
        <div
          style={{
            width: 88,
            height: 88,
            borderRadius: '50%',
            background: 'rgba(255, 45, 85, 0.1)',
            border: '1.5px solid rgba(255, 45, 85, 0.3)',
            boxShadow: '0 0 40px rgba(255, 45, 85, 0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Watch style={{ width: 36, height: 36, color: '#FF2D55' }} />
        </div>

        {/* Check mark */}
        <motion.div
          style={{
            position: 'absolute',
            bottom: -4,
            right: -4,
            width: 28,
            height: 28,
            borderRadius: '50%',
            background: '#FF2D55',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.4, duration: 0.3 }}
        >
          <Check style={{ width: 14, height: 14, color: '#ffffff' }} strokeWidth={3} />
        </motion.div>
      </motion.div>

      {/* Message */}
      <motion.h1
        style={{
          fontFamily: 'var(--font-heading)',
          fontWeight: 200,
          fontSize: '24px',
          color: '#ffffff',
          letterSpacing: '0.1em',
          textAlign: 'center',
        }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.5 }}
      >
        tracker connected
      </motion.h1>

      {/* Loading dots */}
      <div style={{ display: 'flex', gap: 8, marginTop: 24 }}>
        {[0, 1, 2].map((index) => (
          <motion.div
            key={index}
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              backgroundColor: '#FF2D55',
            }}
            animate={{
              opacity: [0.3, 1, 0.3],
              scale: [0.8, 1.2, 0.8],
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              delay: index * 0.2,
            }}
          />
        ))}
      </div>
    </div>
  );
}
