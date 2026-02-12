import { motion } from 'motion/react';
import { Music, AlertCircle } from 'lucide-react';
import { useState } from 'react';

interface AppleMusicConnectProps {
  onConnected?: () => void;
  onSkip?: () => void;
}

export function AppleMusicConnect({ onConnected, onSkip }: AppleMusicConnectProps) {
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConnect = async () => {
    setIsConnecting(true);
    setError(null);

    try {
      const music = window.MusicKit?.getInstance?.();
      if (!music) {
        // MusicKit not yet configured — configure it now
        const developerToken = import.meta.env.VITE_APPLE_MUSIC_DEVELOPER_TOKEN;
        if (!developerToken) {
          throw new Error('Missing developer token. Check your .env file.');
        }
        const instance = await window.MusicKit.configure({
          developerToken,
          app: { name: 'HeartBeats', build: '1.0.0' },
        });
        await instance.authorize();
      } else {
        await music.authorize();
      }

      onConnected?.();
    } catch (err: any) {
      console.error('Apple Music auth failed:', err);
      const msg = err?.message || String(err);
      if (msg.includes('cancelled') || msg.includes('denied')) {
        setError('Authorization was cancelled. Please try again.');
      } else {
        setError(msg);
      }
      setIsConnecting(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '48px 24px', position: 'relative' }}>
      {/* Skip Button */}
      {onSkip && (
        <button
          onClick={onSkip}
          style={{
            position: 'absolute',
            top: 20,
            right: 20,
            fontFamily: 'var(--font-body)',
            fontSize: '13px',
            color: 'rgba(255,255,255,0.4)',
            fontWeight: 400,
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            letterSpacing: '0.05em',
          }}
        >
          skip
        </button>
      )}

      {/* Logo / Icon */}
      <motion.div
        style={{ marginBottom: 48 }}
        animate={isConnecting ? { scale: [1, 1.05, 1] } : {}}
        transition={{ duration: 2, repeat: isConnecting ? Infinity : 0, ease: 'easeInOut' }}
      >
        <div
          style={{
            width: 100,
            height: 100,
            borderRadius: '50%',
            border: '1.5px solid rgba(255, 45, 85, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: isConnecting
              ? 'rgba(255, 45, 85, 0.1)'
              : 'rgba(255, 255, 255, 0.03)',
            boxShadow: isConnecting
              ? '0 0 40px rgba(255, 45, 85, 0.2)'
              : '0 0 30px rgba(255, 45, 85, 0.08)',
          }}
        >
          <Music style={{ width: 36, height: 36, color: '#FF2D55' }} />
        </div>
      </motion.div>

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: 48 }}>
        <h1
          style={{
            fontFamily: 'var(--font-heading)',
            fontWeight: 200,
            fontSize: '36px',
            color: '#ffffff',
            letterSpacing: '0.12em',
            marginBottom: 16,
            textTransform: 'lowercase',
          }}
        >
          heartbeats
        </h1>
        <p
          style={{
            fontFamily: 'var(--font-body)',
            fontSize: '14px',
            color: 'rgba(255,255,255,0.45)',
            fontWeight: 300,
            lineHeight: 1.7,
            maxWidth: '280px',
            margin: '0 auto',
          }}
        >
          connect apple music to sync your library and get pace-matched recommendations
        </p>
      </div>

      {/* Connect Button */}
      <motion.button
        onClick={handleConnect}
        disabled={isConnecting}
        style={{
          width: '100%',
          maxWidth: '320px',
          padding: '16px 24px',
          borderRadius: '12px',
          background: isConnecting
            ? 'rgba(255, 45, 85, 0.15)'
            : '#FF2D55',
          color: '#ffffff',
          fontFamily: 'var(--font-body)',
          fontSize: '15px',
          fontWeight: 500,
          border: isConnecting ? '1px solid rgba(255, 45, 85, 0.3)' : '1px solid transparent',
          cursor: isConnecting ? 'wait' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '10px',
          letterSpacing: '0.03em',
        }}
        whileHover={!isConnecting ? { scale: 1.02, boxShadow: '0 0 30px rgba(255, 45, 85, 0.3)' } : {}}
        whileTap={!isConnecting ? { scale: 0.98 } : {}}
        transition={{ duration: 0.2 }}
      >
        {isConnecting ? (
          <>
            <motion.div
              style={{ width: 18, height: 18, border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#FF2D55', borderRadius: '50%' }}
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            />
            <span>connecting...</span>
          </>
        ) : (
          <>
            <Music style={{ width: 18, height: 18 }} />
            <span>connect to apple music</span>
          </>
        )}
      </motion.button>

      {/* Benefits */}
      {!error && (
        <motion.div
          style={{ marginTop: 48, width: '100%', maxWidth: '320px' }}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          <div
            style={{
              padding: '20px 24px',
              borderRadius: '12px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
            }}
          >
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {[
                'full track playback',
                'pace-matched playlists',
                'crossfade transitions',
                'no user limits'
              ].map((benefit, index) => (
                <li
                  key={index}
                  style={{
                    fontFamily: 'var(--font-body)',
                    fontSize: '13px',
                    color: 'rgba(255,255,255,0.4)',
                    fontWeight: 300,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                  }}
                >
                  <span style={{ color: '#FF2D55', fontSize: '6px', lineHeight: 1 }}>&#9679;</span>
                  {benefit}
                </li>
              ))}
            </ul>
          </div>
        </motion.div>
      )}

      {/* Error */}
      {error && (
        <motion.div
          style={{ marginTop: 32, width: '100%', maxWidth: '320px' }}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div
            style={{
              padding: '16px 20px',
              borderRadius: '12px',
              background: 'rgba(255, 59, 48, 0.08)',
              border: '1px solid rgba(255, 59, 48, 0.15)',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '12px',
            }}
          >
            <AlertCircle style={{ width: 18, height: 18, color: '#FF3B30', flexShrink: 0, marginTop: 1 }} />
            <div>
              <p style={{ fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 500, color: '#ffffff', marginBottom: 4 }}>
                Connection failed
              </p>
              <p style={{ fontFamily: 'var(--font-body)', fontSize: '12px', color: 'rgba(255,255,255,0.5)' }}>
                {error}
              </p>
              <button
                onClick={() => setError(null)}
                style={{
                  fontFamily: 'var(--font-body)',
                  fontSize: '12px',
                  color: '#FF2D55',
                  marginTop: 8,
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  textDecoration: 'underline',
                  textUnderlineOffset: '2px',
                }}
              >
                try again
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
