import { motion } from 'motion/react';
import { Music, ChevronLeft, Check } from 'lucide-react';
import { useState } from 'react';

interface SpotifyConnectProps {
  onConnected: () => void;
  onBack?: () => void;
  onSkip?: () => void;
}

export function SpotifyConnect({ onConnected, onBack, onSkip }: SpotifyConnectProps) {
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  const handleConnect = async () => {
    setIsConnecting(true);

    try {
      // TODO: Replace with actual backend API call
      // For now, simulate connection
      // const response = await fetch('/api/spotify/connect', { method: 'POST' });
      // const data = await response.json();

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));

      setIsConnected(true);
      setIsConnecting(false);

      // Auto-advance after showing success
      setTimeout(() => {
        onConnected();
      }, 1500);
    } catch (error) {
      console.error('Failed to connect to Spotify:', error);
      setIsConnecting(false);
      // TODO: Show error message to user
    }
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
      <div className="relative z-10 w-full h-full flex flex-col items-center justify-center px-6 py-12">
        {/* Back Button */}
        {onBack && (
          <button
            onClick={onBack}
            className="absolute top-4 left-4 p-2 rounded-full transition-all z-20"
            style={{
              backgroundColor: 'rgba(0, 48, 73, 0.8)',
              color: '#FCBF49'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(0, 48, 73, 1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(0, 48, 73, 0.8)';
            }}
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
        )}

        {/* Skip Button */}
        {onSkip && !isConnected && (
          <button
            onClick={onSkip}
            className="absolute top-4 right-4 p-2 transition-all z-20"
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '14px',
              color: '#EAE2B7',
              fontWeight: 500,
              opacity: 0.7
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.opacity = '1';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.opacity = '0.7';
            }}
          >
            skip
          </button>
        )}

        {/* Header */}
        <div className="mb-12 text-center">
          <h1
            className="mb-4"
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontWeight: 700,
              fontSize: '32px',
              color: '#EAE2B7',
              textShadow: '0 2px 8px rgba(252, 191, 73, 0.4)'
            }}
          >
            connect to spotify
          </h1>
          <p
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '16px',
              color: '#EAE2B7',
              fontWeight: 400,
              opacity: 0.8,
              lineHeight: 1.5,
              maxWidth: '280px',
              margin: '0 auto'
            }}
          >
            connect your account to sync your music library and get personalized recommendations
          </p>
        </div>

        {/* Spotify Icon/Logo */}
        {!isConnected ? (
          <motion.div
            className="mb-12"
            animate={isConnecting ? {
              scale: [1, 1.1, 1],
            } : {}}
            transition={{
              duration: 1.5,
              repeat: isConnecting ? Infinity : 0,
              ease: "easeInOut"
            }}
          >
            <div
              className="rounded-full p-8"
              style={{
                background: isConnecting
                  ? 'linear-gradient(135deg, #1DB954 0%, #1ed760 100%)'
                  : 'linear-gradient(135deg, rgba(252, 191, 73, 0.15) 0%, rgba(247, 127, 0, 0.15) 100%)',
                border: '2px solid #FCBF49',
                boxShadow: isConnecting
                  ? '0 8px 32px rgba(29, 185, 84, 0.5)'
                  : '0 8px 24px rgba(252, 191, 73, 0.3)',
                width: '140px',
                height: '140px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <Music className="w-16 h-16" style={{ color: '#FCBF49' }} />
            </div>
          </motion.div>
        ) : (
          <motion.div
            className="mb-12"
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          >
            <div
              className="rounded-full p-8 relative"
              style={{
                background: 'linear-gradient(135deg, #1DB954 0%, #1ed760 100%)',
                boxShadow: '0 8px 32px rgba(29, 185, 84, 0.6)',
                width: '140px',
                height: '140px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <Music className="w-16 h-16 text-white" />

              {/* Check mark overlay */}
              <motion.div
                className="absolute -bottom-2 -right-2 rounded-full p-2"
                style={{
                  background: '#FCBF49',
                  boxShadow: '0 4px 16px rgba(252, 191, 73, 0.6)'
                }}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.4, duration: 0.3 }}
              >
                <Check className="w-6 h-6 text-white" strokeWidth={3} />
              </motion.div>
            </div>
          </motion.div>
        )}

        {/* Connect Button */}
        {!isConnected && (
          <motion.button
            onClick={handleConnect}
            disabled={isConnecting}
            className="w-full max-w-sm rounded-2xl py-4 flex items-center justify-center gap-3"
            style={{
              background: isConnecting
                ? 'linear-gradient(135deg, #1DB954 0%, #1ed760 100%)'
                : 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)',
              color: 'white',
              fontFamily: 'Poppins, sans-serif',
              fontSize: '18px',
              fontWeight: 700,
              boxShadow: isConnecting
                ? '0 4px 16px rgba(29, 185, 84, 0.4)'
                : '0 4px 16px rgba(247, 127, 0, 0.4)',
              border: 'none',
              cursor: isConnecting ? 'wait' : 'pointer',
              opacity: isConnecting ? 0.9 : 1
            }}
            whileHover={!isConnecting ? {
              scale: 1.02,
              boxShadow: '0 6px 24px rgba(247, 127, 0, 0.6)'
            } : {}}
            whileTap={!isConnecting ? { scale: 0.98 } : {}}
            transition={{ duration: 0.2 }}
          >
            {isConnecting ? (
              <>
                <motion.div
                  className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{
                    duration: 1,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                />
                <span>connecting...</span>
              </>
            ) : (
              <>
                <Music className="w-5 h-5" />
                <span>connect to spotify</span>
              </>
            )}
          </motion.button>
        )}

        {/* Success Message */}
        {isConnected && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="text-center"
          >
            <h2
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontWeight: 700,
                fontSize: '24px',
                color: '#EAE2B7',
                textShadow: '0 2px 8px rgba(252, 191, 73, 0.4)',
                marginBottom: '8px'
              }}
            >
              connected successfully
            </h2>
            <p
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '14px',
                color: '#EAE2B7',
                opacity: 0.8
              }}
            >
              your spotify account is now linked
            </p>
          </motion.div>
        )}

        {/* Benefits Section */}
        {!isConnected && (
          <motion.div
            className="mt-12 w-full max-w-sm"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            <div
              className="rounded-2xl p-6"
              style={{
                backgroundColor: 'rgba(0, 48, 73, 0.4)',
                border: '1px solid rgba(252, 191, 73, 0.2)'
              }}
            >
              <h3
                style={{
                  fontFamily: 'Poppins, sans-serif',
                  fontSize: '14px',
                  fontWeight: 600,
                  color: '#FCBF49',
                  marginBottom: '12px'
                }}
              >
                what you'll get:
              </h3>
              <ul className="space-y-2" style={{ listStyle: 'none', padding: 0 }}>
                {[
                  'access to your saved tracks',
                  'personalized recommendations',
                  'sync across devices',
                  'create custom playlists'
                ].map((benefit, index) => (
                  <li
                    key={index}
                    style={{
                      fontFamily: 'Poppins, sans-serif',
                      fontSize: '13px',
                      color: '#EAE2B7',
                      opacity: 0.9,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}
                  >
                    <span style={{ color: '#FCBF49', fontSize: '16px' }}>•</span>
                    {benefit}
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
