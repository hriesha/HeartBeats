import { motion } from 'motion/react';
import { Music, AlertCircle } from 'lucide-react';
import { useState } from 'react';
import { getSpotifyAuthUrl } from '../utils/api';

interface SpotifyConnectProps {
  onConnected?: () => void;  // Optional - not used with redirect flow, but kept for compatibility
  onSkip?: () => void;
}

export function SpotifyConnect({ onSkip }: SpotifyConnectProps) {
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Note: isConnected state removed - with OAuth redirect flow, user is redirected
  // to Spotify then back to the app, so this component unmounts during the process

  const handleConnect = async () => {
    setIsConnecting(true);
    setError(null);

    try {
      // Get the Spotify authorization URL from backend
      const authUrl = await getSpotifyAuthUrl();

      if (authUrl) {
        // Redirect user to Spotify login page
        // After login, Spotify will redirect back to our callback endpoint
        // which then redirects to frontend with ?spotify_connected=true
        window.location.href = authUrl;
      } else {
        throw new Error('Failed to get Spotify authorization URL. Make sure the backend is running.');
      }
    } catch (err) {
      console.error('Failed to start Spotify OAuth:', err);
      setError(String(err));
      setIsConnecting(false);
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
        {/* Skip Button */}
        {onSkip && (
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

        {/* Connect Button */}
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

        {/* Benefits Section */}
        {!error && (
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
                  'play music directly on Spotify',
                  'personalized recommendations',
                  'control playback from the app',
                  'seamless workout experience'
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

        {/* Error Message */}
        {error && (
          <motion.div
            className="mt-8 w-full max-w-sm"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div
              className="rounded-2xl p-4 flex items-start gap-3"
              style={{
                backgroundColor: 'rgba(214, 40, 40, 0.2)',
                border: '1px solid rgba(214, 40, 40, 0.4)'
              }}
            >
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: '#D62828' }} />
              <div>
                <p
                  style={{
                    fontFamily: 'Poppins, sans-serif',
                    fontSize: '14px',
                    fontWeight: 600,
                    color: '#EAE2B7',
                    marginBottom: '4px'
                  }}
                >
                  Connection failed
                </p>
                <p
                  style={{
                    fontFamily: 'Poppins, sans-serif',
                    fontSize: '12px',
                    color: '#EAE2B7',
                    opacity: 0.8
                  }}
                >
                  {error}
                </p>
                <button
                  onClick={() => setError(null)}
                  style={{
                    fontFamily: 'Poppins, sans-serif',
                    fontSize: '12px',
                    color: '#FCBF49',
                    marginTop: '8px',
                    textDecoration: 'underline'
                  }}
                >
                  Try again
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
