import { motion } from 'motion/react';
import { ChevronLeft } from 'lucide-react';
import { VibeType } from '../App';
import { useState } from 'react';

interface VibeSelectionProps {
  bpm: number;
  onVibeSelect: (vibe: VibeType) => void;
  onBack: () => void;
}

const vibes: VibeType[] = [
  {
    id: 'chill',
    name: 'Chill Flow',
    color: '#EAE2B7',
    tags: ['lo-fi', 'calm']
  },
  {
    id: 'focus',
    name: 'Focus Pulse',
    color: '#FCBF49',
    tags: ['deep work', 'concentration']
  },
  {
    id: 'energize',
    name: 'Energy Rush',
    color: '#F77F00',
    tags: ['upbeat', 'motivating']
  },
  {
    id: 'intense',
    name: 'Intense Beats',
    color: '#D62828',
    tags: ['powerful', 'high-energy']
  }
];

export function VibeSelection({ bpm, onVibeSelect, onBack }: VibeSelectionProps) {
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [selectedVibe, setSelectedVibe] = useState<VibeType | null>(null);

  const handleVibeClick = (vibe: VibeType) => {
    setSelectedVibe(vibe);
    setIsTransitioning(true);
    // Wait for transition before navigating
    setTimeout(() => {
      onVibeSelect(vibe);
    }, 800);
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
      
      {/* Zoom Transition Overlay */}
      {isTransitioning && selectedVibe && (
        <motion.div
          className="absolute inset-0 z-50 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <motion.div
            className="rounded-full"
            style={{
              backgroundColor: selectedVibe.color,
              width: '155px',
              height: '155px'
            }}
            initial={{ scale: 1 }}
            animate={{ scale: 20 }}
            transition={{ duration: 0.8, ease: "easeInOut" }}
          />
        </motion.div>
      )}
      
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
        <div className="mt-8 mb-8">
          <p 
            className="mb-2"
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '14px',
              color: '#FCBF49',
              fontWeight: 500
            }}
          >
            based on {bpm} bpm
          </p>
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
            Choose your vibe
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
            we've clustered your tracks into 4 moods. pick how you want to feel.
          </p>
        </div>

        {/* Vibe Bubbles */}
        <div className="flex-1 flex items-center justify-center">
          <div className="relative w-full" style={{ height: '400px' }}>
            {/* Top Left */}
            <motion.button
              onClick={() => handleVibeClick(vibes[0])}
              className="absolute"
              style={{
                top: '10%',
                left: '5%',
                width: '140px',
                height: '140px',
                borderRadius: '50%',
                backgroundColor: vibes[0].color,
                boxShadow: `0 8px 24px rgba(234, 226, 183, 0.4), inset 0 2px 8px rgba(255, 255, 255, 0.3)`,
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '16px'
              }}
              animate={{
                y: [0, -15, 0],
                x: [0, 10, 0],
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              whileHover={{ 
                scale: 1.1,
                boxShadow: `0 12px 32px rgba(234, 226, 183, 0.6), inset 0 2px 8px rgba(255, 255, 255, 0.3)`
              }}
              whileTap={{ scale: 0.95 }}
            >
              <span style={{ fontFamily: 'Poppins, sans-serif', fontSize: '16px', fontWeight: 700, color: '#370617', textAlign: 'center', lineHeight: 1.2 }}>
                {vibes[0].name}
              </span>
              <span style={{ fontFamily: 'Poppins, sans-serif', fontSize: '11px', fontWeight: 500, color: '#6A040F', marginTop: '4px' }}>
                {vibes[0].tags.join(' • ')}
              </span>
            </motion.button>

            {/* Top Right */}
            <motion.button
              onClick={() => handleVibeClick(vibes[1])}
              className="absolute"
              style={{
                top: '15%',
                right: '8%',
                width: '150px',
                height: '150px',
                borderRadius: '50%',
                backgroundColor: vibes[1].color,
                boxShadow: `0 8px 24px rgba(252, 191, 73, 0.4), inset 0 2px 8px rgba(255, 255, 255, 0.3)`,
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '16px'
              }}
              animate={{
                y: [0, 20, 0],
                x: [0, -8, 0],
              }}
              transition={{
                duration: 5,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 0.5
              }}
              whileHover={{ 
                scale: 1.1,
                boxShadow: `0 12px 32px rgba(252, 191, 73, 0.6), inset 0 2px 8px rgba(255, 255, 255, 0.3)`
              }}
              whileTap={{ scale: 0.95 }}
            >
              <span style={{ fontFamily: 'Poppins, sans-serif', fontSize: '16px', fontWeight: 700, color: '#03071E', textAlign: 'center', lineHeight: 1.2 }}>
                {vibes[1].name}
              </span>
              <span style={{ fontFamily: 'Poppins, sans-serif', fontSize: '11px', fontWeight: 500, color: '#370617', marginTop: '4px' }}>
                {vibes[1].tags.join(' • ')}
              </span>
            </motion.button>

            {/* Bottom Left */}
            <motion.button
              onClick={() => handleVibeClick(vibes[2])}
              className="absolute"
              style={{
                bottom: '20%',
                left: '12%',
                width: '145px',
                height: '145px',
                borderRadius: '50%',
                backgroundColor: vibes[2].color,
                boxShadow: `0 8px 24px rgba(247, 127, 0, 0.4), inset 0 2px 8px rgba(255, 255, 255, 0.3)`,
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '16px'
              }}
              animate={{
                y: [0, -18, 0],
                x: [0, 12, 0],
              }}
              transition={{
                duration: 4.5,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 1
              }}
              whileHover={{ 
                scale: 1.1,
                boxShadow: `0 12px 32px rgba(247, 127, 0, 0.6), inset 0 2px 8px rgba(255, 255, 255, 0.3)`
              }}
              whileTap={{ scale: 0.95 }}
            >
              <span style={{ fontFamily: 'Poppins, sans-serif', fontSize: '16px', fontWeight: 700, color: '#03071E', textAlign: 'center', lineHeight: 1.2 }}>
                {vibes[2].name}
              </span>
              <span style={{ fontFamily: 'Poppins, sans-serif', fontSize: '11px', fontWeight: 500, color: '#370617', marginTop: '4px' }}>
                {vibes[2].tags.join(' • ')}
              </span>
            </motion.button>

            {/* Bottom Right */}
            <motion.button
              onClick={() => handleVibeClick(vibes[3])}
              className="absolute"
              style={{
                bottom: '15%',
                right: '5%',
                width: '155px',
                height: '155px',
                borderRadius: '50%',
                backgroundColor: vibes[3].color,
                boxShadow: `0 8px 24px rgba(214, 40, 40, 0.4), inset 0 2px 8px rgba(255, 255, 255, 0.3)`,
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '16px'
              }}
              animate={{
                y: [0, 22, 0],
                x: [0, -15, 0],
              }}
              transition={{
                duration: 5.5,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 1.5
              }}
              whileHover={{ 
                scale: 1.1,
                boxShadow: `0 12px 32px rgba(214, 40, 40, 0.6), inset 0 2px 8px rgba(255, 255, 255, 0.3)`
              }}
              whileTap={{ scale: 0.95 }}
            >
              <span style={{ fontFamily: 'Poppins, sans-serif', fontSize: '16px', fontWeight: 700, color: '#370617', textAlign: 'center', lineHeight: 1.2 }}>
                {vibes[3].name}
              </span>
              <span style={{ fontFamily: 'Poppins, sans-serif', fontSize: '11px', fontWeight: 500, color: '#6A040F', marginTop: '4px' }}>
                {vibes[3].tags.join(' • ')}
              </span>
            </motion.button>
          </div>
        </div>

        {/* Bottom Hint */}
        <div className="text-center">
          <p 
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '14px',
              color: '#FFDAB9',
              fontWeight: 500,
              opacity: 0.7
            }}
          >
            tap a vibe bubble to dive in
          </p>
        </div>
      </div>
    </div>
  );
}