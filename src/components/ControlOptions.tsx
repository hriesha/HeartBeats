import { motion } from 'motion/react';
import { Sliders, Watch, ChevronLeft } from 'lucide-react';

interface ControlOptionsProps {
  onSelectCustom: () => void;
  onSelectWatch: () => void;
  onBack?: () => void;
}

export function ControlOptions({ onSelectCustom, onSelectWatch, onBack }: ControlOptionsProps) {
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
            className="absolute top-4 left-4 p-2 rounded-full transition-all"
            style={{
              backgroundColor: 'rgba(0, 48, 73, 0.8)',
              color: '#FCBF49'
            }}
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
        )}

        {/* Header */}
        <div className="mb-16 text-center">
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
            how do you want to track?
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
            choose your preferred control method
          </p>
        </div>

        {/* Options */}
        <div className="w-full space-y-6">
          {/* Custom Controls */}
          <motion.button
            onClick={onSelectCustom}
            className="w-full rounded-3xl p-8 flex flex-col items-center justify-center"
            style={{
              background: 'linear-gradient(135deg, rgba(252, 191, 73, 0.15) 0%, rgba(247, 127, 0, 0.15) 100%)',
              border: '2px solid #FCBF49',
              boxShadow: '0 8px 24px rgba(252, 191, 73, 0.3)',
              cursor: 'pointer'
            }}
            whileHover={{ 
              scale: 1.03,
              boxShadow: '0 12px 32px rgba(252, 191, 73, 0.5)'
            }}
            whileTap={{ scale: 0.97 }}
          >
            <div 
              className="rounded-full p-4 mb-4"
              style={{
                background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)'
              }}
            >
              <Sliders className="w-8 h-8 text-white" />
            </div>
            <h2 
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '24px',
                fontWeight: 700,
                color: '#EAE2B7',
                marginBottom: '8px'
              }}
            >
              custom controls
            </h2>
            <p 
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '14px',
                color: '#EAE2B7',
                textAlign: 'center',
                opacity: 0.8
              }}
            >
              manually set your BPM and preferences
            </p>
          </motion.button>

          {/* Connect Watch */}
          <motion.button
            onClick={onSelectWatch}
            className="w-full rounded-3xl p-8 flex flex-col items-center justify-center"
            style={{
              background: 'linear-gradient(135deg, rgba(252, 191, 73, 0.15) 0%, rgba(247, 127, 0, 0.15) 100%)',
              border: '2px solid #FCBF49',
              boxShadow: '0 8px 24px rgba(252, 191, 73, 0.3)',
              cursor: 'pointer'
            }}
            whileHover={{ 
              scale: 1.03,
              boxShadow: '0 12px 32px rgba(252, 191, 73, 0.5)'
            }}
            whileTap={{ scale: 0.97 }}
          >
            <div 
              className="rounded-full p-4 mb-4"
              style={{
                background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)'
              }}
            >
              <Watch className="w-8 h-8 text-white" />
            </div>
            <h2 
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '24px',
                fontWeight: 700,
                color: '#EAE2B7',
                marginBottom: '8px'
              }}
            >
              connect your watch
            </h2>
            <p 
              style={{
                fontFamily: 'Poppins, sans-serif',
                fontSize: '14px',
                color: '#EAE2B7',
                textAlign: 'center',
                opacity: 0.8
              }}
            >
              sync with your fitness tracker
            </p>
          </motion.button>
        </div>
      </div>
    </div>
  );
}