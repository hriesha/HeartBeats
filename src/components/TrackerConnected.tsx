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
    <div className="relative w-full h-full overflow-hidden" style={{ fontFamily: 'Poppins, sans-serif' }}>
      {/* Background with gradient overlay */}
      <div 
        className="absolute inset-0 z-0"
        style={{
          background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
        }}
      />
      
      {/* Content */}
      <div className="relative z-10 w-full h-full flex flex-col items-center justify-center px-6">
        {/* Success Icon */}
        <motion.div
          className="relative mb-8"
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <div 
            className="rounded-full p-8"
            style={{
              background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)',
              boxShadow: '0 8px 32px rgba(252, 191, 73, 0.6)'
            }}
          >
            <Watch className="w-16 h-16 text-white" />
          </div>
          
          {/* Check mark overlay */}
          <motion.div
            className="absolute -bottom-2 -right-2 rounded-full p-2"
            style={{
              background: '#D62828',
              boxShadow: '0 4px 16px rgba(214, 40, 40, 0.6)'
            }}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.4, duration: 0.3 }}
          >
            <Check className="w-6 h-6 text-white" strokeWidth={3} />
          </motion.div>
        </motion.div>

        {/* Success Message */}
        <motion.h1 
          style={{
            fontFamily: 'Poppins, sans-serif',
            fontWeight: 700,
            fontSize: '32px',
            color: '#EAE2B7',
            textShadow: '0 2px 8px rgba(252, 191, 73, 0.4)',
            textAlign: 'center'
          }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          tracker connected
        </motion.h1>

        {/* Loading dots */}
        <div className="flex gap-2 mt-6">
          {[0, 1, 2].map((index) => (
            <motion.div
              key={index}
              className="rounded-full"
              style={{
                width: '8px',
                height: '8px',
                backgroundColor: '#FCBF49'
              }}
              animate={{
                opacity: [0.3, 1, 0.3],
                scale: [0.8, 1.2, 0.8]
              }}
              transition={{
                duration: 1,
                repeat: Infinity,
                delay: index * 0.2
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
