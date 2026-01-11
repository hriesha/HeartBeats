import { motion } from 'motion/react';
import { Heart } from 'lucide-react';

export function LoginHeader() {
  return (
    <div className="flex flex-col items-center mb-8">
      {/* Animated Logo */}
      <motion.div
        className="relative w-[120px] h-[120px] rounded-full mb-6 flex items-center justify-center shadow-lg"
        style={{
          background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)'
        }}
        animate={{
          scale: [1, 1.05, 1],
        }}
        transition={{
          duration: 1.5,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      >
        <Heart 
          className="w-16 h-16 text-white fill-white"
        />
      </motion.div>

      {/* Title */}
      <h1 
        className="mb-2 text-center tracking-tight"
        style={{
          fontFamily: 'Poppins, sans-serif',
          fontWeight: 700,
          fontSize: '36px',
          color: '#EAE2B7',
          textShadow: '0 2px 8px rgba(252, 191, 73, 0.4)'
        }}
      >
        HeartBeats
      </h1>

      {/* Subtitle */}
      <p 
        className="text-center"
        style={{
          fontFamily: 'Poppins, sans-serif',
          fontSize: '16px',
          color: '#EAE2B7',
          fontWeight: 400,
          opacity: 0.8
        }}
      >
        music that moves with you
      </p>
    </div>
  );
}