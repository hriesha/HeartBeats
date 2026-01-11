import { motion } from 'motion/react';

export function SignInButton() {
  return (
    <motion.button
      type="submit"
      className="w-full py-3 rounded-xl transition-all flex items-center justify-center"
      style={{
        background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)',
        color: 'white',
        fontFamily: 'Poppins, sans-serif',
        fontSize: '16px',
        fontWeight: 700,
        boxShadow: '0 4px 16px rgba(247, 127, 0, 0.4)',
        border: 'none',
        cursor: 'pointer'
      }}
      whileHover={{ 
        scale: 1.02,
        boxShadow: '0 6px 24px rgba(247, 127, 0, 0.6)'
      }}
      whileTap={{ scale: 0.98 }}
      transition={{ duration: 0.2 }}
    >
      sign in
    </motion.button>
  );
}