import { motion, AnimatePresence } from 'motion/react';
import { X, Send } from 'lucide-react';
import { useState } from 'react';

interface FeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const FEEDBACK_EMAIL = 'hriesha.p@gmail.com';

const quickTags = [
  'love it',
  'needs work',
  'song selection',
  'playback issue',
  'ui/design',
  'feature request',
];

export function FeedbackModal({ isOpen, onClose }: FeedbackModalProps) {
  const [message, setMessage] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [sent, setSent] = useState(false);

  const toggleTag = (tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    );
  };

  const handleSend = () => {
    const tagLine = selectedTags.length > 0 ? `[${selectedTags.join(', ')}]\n\n` : '';
    const body = encodeURIComponent(`${tagLine}${message}\n\n---\nSent from HeartBeats beta`);
    const subject = encodeURIComponent(`HeartBeats Beta Feedback${selectedTags.length > 0 ? ` - ${selectedTags[0]}` : ''}`);
    window.open(`mailto:${FEEDBACK_EMAIL}?subject=${subject}&body=${body}`, '_self');
    setSent(true);
    setTimeout(() => {
      setSent(false);
      setMessage('');
      setSelectedTags([]);
      onClose();
    }, 1500);
  };

  const handleClose = () => {
    setMessage('');
    setSelectedTags([]);
    setSent(false);
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          style={{
            position: 'fixed', inset: 0, zIndex: 100,
            display: 'flex', alignItems: 'flex-end', justifyContent: 'center',
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Backdrop */}
          <motion.div
            style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.6)' }}
            onClick={handleClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />

          {/* Modal */}
          <motion.div
            style={{
              position: 'relative', width: '100%', maxWidth: 400,
              background: '#141414',
              borderRadius: '20px 20px 0 0',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderBottom: 'none',
              padding: `24px 24px calc(24px + var(--safe-area-bottom))`,
            }}
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          >
            {/* Handle bar */}
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16 }}>
              <div style={{ width: 36, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.15)' }} />
            </div>

            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
              <div>
                <h2 style={{
                  fontFamily: 'var(--font-heading)', fontSize: '22px', fontWeight: 200,
                  color: '#ffffff', letterSpacing: '0.08em',
                }}>
                  send feedback
                </h2>
                <p style={{
                  fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 300,
                  color: 'rgba(255,255,255,0.4)', marginTop: 4,
                }}>
                  help us make heartbeats better
                </p>
              </div>
              <button
                onClick={handleClose}
                style={{
                  width: 36, height: 36, borderRadius: '50%',
                  background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                  color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer',
                }}
              >
                <X style={{ width: 16, height: 16 }} />
              </button>
            </div>

            {sent ? (
              <motion.div
                style={{ textAlign: 'center', padding: '32px 0' }}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
              >
                <p style={{
                  fontFamily: 'var(--font-heading)', fontSize: '20px', fontWeight: 200,
                  color: '#FF2D55', letterSpacing: '0.08em',
                }}>
                  thank you!
                </p>
                <p style={{
                  fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 300,
                  color: 'rgba(255,255,255,0.4)', marginTop: 8,
                }}>
                  your feedback means everything
                </p>
              </motion.div>
            ) : (
              <>
                {/* Quick Tags */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
                  {quickTags.map(tag => (
                    <button
                      key={tag}
                      onClick={() => toggleTag(tag)}
                      style={{
                        padding: '6px 14px', borderRadius: '16px',
                        fontFamily: 'var(--font-body)', fontSize: '12px', fontWeight: 400,
                        cursor: 'pointer', minHeight: 32,
                        background: selectedTags.includes(tag) ? 'rgba(255, 45, 85, 0.15)' : 'rgba(255,255,255,0.04)',
                        color: selectedTags.includes(tag) ? '#FF2D55' : 'rgba(255,255,255,0.5)',
                        border: selectedTags.includes(tag) ? '1px solid rgba(255, 45, 85, 0.3)' : '1px solid rgba(255,255,255,0.08)',
                      }}
                    >
                      {tag}
                    </button>
                  ))}
                </div>

                {/* Text Area */}
                <textarea
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="what's on your mind? anything you wish was different, or features you'd love to see..."
                  style={{
                    width: '100%', minHeight: 120, padding: 16,
                    borderRadius: '12px', resize: 'vertical',
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    color: '#ffffff', fontFamily: 'var(--font-body)',
                    fontSize: '14px', fontWeight: 300, lineHeight: 1.6,
                    outline: 'none',
                    WebkitUserSelect: 'text',
                    userSelect: 'text',
                  }}
                />

                {/* Send Button */}
                <motion.button
                  onClick={handleSend}
                  disabled={!message.trim() && selectedTags.length === 0}
                  style={{
                    width: '100%', marginTop: 16,
                    padding: '14px', borderRadius: '12px',
                    background: (!message.trim() && selectedTags.length === 0) ? 'rgba(255, 45, 85, 0.1)' : '#FF2D55',
                    color: '#ffffff', border: 'none',
                    fontFamily: 'var(--font-body)', fontSize: '15px', fontWeight: 500,
                    cursor: (!message.trim() && selectedTags.length === 0) ? 'default' : 'pointer',
                    opacity: (!message.trim() && selectedTags.length === 0) ? 0.4 : 1,
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                    minHeight: 48,
                  }}
                  whileTap={message.trim() || selectedTags.length > 0 ? { scale: 0.98 } : {}}
                >
                  <Send style={{ width: 16, height: 16 }} />
                  send feedback
                </motion.button>
              </>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
