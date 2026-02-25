import { motion, AnimatePresence } from 'motion/react';
import { ChevronLeft, Search, X } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { VibeType } from '../App';
import { getVibeArtists } from '../utils/api';

const MAX_ARTISTS = 5;

interface ArtistSelectionProps {
  vibe: VibeType;
  onArtistSelect: (artists: string[]) => void;
  onBack: () => void;
}

export function ArtistSelection({ vibe, onArtistSelect, onBack }: ArtistSelectionProps) {
  const [artists, setArtists] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [customInput, setCustomInput] = useState('');
  const [showInput, setShowInput] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  const vibeId = parseInt(vibe.id.split('-')[1] || '0');

  useEffect(() => {
    getVibeArtists(vibeId).then((list) => {
      setArtists(list);
      setIsLoading(false);
    });
  }, [vibeId]);

  useEffect(() => {
    if (showInput) inputRef.current?.focus();
  }, [showInput]);

  const toggleChip = (name: string) => {
    setSelected(prev => {
      if (prev.includes(name)) return prev.filter(n => n !== name);
      if (prev.length >= MAX_ARTISTS) return prev;
      return [...prev, name];
    });
  };

  const handleCustomSubmit = () => {
    const trimmed = customInput.trim();
    if (!trimmed) return;
    if (!selected.includes(trimmed) && selected.length < MAX_ARTISTS) {
      setSelected(prev => [...prev, trimmed]);
    }
    setCustomInput('');
    setShowInput(false);
  };

  const removeSelected = (name: string) => {
    setSelected(prev => prev.filter(n => n !== name));
  };

  const vibeColor = vibe.color || '#FF2D55';
  const vibeColorRgb = vibeColor.replace('#', '').match(/.{2}/g)
    ?.map(h => parseInt(h, 16)).join(', ') || '255, 45, 85';

  const atMax = selected.length >= MAX_ARTISTS;

  return (
    <div style={{
      minHeight: '100dvh',
      display: 'flex',
      flexDirection: 'column',
      padding: 'calc(72px + var(--safe-area-top)) 24px calc(32px + var(--safe-area-bottom))',
      position: 'relative',
    }}>
      {/* Back */}
      <button
        onClick={onBack}
        style={{
          position: 'absolute',
          top: 'calc(20px + var(--safe-area-top))',
          left: 20,
          width: 44, height: 44, borderRadius: '50%',
          background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.1)',
          color: '#ffffff',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer',
        }}
      >
        <ChevronLeft style={{ width: 20, height: 20 }} />
      </button>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        style={{ marginBottom: 32 }}
      >
        <p style={{
          fontFamily: 'var(--font-body)', fontSize: '12px', color: vibeColor,
          fontWeight: 400, marginBottom: 8, letterSpacing: '0.05em', textTransform: 'uppercase',
        }}>
          {vibe.name}
        </p>
        <h1 style={{
          fontFamily: 'var(--font-heading)', fontWeight: 200, fontSize: '28px',
          color: '#ffffff', letterSpacing: '0.1em', marginBottom: 8,
        }}>
          pick your artists
        </h1>
        <p style={{
          fontFamily: 'var(--font-body)', fontSize: '14px',
          color: 'rgba(255,255,255,0.35)', fontWeight: 300,
        }}>
          up to {MAX_ARTISTS} · we'll match their hits to your pace
        </p>
      </motion.div>

      {/* Selected pills strip */}
      <AnimatePresence>
        {selected.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0, marginBottom: 0 }}
            animate={{ opacity: 1, height: 'auto', marginBottom: 20 }}
            exit={{ opacity: 0, height: 0, marginBottom: 0 }}
            style={{ display: 'flex', flexWrap: 'wrap', gap: 8, overflow: 'hidden' }}
          >
            {selected.map(name => (
              <motion.div
                key={name}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  padding: '6px 12px', borderRadius: 20,
                  background: `rgba(${vibeColorRgb}, 0.15)`,
                  border: `1.5px solid ${vibeColor}`,
                  color: vibeColor,
                  fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 500,
                }}
              >
                {name}
                <button
                  onClick={() => removeSelected(name)}
                  style={{ background: 'none', border: 'none', color: vibeColor, cursor: 'pointer', padding: 0, display: 'flex', lineHeight: 1 }}
                >
                  <X style={{ width: 11, height: 11 }} />
                </button>
              </motion.div>
            ))}
            {/* Counter */}
            <div style={{
              padding: '6px 10px', borderRadius: 20,
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.08)',
              color: atMax ? vibeColor : 'rgba(255,255,255,0.3)',
              fontFamily: 'var(--font-body)', fontSize: '12px', fontWeight: 400,
              display: 'flex', alignItems: 'center',
            }}>
              {selected.length}/{MAX_ARTISTS}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Artist chips grid */}
      <div style={{ flex: 1 }}>
        {isLoading ? (
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {[1,2,3,4,5,6,7,8,9].map(i => (
              <div key={i} style={{
                height: 40, width: 80 + i * 7, borderRadius: 20,
                background: 'rgba(255,255,255,0.05)',
              }} />
            ))}
          </div>
        ) : (
          <motion.div
            style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}
            initial="hidden"
            animate="visible"
            variants={{ visible: { transition: { staggerChildren: 0.05 } } }}
          >
            {artists.map((name) => {
              const isActive = selected.includes(name);
              const isDisabled = atMax && !isActive;
              return (
                <motion.button
                  key={name}
                  onClick={() => !isDisabled && toggleChip(name)}
                  variants={{ hidden: { opacity: 0, scale: 0.85 }, visible: { opacity: 1, scale: 1 } }}
                  whileTap={!isDisabled ? { scale: 0.95 } : {}}
                  style={{
                    padding: '10px 18px', borderRadius: 24,
                    border: isActive
                      ? `1.5px solid ${vibeColor}`
                      : '1px solid rgba(255,255,255,0.1)',
                    background: isActive
                      ? `rgba(${vibeColorRgb}, 0.12)`
                      : 'rgba(255,255,255,0.04)',
                    color: isActive ? vibeColor : isDisabled ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.7)',
                    fontFamily: 'var(--font-body)', fontSize: '14px',
                    fontWeight: isActive ? 500 : 400,
                    cursor: isDisabled ? 'not-allowed' : 'pointer',
                    transition: 'all 0.15s ease',
                    boxShadow: isActive ? `0 0 16px rgba(${vibeColorRgb}, 0.2)` : 'none',
                    opacity: isDisabled ? 0.4 : 1,
                  }}
                >
                  {name}
                </motion.button>
              );
            })}
          </motion.div>
        )}

        {/* Custom artist input */}
        <motion.div
          style={{ marginTop: 24 }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          {!showInput ? (
            <button
              onClick={() => !atMax && setShowInput(true)}
              disabled={atMax}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '10px 18px', borderRadius: 24,
                border: '1px dashed rgba(255,255,255,0.15)',
                background: 'transparent',
                color: atMax ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.4)',
                fontFamily: 'var(--font-body)', fontSize: '14px', fontWeight: 300,
                cursor: atMax ? 'not-allowed' : 'pointer',
              }}
            >
              <Search style={{ width: 14, height: 14 }} />
              type an artist name
            </button>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '8px 16px', borderRadius: 24,
                border: `1.5px solid rgba(${vibeColorRgb}, 0.4)`,
                background: `rgba(${vibeColorRgb}, 0.06)`,
              }}
            >
              <Search style={{ width: 14, height: 14, color: 'rgba(255,255,255,0.4)', flexShrink: 0 }} />
              <input
                ref={inputRef}
                value={customInput}
                onChange={e => setCustomInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleCustomSubmit(); }}
                placeholder="artist name..."
                style={{
                  flex: 1, background: 'transparent', border: 'none', outline: 'none',
                  color: '#ffffff', fontFamily: 'var(--font-body)', fontSize: '14px',
                }}
              />
              {customInput ? (
                <button
                  onClick={handleCustomSubmit}
                  style={{
                    padding: '4px 12px', borderRadius: 12, background: vibeColor,
                    border: 'none', color: '#000', fontFamily: 'var(--font-body)',
                    fontSize: '12px', fontWeight: 600, cursor: 'pointer', flexShrink: 0,
                  }}
                >
                  add
                </button>
              ) : (
                <button
                  onClick={() => setShowInput(false)}
                  style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.3)', cursor: 'pointer', padding: 4 }}
                >
                  <X style={{ width: 14, height: 14 }} />
                </button>
              )}
            </motion.div>
          )}
        </motion.div>
      </div>

      {/* Bottom actions */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        style={{ display: 'flex', flexDirection: 'column', gap: 12, paddingTop: 24 }}
      >
        <AnimatePresence>
          {selected.length > 0 && (
            <motion.button
              onClick={() => onArtistSelect(selected)}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              whileTap={{ scale: 0.97 }}
              style={{
                width: '100%', padding: '16px', borderRadius: 14,
                background: vibeColor, border: 'none',
                color: '#000000', fontFamily: 'var(--font-body)',
                fontSize: '15px', fontWeight: 600, cursor: 'pointer',
                letterSpacing: '0.02em',
              }}
            >
              {selected.length === 1
                ? `play ${selected[0]}`
                : `play ${selected.length} artists`}
            </motion.button>
          )}
        </AnimatePresence>

        <button
          onClick={() => onArtistSelect([])}
          style={{
            width: '100%', padding: '14px', borderRadius: 14,
            background: 'transparent', border: '1px solid rgba(255,255,255,0.08)',
            color: 'rgba(255,255,255,0.35)', fontFamily: 'var(--font-body)',
            fontSize: '14px', fontWeight: 300, cursor: 'pointer',
          }}
        >
          no preference — surprise me
        </button>
      </motion.div>
    </div>
  );
}
