import { motion, AnimatePresence } from 'motion/react';
import { ChevronLeft, Search, X, Plus, Check } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import { VibeType } from '../App';
import { searchArtists } from '../utils/api';

interface ArtistSelectionProps {
  vibe: VibeType;
  bpm: number;
  topArtists: string[];
  onComplete: (artistNames: string[]) => void;
  onBack: () => void;
}

export function ArtistSelection({ vibe, topArtists, onComplete, onBack }: ArtistSelectionProps) {
  const [selected, setSelected] = useState<string[]>([]);
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Array<{ name: string; artworkUrl?: string }>>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const toggleArtist = (artist: string) => {
    setSelected(prev => {
      if (prev.includes(artist)) return prev.filter(a => a !== artist);
      if (prev.length >= 5) return [...prev.slice(1), artist];
      return [...prev, artist];
    });
  };

  const handleQueryChange = (value: string) => {
    setQuery(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!value.trim()) {
      setSearchResults([]);
      setShowDropdown(false);
      setIsSearching(false);
      return;
    }
    setIsSearching(true);
    setShowDropdown(true);
    debounceRef.current = setTimeout(async () => {
      const results = await searchArtists(value);
      setSearchResults(results);
      setIsSearching(false);
    }, 300);
  };

  const handleSelectFromSearch = (artist: string) => {
    toggleArtist(artist);
    setQuery('');
    setShowDropdown(false);
    setSearchResults([]);
    inputRef.current?.blur();
  };

  const clearSearch = () => {
    setQuery('');
    setShowDropdown(false);
    setSearchResults([]);
    setIsSearching(false);
    inputRef.current?.focus();
  };

  // Close dropdown when tapping outside
  useEffect(() => {
    const handler = (e: TouchEvent | MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('[data-search-container]')) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handler);
    document.addEventListener('touchstart', handler);
    return () => {
      document.removeEventListener('mousedown', handler);
      document.removeEventListener('touchstart', handler);
    };
  }, []);

  const artists = topArtists.length > 0 ? topArtists : [];

  return (
    <div style={{ minHeight: '100dvh', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      {/* Back button */}
      <button
        onClick={onBack}
        style={{
          position: 'absolute', top: 'calc(20px + var(--safe-area-top))', left: 20,
          width: 44, height: 44, borderRadius: '50%', zIndex: 10,
          background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)',
          color: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
        }}
      >
        <ChevronLeft style={{ width: 20, height: 20 }} />
      </button>

      {/* Content */}
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        padding: 'calc(80px + var(--safe-area-top)) 24px calc(160px + var(--safe-area-bottom))',
        overflowY: 'auto',
      }}>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          style={{ marginBottom: 24 }}
        >
          <h1 style={{
            fontFamily: 'var(--font-heading)', fontWeight: 200, fontSize: '26px',
            color: '#ffffff', letterSpacing: '0.08em', marginBottom: 8,
          }}>
            who do you want to hear?
          </h1>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'rgba(255,255,255,0.35)', fontWeight: 300 }}>
              pick up to 5 artists
            </p>
            {selected.length > 0 && (
              <motion.span
                key={selected.length}
                initial={{ scale: 1.15, color: '#FF2D55' }}
                animate={{ scale: 1, color: 'rgba(255,255,255,0.4)' }}
                transition={{ duration: 0.2 }}
                style={{ fontFamily: 'var(--font-body)', fontSize: '12px', fontWeight: 400 }}
              >
                {selected.length} / 5
              </motion.span>
            )}
          </div>

          {/* Vibe pill */}
          <div style={{ marginTop: 10 }}>
            <span style={{
              display: 'inline-block', padding: '4px 12px', borderRadius: '20px',
              background: 'rgba(255, 45, 85, 0.1)', border: '1px solid rgba(255, 45, 85, 0.2)',
              fontFamily: 'var(--font-body)', fontSize: '11px', fontWeight: 400, color: '#FF6B8A',
            }}>
              {vibe.name}
            </span>
          </div>
        </motion.div>

        {/* Search input */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          data-search-container
          style={{ position: 'relative', marginBottom: 24, zIndex: 30 }}
        >
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '12px 16px', borderRadius: '14px',
            background: 'rgba(255, 255, 255, 0.05)',
            border: `1px solid ${query ? 'rgba(255, 45, 85, 0.35)' : 'rgba(255, 255, 255, 0.1)'}`,
            transition: 'border-color 0.15s',
          }}>
            {isSearching ? (
              <motion.div
                style={{ width: 16, height: 16, border: '1.5px solid rgba(255,45,85,0.6)', borderTopColor: 'transparent', borderRadius: '50%', flexShrink: 0 }}
                animate={{ rotate: 360 }}
                transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
              />
            ) : (
              <Search style={{ width: 16, height: 16, color: query ? '#FF2D55' : 'rgba(255,255,255,0.3)', flexShrink: 0 }} />
            )}
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={e => handleQueryChange(e.target.value)}
              onFocus={() => query && setShowDropdown(true)}
              placeholder="search any artist..."
              style={{
                flex: 1, background: 'none', border: 'none', outline: 'none',
                fontFamily: 'var(--font-body)', fontSize: '14px', fontWeight: 300,
                color: '#ffffff',
              }}
            />
            {query && (
              <button onClick={clearSearch} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, display: 'flex' }}>
                <X style={{ width: 14, height: 14, color: 'rgba(255,255,255,0.4)' }} />
              </button>
            )}
          </div>

          {/* Dropdown */}
          <AnimatePresence>
            {showDropdown && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15 }}
                style={{
                  position: 'absolute', top: 'calc(100% + 6px)', left: 0, right: 0,
                  background: '#1a1a1a', border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '14px', overflow: 'hidden',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
                }}
              >
                {searchResults.length === 0 && !isSearching && (
                  <div style={{ padding: '14px 16px' }}>
                    <p style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'rgba(255,255,255,0.3)', fontWeight: 300 }}>
                      no results for "{query}"
                    </p>
                  </div>
                )}
                {searchResults.map((result, i) => {
                  const isSelected = selected.includes(result.name);
                  return (
                    <motion.button
                      key={result.name}
                      onClick={() => handleSelectFromSearch(result.name)}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.03 }}
                      style={{
                        width: '100%', padding: '12px 16px',
                        display: 'flex', alignItems: 'center', gap: 12,
                        background: isSelected ? 'rgba(255, 45, 85, 0.08)' : 'transparent',
                        border: 'none', borderBottom: i < searchResults.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none',
                        cursor: 'pointer', textAlign: 'left',
                      }}
                    >
                      {/* Artist avatar */}
                      <div style={{
                        width: 36, height: 36, borderRadius: '50%', flexShrink: 0, overflow: 'hidden',
                        background: result.artworkUrl ? `url(${result.artworkUrl}) center/cover` : 'rgba(255,45,85,0.1)',
                        border: '1px solid rgba(255,255,255,0.08)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                      }}>
                        {!result.artworkUrl && (
                          <span style={{ fontFamily: 'var(--font-heading)', fontSize: '14px', color: 'rgba(255,45,85,0.6)' }}>
                            {result.name[0]?.toUpperCase()}
                          </span>
                        )}
                      </div>

                      <span style={{
                        flex: 1, fontFamily: 'var(--font-body)', fontSize: '14px', fontWeight: 400,
                        color: isSelected ? '#FF6B8A' : '#ffffff',
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      }}>
                        {result.name}
                      </span>

                      {isSelected ? (
                        <Check style={{ width: 14, height: 14, color: '#FF2D55', flexShrink: 0 }} />
                      ) : (
                        <Plus style={{ width: 14, height: 14, color: 'rgba(255,255,255,0.3)', flexShrink: 0 }} />
                      )}
                    </motion.button>
                  );
                })}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Selected artists chips (if any not in topArtists) */}
        {selected.filter(a => !artists.includes(a)).length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            style={{ marginBottom: 16 }}
          >
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontWeight: 300, marginBottom: 8, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
              added
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {selected.filter(a => !artists.includes(a)).map(artist => (
                <motion.button
                  key={artist}
                  onClick={() => toggleArtist(artist)}
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  whileTap={{ scale: 0.95 }}
                  style={{
                    padding: '8px 14px', borderRadius: '24px', cursor: 'pointer',
                    background: 'rgba(255, 45, 85, 0.15)', border: '1px solid rgba(255, 45, 85, 0.5)',
                    display: 'flex', alignItems: 'center', gap: 6,
                    fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 500, color: '#FF6B8A',
                  }}
                >
                  {artist}
                  <X style={{ width: 10, height: 10, opacity: 0.6 }} />
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}

        {/* Suggested artist pills */}
        {artists.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.3 }}
          >
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontWeight: 300, marginBottom: 12, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
              suggested
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {artists.map((artist, i) => {
                const isSelected = selected.includes(artist);
                return (
                  <motion.button
                    key={artist}
                    onClick={() => toggleArtist(artist)}
                    initial={{ opacity: 0, scale: 0.85 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.05, duration: 0.25, ease: 'backOut' }}
                    whileTap={{ scale: 0.95 }}
                    style={{
                      padding: '10px 18px', borderRadius: '24px', cursor: 'pointer',
                      background: isSelected ? 'rgba(255, 45, 85, 0.15)' : 'rgba(255, 255, 255, 0.04)',
                      border: isSelected ? '1px solid rgba(255, 45, 85, 0.5)' : '1px solid rgba(255, 255, 255, 0.1)',
                      boxShadow: isSelected ? '0 0 16px rgba(255, 45, 85, 0.12)' : 'none',
                      fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 500,
                      color: isSelected ? '#FF6B8A' : 'rgba(255,255,255,0.7)',
                      transition: 'background 0.15s, border-color 0.15s, color 0.15s',
                    }}
                  >
                    {artist}
                  </motion.button>
                );
              })}
            </div>
          </motion.div>
        )}

        {artists.length === 0 && !query && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', paddingTop: 40 }}>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '14px', color: 'rgba(255,255,255,0.3)', textAlign: 'center', fontWeight: 300 }}>
              search for artists above or skip to continue
            </p>
          </div>
        )}
      </div>

      {/* Bottom CTA */}
      <div style={{
        position: 'fixed', bottom: 0, left: 0, right: 0, zIndex: 20,
        padding: '16px 24px calc(24px + var(--safe-area-bottom))',
        background: 'linear-gradient(to top, #0a0a0a 60%, transparent)',
        display: 'flex', flexDirection: 'column', gap: 10,
      }}>
        <motion.button
          onClick={() => onComplete(selected)}
          disabled={selected.length === 0}
          style={{
            width: '100%', padding: '16px', borderRadius: '14px',
            background: selected.length > 0 ? '#FF2D55' : 'rgba(255, 45, 85, 0.2)',
            border: 'none', cursor: selected.length > 0 ? 'pointer' : 'default',
            fontFamily: 'var(--font-body)', fontSize: '15px', fontWeight: 600,
            color: selected.length > 0 ? '#ffffff' : 'rgba(255,255,255,0.3)',
            letterSpacing: '0.04em',
            transition: 'background 0.2s, color 0.2s',
          }}
          whileHover={selected.length > 0 ? { scale: 1.02 } : {}}
          whileTap={selected.length > 0 ? { scale: 0.98 } : {}}
        >
          {selected.length > 0
            ? `start with ${selected.length} artist${selected.length > 1 ? 's' : ''}`
            : 'select an artist to start'}
        </motion.button>

        <button
          onClick={() => onComplete([])}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            fontFamily: 'var(--font-body)', fontSize: '13px', fontWeight: 300,
            color: 'rgba(255,255,255,0.3)', textAlign: 'center', padding: '4px',
          }}
        >
          skip — any artist
        </button>
      </div>
    </div>
  );
}
