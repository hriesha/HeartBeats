import { motion } from 'motion/react';
import { ChevronLeft, Play, Heart } from 'lucide-react';
import { VibeType } from '../App';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Area, AreaChart } from 'recharts';

interface VibeDetailProps {
  vibe: VibeType;
  onBack: () => void;
}

// Sample data for the BPM graph
const bpmData = [
  { time: '0:00', bpm: 120 },
  { time: '0:30', bpm: 125 },
  { time: '1:00', bpm: 130 },
  { time: '1:30', bpm: 128 },
  { time: '2:00', bpm: 135 },
  { time: '2:30', bpm: 140 },
  { time: '3:00', bpm: 138 },
  { time: '3:30', bpm: 142 },
  { time: '4:00', bpm: 145 },
  { time: '4:30', bpm: 140 },
  { time: '5:00', bpm: 135 },
];

// Sample playlist data
const playlistSongs = [
  { id: 1, title: 'Summer Vibes', artist: 'The Chill Makers', duration: '3:45', bpm: 128 },
  { id: 2, title: 'Midnight Drive', artist: 'Lo-Fi Dreams', duration: '4:12', bpm: 125 },
  { id: 3, title: 'Ocean Breeze', artist: 'Calm Waves', duration: '3:28', bpm: 130 },
  { id: 4, title: 'Golden Hour', artist: 'Sunset Collective', duration: '4:05', bpm: 132 },
  { id: 5, title: 'Peaceful Mind', artist: 'Zen Masters', duration: '3:52', bpm: 126 },
];

export function VibeDetail({ vibe, onBack }: VibeDetailProps) {
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
      <div className="relative z-10 w-full h-full">
        {/* Back Button */}
        <button
          onClick={onBack}
          className="absolute top-4 left-4 p-2 rounded-full transition-all z-20"
          style={{
            backgroundColor: 'rgba(0, 48, 73, 0.8)',
            color: '#FCBF49'
          }}
        >
          <ChevronLeft className="w-6 h-6" />
        </button>

        {/* Vibe Header */}
        <div className="pt-16 pb-6 px-6">
          <motion.div 
            className="rounded-2xl p-6 flex items-center justify-between"
            style={{
              backgroundColor: vibe.color,
              boxShadow: `0 8px 24px ${vibe.color}80, inset 0 2px 8px rgba(255, 255, 255, 0.3)`,
            }}
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div>
              <h1 
                style={{
                  fontFamily: 'Poppins, sans-serif',
                  fontSize: '24px',
                  fontWeight: 700,
                  color: '#03071E',
                  marginBottom: '4px'
                }}
              >
                {vibe.name}
              </h1>
              <p 
                style={{
                  fontFamily: 'Poppins, sans-serif',
                  fontSize: '13px',
                  fontWeight: 600,
                  color: '#370617',
                }}
              >
                {vibe.tags.join(' • ')}
              </p>
            </div>
            <Heart className="w-6 h-6" style={{ color: '#03071E' }} />
          </motion.div>
        </div>

        {/* BPM Graph Section */}
        <div className="px-6 mb-6">
          <h2 
            className="mb-3"
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '18px',
              fontWeight: 700,
              color: '#EAE2B7',
            }}
          >
            Your Heart Rate
          </h2>
          <div 
            className="rounded-2xl p-4"
            style={{
              backgroundColor: 'rgba(0, 48, 73, 0.6)',
              border: '1px solid rgba(252, 191, 73, 0.3)',
            }}
          >
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={bpmData}>
                <defs>
                  <linearGradient id="colorBpm" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#F77F00" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#F77F00" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(252, 191, 73, 0.1)" />
                <XAxis 
                  dataKey="time" 
                  stroke="#EAE2B7" 
                  style={{ fontSize: '11px', fontFamily: 'Poppins, sans-serif' }}
                />
                <YAxis 
                  stroke="#EAE2B7"
                  style={{ fontSize: '11px', fontFamily: 'Poppins, sans-serif' }}
                  domain={[100, 160]}
                />
                <Area 
                  type="monotone" 
                  dataKey="bpm" 
                  stroke="#FCBF49" 
                  strokeWidth={2}
                  fill="url(#colorBpm)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Curated Playlist Section */}
        <div className="px-6 pb-6">
          <h2 
            className="mb-3"
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '18px',
              fontWeight: 700,
              color: '#EAE2B7',
            }}
          >
            Curated Playlist
          </h2>
          <div className="space-y-3">
            {playlistSongs.map((song, index) => (
              <motion.div
                key={song.id}
                className="rounded-xl p-4 flex items-center gap-3"
                style={{
                  backgroundColor: 'rgba(0, 48, 73, 0.6)',
                  border: '1px solid rgba(252, 191, 73, 0.2)',
                }}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                {/* Album Art Placeholder */}
                <div 
                  className="rounded-lg flex-shrink-0"
                  style={{
                    width: '50px',
                    height: '50px',
                    background: `linear-gradient(135deg, ${vibe.color}80 0%, ${vibe.color}40 100%)`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <span style={{ 
                    fontFamily: 'Poppins, sans-serif', 
                    fontSize: '12px', 
                    fontWeight: 700,
                    color: '#03071E'
                  }}>
                    {song.bpm}
                  </span>
                </div>
                
                {/* Song Info */}
                <div className="flex-1 min-w-0">
                  <h3 
                    className="truncate"
                    style={{
                      fontFamily: 'Poppins, sans-serif',
                      fontSize: '14px',
                      fontWeight: 600,
                      color: '#EAE2B7',
                      marginBottom: '2px'
                    }}
                  >
                    {song.title}
                  </h3>
                  <p 
                    className="truncate"
                    style={{
                      fontFamily: 'Poppins, sans-serif',
                      fontSize: '12px',
                      color: '#EAE2B7',
                      opacity: 0.7
                    }}
                  >
                    {song.artist}
                  </p>
                </div>

                {/* Duration */}
                <span 
                  style={{
                    fontFamily: 'Poppins, sans-serif',
                    fontSize: '12px',
                    color: '#EAE2B7',
                    marginRight: '8px',
                    opacity: 0.7
                  }}
                >
                  {song.duration}
                </span>

                {/* Play Button */}
                <button
                  className="rounded-full p-2 flex-shrink-0 transition-all"
                  style={{
                    background: 'linear-gradient(135deg, #FCBF49 0%, #F77F00 100%)',
                  }}
                >
                  <Play className="w-4 h-4 text-white fill-white" />
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}