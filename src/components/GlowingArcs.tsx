/**
 * Ambient background — soft, light pink orbs that drift slowly.
 */
export function GlowingArcs() {
  const pink = (a: number) => `rgba(255, 45, 85, ${a})`;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        overflow: 'hidden',
        pointerEvents: 'none',
        zIndex: 0,
      }}
    >
      {/* Orb 1 — top right */}
      <div
        style={{
          position: 'absolute', top: '2%', right: '5%',
          width: 200, height: 200, borderRadius: '50%',
          background: `radial-gradient(circle at 45% 45%, ${pink(0.07)}, ${pink(0.025)} 50%, transparent 72%)`,
          border: `1px solid ${pink(0.05)}`,
          boxShadow: `0 0 50px ${pink(0.04)}`,
          animation: 'orbDrift1 30s ease-in-out infinite',
        }}
      />

      {/* Orb 2 — left */}
      <div
        style={{
          position: 'absolute', top: '25%', left: '2%',
          width: 160, height: 160, borderRadius: '50%',
          background: `radial-gradient(circle at 45% 45%, ${pink(0.06)}, ${pink(0.02)} 50%, transparent 72%)`,
          border: `1px solid ${pink(0.04)}`,
          boxShadow: `0 0 40px ${pink(0.03)}`,
          animation: 'orbDrift2 35s ease-in-out infinite',
        }}
      />

      {/* Orb 3 — center right */}
      <div
        style={{
          position: 'absolute', top: '42%', right: '10%',
          width: 130, height: 130, borderRadius: '50%',
          background: `radial-gradient(circle at 45% 45%, ${pink(0.08)}, ${pink(0.03)} 50%, transparent 72%)`,
          border: `1px solid ${pink(0.06)}`,
          boxShadow: `0 0 40px ${pink(0.05)}`,
          animation: 'orbDrift3 28s ease-in-out infinite',
        }}
      />

      {/* Orb 4 — bottom left */}
      <div
        style={{
          position: 'absolute', bottom: '10%', left: '8%',
          width: 180, height: 180, borderRadius: '50%',
          background: `radial-gradient(circle at 45% 45%, ${pink(0.06)}, ${pink(0.02)} 50%, transparent 72%)`,
          border: `1px solid ${pink(0.045)}`,
          boxShadow: `0 0 45px ${pink(0.035)}`,
          animation: 'orbDrift4 32s ease-in-out infinite',
        }}
      />

      {/* Orb 5 — bottom right */}
      <div
        style={{
          position: 'absolute', bottom: '20%', right: '22%',
          width: 110, height: 110, borderRadius: '50%',
          background: `radial-gradient(circle at 45% 45%, ${pink(0.07)}, ${pink(0.025)} 50%, transparent 72%)`,
          border: `1px solid ${pink(0.05)}`,
          boxShadow: `0 0 35px ${pink(0.04)}`,
          animation: 'orbDrift5 36s ease-in-out infinite',
        }}
      />

      {/* Orb 6 — top left */}
      <div
        style={{
          position: 'absolute', top: '12%', left: '28%',
          width: 90, height: 90, borderRadius: '50%',
          background: `radial-gradient(circle at 45% 45%, ${pink(0.06)}, ${pink(0.02)} 50%, transparent 72%)`,
          border: `1px solid ${pink(0.04)}`,
          boxShadow: `0 0 25px ${pink(0.03)}`,
          animation: 'orbDrift6 34s ease-in-out infinite',
        }}
      />

      {/* Orb 7 — mid left */}
      <div
        style={{
          position: 'absolute', top: '58%', left: '20%',
          width: 100, height: 100, borderRadius: '50%',
          background: `radial-gradient(circle at 45% 45%, ${pink(0.05)}, ${pink(0.015)} 50%, transparent 72%)`,
          border: `1px solid ${pink(0.035)}`,
          boxShadow: `0 0 30px ${pink(0.025)}`,
          animation: 'orbDrift7 38s ease-in-out infinite',
        }}
      />

      <style>{`
        @keyframes orbDrift1 {
          0%, 100% { transform: translate(0, 0); }
          25% { transform: translate(-18px, 22px); }
          50% { transform: translate(12px, -14px); }
          75% { transform: translate(-8px, -18px); }
        }
        @keyframes orbDrift2 {
          0%, 100% { transform: translate(0, 0); }
          33% { transform: translate(20px, -15px); }
          66% { transform: translate(-12px, 18px); }
        }
        @keyframes orbDrift3 {
          0%, 100% { transform: translate(0, 0); }
          25% { transform: translate(15px, 10px); }
          50% { transform: translate(-10px, -18px); }
          75% { transform: translate(-15px, 8px); }
        }
        @keyframes orbDrift4 {
          0%, 100% { transform: translate(0, 0); }
          33% { transform: translate(-16px, -12px); }
          66% { transform: translate(14px, 16px); }
        }
        @keyframes orbDrift5 {
          0%, 100% { transform: translate(0, 0); }
          50% { transform: translate(12px, -16px); }
        }
        @keyframes orbDrift6 {
          0%, 100% { transform: translate(0, 0); }
          33% { transform: translate(10px, 14px); }
          66% { transform: translate(-14px, -8px); }
        }
        @keyframes orbDrift7 {
          0%, 100% { transform: translate(0, 0); }
          50% { transform: translate(-10px, -12px); }
        }
      `}</style>
    </div>
  );
}
