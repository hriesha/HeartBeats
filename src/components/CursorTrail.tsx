import { useEffect, useRef } from 'react';

/**
 * Light, flowy pink glow that follows the cursor.
 * Short trail, soft and subtle — not a long trailing stroke.
 */
export function CursorTrail() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -200, y: -200 });
  const smoothRef = useRef({ x: -200, y: -200 });
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    window.addEventListener('mousemove', handleMouseMove);

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Smooth follow — lerp toward mouse position for flowy feel
      const lerp = 0.12;
      smoothRef.current.x += (mouseRef.current.x - smoothRef.current.x) * lerp;
      smoothRef.current.y += (mouseRef.current.y - smoothRef.current.y) * lerp;

      const sx = smoothRef.current.x;
      const sy = smoothRef.current.y;

      // Soft outer glow
      const outerGrad = ctx.createRadialGradient(sx, sy, 0, sx, sy, 60);
      outerGrad.addColorStop(0, 'rgba(255, 45, 85, 0.06)');
      outerGrad.addColorStop(0.5, 'rgba(255, 45, 85, 0.02)');
      outerGrad.addColorStop(1, 'rgba(255, 45, 85, 0)');
      ctx.beginPath();
      ctx.arc(sx, sy, 60, 0, Math.PI * 2);
      ctx.fillStyle = outerGrad;
      ctx.fill();

      // Inner glow — slightly brighter core
      const innerGrad = ctx.createRadialGradient(sx, sy, 0, sx, sy, 20);
      innerGrad.addColorStop(0, 'rgba(255, 45, 85, 0.12)');
      innerGrad.addColorStop(0.6, 'rgba(255, 45, 85, 0.04)');
      innerGrad.addColorStop(1, 'rgba(255, 45, 85, 0)');
      ctx.beginPath();
      ctx.arc(sx, sy, 20, 0, Math.PI * 2);
      ctx.fillStyle = innerGrad;
      ctx.fill();

      // Tiny bright dot at center
      ctx.beginPath();
      ctx.arc(sx, sy, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 45, 85, 0.2)';
      ctx.fill();

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 10,
      }}
    />
  );
}
