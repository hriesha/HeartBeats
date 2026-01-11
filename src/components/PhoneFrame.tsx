import { ReactNode } from 'react';

interface PhoneFrameProps {
  children: ReactNode;
}

export function PhoneFrame({ children }: PhoneFrameProps) {
  const cursorSvg = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 32 32'%3E%3Cdefs%3E%3ClinearGradient id='grad' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%23FFE5B4;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%23FF7F50;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='16' cy='16' r='8' fill='url(%23grad)' opacity='0.8'%3E%3Canimate attributeName='r' values='8;10;8' dur='1.5s' repeatCount='indefinite'/%3E%3C/circle%3E%3Ccircle cx='16' cy='16' r='4' fill='white'/%3E%3C/svg%3E`;

  return (
    <div 
      className="relative bg-black overflow-hidden shadow-2xl"
      style={{
        width: '390px',
        height: '844px',
        borderRadius: '48px',
        border: '12px solid #1a1a1a',
        boxShadow: '0 20px 60px rgba(0, 0, 0, 0.8), inset 0 0 0 2px #333',
        cursor: `url("${cursorSvg}") 16 16, auto`
      }}
    >
      {/* Notch */}
      <div 
        className="absolute top-0 left-1/2 -translate-x-1/2 z-50 bg-black"
        style={{
          width: '120px',
          height: '30px',
          borderBottomLeftRadius: '20px',
          borderBottomRightRadius: '20px'
        }}
      />
      
      {/* Screen Content */}
      <div className="w-full h-full overflow-hidden">
        {children}
      </div>
    </div>
  );
}