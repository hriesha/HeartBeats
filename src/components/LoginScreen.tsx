import { LoginHeader } from './LoginHeader';
import { LoginCard } from './LoginCard';

interface LoginScreenProps {
  onLogin: () => void;
}

export function LoginScreen({ onLogin }: LoginScreenProps) {
  return (
    <div className="relative w-full h-full overflow-auto flex items-center justify-center px-6 py-8" style={{ fontFamily: 'Poppins, sans-serif' }}>
      {/* Background with gradient overlay */}
      <div 
        className="absolute inset-0 z-0"
        style={{
          background: `linear-gradient(180deg, #003049 0%, #D62828 50%, #003049 100%)`
        }}
      />
      
      {/* Content */}
      <div className="relative z-10 w-full max-w-sm flex flex-col items-center">
        <LoginHeader />
        <LoginCard onLogin={onLogin} />
      </div>
    </div>
  );
}