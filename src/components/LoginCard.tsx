import { useState } from 'react';
import { EmailField } from './EmailField';
import { PasswordField } from './PasswordField';
import { SignInButton } from './SignInButton';

interface LoginCardProps {
  onLogin: () => void;
}

export function LoginCard({ onLogin }: LoginCardProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSignIn = (e: React.FormEvent) => {
    e.preventDefault();
    onLogin();
  };

  return (
    <div 
      className="w-full p-8"
      style={{
        backgroundColor: 'rgba(0, 48, 73, 0.8)',
        borderRadius: '24px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), 0 2px 8px rgba(252, 191, 73, 0.2)',
        border: '1px solid rgba(252, 191, 73, 0.2)'
      }}
    >
      <form onSubmit={handleSignIn} className="space-y-6">
        {/* Email Field */}
        <EmailField 
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        {/* Password Field */}
        <PasswordField 
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {/* Forgot Password Link */}
        <div className="flex justify-end">
          <button 
            type="button"
            className="transition-colors"
            style={{
              fontFamily: 'Poppins, sans-serif',
              fontSize: '14px',
              color: '#FCBF49',
              fontWeight: 500
            }}
            onMouseEnter={(e) => e.currentTarget.style.color = '#F77F00'}
            onMouseLeave={(e) => e.currentTarget.style.color = '#FCBF49'}
          >
            forgot password?
          </button>
        </div>

        {/* Sign In Button */}
        <SignInButton />

        {/* Sign Up Link */}
        <div className="text-center pt-2">
          <span style={{ fontFamily: 'Poppins, sans-serif', color: '#EAE2B7', fontSize: '14px' }}>
            don't have an account?{' '}
          </span>
          <button
            type="button"
            className="transition-opacity"
            style={{
              fontFamily: 'Poppins, sans-serif',
              color: '#FCBF49',
              fontSize: '14px',
              fontWeight: 600
            }}
            onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
            onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
          >
            sign up
          </button>
        </div>
      </form>
    </div>
  );
}