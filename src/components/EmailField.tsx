import { Mail } from 'lucide-react';

interface EmailFieldProps {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export function EmailField({ value, onChange }: EmailFieldProps) {
  return (
    <div className="space-y-2">
      <label 
        htmlFor="email"
        style={{
          display: 'block',
          fontFamily: 'Poppins, sans-serif',
          fontSize: '14px',
          fontWeight: 600,
          color: '#FCBF49',
          marginBottom: '8px'
        }}
      >
        email
      </label>
      <div className="relative">
        <div className="absolute left-4 top-1/2 -translate-y-1/2 pointer-events-none">
          <Mail className="w-5 h-5" style={{ color: '#F77F00' }} />
        </div>
        <input
          id="email"
          type="email"
          value={value}
          onChange={onChange}
          placeholder="enter your email"
          required
          className="w-full pl-12 pr-4 py-3 rounded-xl outline-none transition-all focus:ring-2"
          style={{
            fontFamily: 'Poppins, sans-serif',
            backgroundColor: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(252, 191, 73, 0.3)',
            color: '#EAE2B7',
            fontSize: '16px',
            boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.2)'
          }}
          onFocus={(e) => {
            e.currentTarget.style.borderColor = '#FCBF49';
            e.currentTarget.style.boxShadow = '0 0 0 3px rgba(252, 191, 73, 0.15), inset 0 2px 4px rgba(0, 0, 0, 0.2)';
          }}
          onBlur={(e) => {
            e.currentTarget.style.borderColor = 'rgba(252, 191, 73, 0.3)';
            e.currentTarget.style.boxShadow = 'inset 0 2px 4px rgba(0, 0, 0, 0.2)';
          }}
        />
      </div>
    </div>
  );
}