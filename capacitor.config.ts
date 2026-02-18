import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.heartbeats.app',
  appName: 'HeartBeats',
  webDir: 'dist',
  server: {
    // For dev: uncomment and set to your Mac's local IP for live reload
    // url: 'http://192.168.1.100:5173',
    // cleartext: true,
    allowNavigation: [
      'js-cdn.music.apple.com',
      'api.music.apple.com',
      '*.apple.com',
    ],
  },
  ios: {
    contentInset: 'always',
    allowsLinkPreview: false,
    preferredContentMode: 'mobile',
  },
  plugins: {
    StatusBar: {
      style: 'DARK',
      backgroundColor: '#0a0a0a',
      overlaysWebView: true,
    },
    Keyboard: {
      resize: 'body',
      style: 'DARK',
    },
    SplashScreen: {
      launchShowDuration: 1500,
      backgroundColor: '#0a0a0a',
      showSpinner: false,
      launchAutoHide: true,
    },
  },
};

export default config;
