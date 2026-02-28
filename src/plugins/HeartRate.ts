import { registerPlugin } from '@capacitor/core';

export interface HeartRatePlugin {
  isAvailable(): Promise<{ available: boolean }>;
  requestAuthorization(): Promise<{ authorized: boolean }>;
  getLatestHeartRate(): Promise<{ bpm: number | null; timestamp: number | null }>;
}

const HeartRate = registerPlugin<HeartRatePlugin>('HeartRate', {
  web: {
    isAvailable: async () => ({ available: false }),
    requestAuthorization: async () => ({ authorized: false }),
    getLatestHeartRate: async () => ({ bpm: null, timestamp: null }),
  },
});

export { HeartRate };
