import { registerPlugin } from '@capacitor/core';
import type { PluginListenerHandle } from '@capacitor/core';

export interface HealthKitPlugin {
  requestPermission(): Promise<{ granted: boolean }>;
  startHeartRateMonitoring(): Promise<void>;
  stopHeartRateMonitoring(): Promise<void>;
  addListener(
    eventName: 'heartRateUpdate',
    listenerFunc: (data: { bpm: number }) => void
  ): Promise<PluginListenerHandle>;
  removeAllListeners(): Promise<void>;
}

const HealthKit = registerPlugin<HealthKitPlugin>('HealthKit');
export default HealthKit;
