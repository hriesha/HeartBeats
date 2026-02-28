import { useState, useEffect, useRef, useCallback } from 'react';
import { Capacitor } from '@capacitor/core';
import { HeartRate } from '../plugins/HeartRate';

export type HeartRateStatus = 'idle' | 'requesting' | 'polling' | 'unsupported' | 'error';

export interface HeartRateState {
  bpm: number | null;
  status: HeartRateStatus;
  isSupported: boolean;
  lastUpdated: Date | null;
  requestPermission: () => Promise<boolean>;
  startPolling: () => void;
  stopPolling: () => void;
}

const POLL_INTERVAL_MS = 5000;

export function useHeartRate(): HeartRateState {
  const [bpm, setBpm] = useState<number | null>(null);
  const [status, setStatus] = useState<HeartRateStatus>('idle');
  const [isSupported, setIsSupported] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!Capacitor.isNativePlatform()) {
      setStatus('unsupported');
      return;
    }
    HeartRate.isAvailable().then(({ available }) => {
      setIsSupported(available);
      if (!available) setStatus('unsupported');
    });
  }, []);

  const fetchLatest = useCallback(async () => {
    try {
      const result = await HeartRate.getLatestHeartRate();
      if (result.bpm !== null) {
        setBpm(result.bpm);
        setLastUpdated(new Date());
      }
    } catch {
      // silently ignore individual poll failures
    }
  }, []);

  const requestPermission = useCallback(async (): Promise<boolean> => {
    if (!Capacitor.isNativePlatform()) return false;
    setStatus('requesting');
    try {
      const { authorized } = await HeartRate.requestAuthorization();
      if (authorized) {
        setStatus('polling');
        await fetchLatest();
      } else {
        setStatus('error');
      }
      return authorized;
    } catch {
      setStatus('error');
      return false;
    }
  }, [fetchLatest]);

  const startPolling = useCallback(() => {
    if (pollRef.current) return;
    setStatus('polling');
    fetchLatest();
    pollRef.current = setInterval(fetchLatest, POLL_INTERVAL_MS);
  }, [fetchLatest]);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    setStatus('idle');
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  return { bpm, status, isSupported, lastUpdated, requestPermission, startPolling, stopPolling };
}
