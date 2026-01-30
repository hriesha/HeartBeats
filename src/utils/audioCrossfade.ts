/**
 * Audio Crossfade Algorithm for HeartBeats
 *
 * Provides smooth transitions between tracks with configurable crossfade duration.
 * Uses Web Audio API for precise volume control.
 *
 * Behavior:
 * - Auto-advance (song ends naturally) → Smooth crossfade
 * - Manual skip (user clicks next) → Instant switch, no fade
 */

export interface CrossfadeOptions {
  crossfadeDuration?: number; // Duration in ms (default: 5000)
  onTrackStart?: (url: string) => void;
  onTrackEnd?: () => void;
  onCrossfadeProgress?: (progress: number) => void;
}

export interface AudioTrack {
  audio: HTMLAudioElement;
  gainNode: GainNode;
  source: MediaElementAudioSourceNode;
}

export class AudioCrossfader {
  private audioContext: AudioContext | null = null;
  private currentTrack: AudioTrack | null = null;
  private nextTrack: AudioTrack | null = null;
  private crossfadeDuration: number;
  private isTransitioning: boolean = false;
  private crossfadeTimer: number | null = null;
  private trackEndCheckInterval: number | null = null;
  private masterVolume: number = 1;

  // Callbacks
  private onTrackStart?: (url: string) => void;
  private onTrackEnd?: () => void;
  private onCrossfadeProgress?: (progress: number) => void;

  constructor(options: CrossfadeOptions = {}) {
    this.crossfadeDuration = options.crossfadeDuration ?? 5000;
    this.onTrackStart = options.onTrackStart;
    this.onTrackEnd = options.onTrackEnd;
    this.onCrossfadeProgress = options.onCrossfadeProgress;
  }

  /**
   * Initialize the AudioContext (must be called after user interaction)
   */
  private async initAudioContext(): Promise<void> {
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
    }
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  /**
   * Create an audio track with gain control
   */
  private async createAudioTrack(url: string): Promise<AudioTrack> {
    await this.initAudioContext();

    const audio = new Audio();
    audio.crossOrigin = 'anonymous';
    audio.src = url;

    const source = this.audioContext!.createMediaElementSource(audio);
    const gainNode = this.audioContext!.createGain();

    source.connect(gainNode);
    gainNode.connect(this.audioContext!.destination);

    return { audio, gainNode, source };
  }

  /**
   * Play a track (stops any current playback)
   */
  async play(url: string): Promise<void> {
    // Stop any current playback
    this.stopCurrentTrack();

    try {
      this.currentTrack = await this.createAudioTrack(url);
      this.currentTrack.gainNode.gain.value = this.masterVolume;

      await this.currentTrack.audio.play();
      this.onTrackStart?.(url);

      // Start monitoring for track end (for auto-crossfade)
      this.startTrackEndMonitoring();
    } catch (error) {
      console.error('Error playing track:', error);
      throw error;
    }
  }

  /**
   * Crossfade to a new track (smooth transition)
   * Used for auto-advance when song ends naturally
   */
  async crossfadeTo(url: string): Promise<void> {
    if (!this.currentTrack || this.isTransitioning) {
      // No current track or already transitioning - just play directly
      return this.play(url);
    }

    this.isTransitioning = true;
    this.stopTrackEndMonitoring();

    try {
      // Create and prepare next track
      this.nextTrack = await this.createAudioTrack(url);
      this.nextTrack.gainNode.gain.value = 0; // Start silent

      // Start playing next track
      await this.nextTrack.audio.play();

      // Perform the crossfade
      await this.performCrossfade();

      // Cleanup old track and swap references
      this.cleanupTrack(this.currentTrack);
      this.currentTrack = this.nextTrack;
      this.nextTrack = null;

      this.onTrackStart?.(url);
      this.startTrackEndMonitoring();
    } catch (error) {
      console.error('Error during crossfade:', error);
      // Fallback to direct play
      this.isTransitioning = false;
      return this.play(url);
    } finally {
      this.isTransitioning = false;
    }
  }

  /**
   * Skip to a new track instantly (no crossfade)
   * Used for manual skip when user clicks next
   */
  async skipTo(url: string): Promise<void> {
    // Cancel any ongoing crossfade
    this.cancelCrossfade();

    // Stop current track immediately
    this.stopCurrentTrack();

    // Play new track
    return this.play(url);
  }

  /**
   * Perform the actual crossfade animation
   */
  private performCrossfade(): Promise<void> {
    return new Promise((resolve) => {
      const startTime = Date.now();
      const duration = this.crossfadeDuration;

      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Exponential curves for natural-sounding fade
        const fadeOutVolume = Math.pow(1 - progress, 2) * this.masterVolume;
        const fadeInVolume = Math.pow(progress, 2) * this.masterVolume;

        if (this.currentTrack) {
          this.currentTrack.gainNode.gain.value = fadeOutVolume;
        }
        if (this.nextTrack) {
          this.nextTrack.gainNode.gain.value = fadeInVolume;
        }

        this.onCrossfadeProgress?.(progress);

        if (progress < 1) {
          this.crossfadeTimer = requestAnimationFrame(animate);
        } else {
          this.crossfadeTimer = null;
          resolve();
        }
      };

      this.crossfadeTimer = requestAnimationFrame(animate);
    });
  }

  /**
   * Cancel an ongoing crossfade
   */
  private cancelCrossfade(): void {
    if (this.crossfadeTimer !== null) {
      cancelAnimationFrame(this.crossfadeTimer);
      this.crossfadeTimer = null;
    }

    // Clean up next track if it was being prepared
    if (this.nextTrack) {
      this.cleanupTrack(this.nextTrack);
      this.nextTrack = null;
    }

    this.isTransitioning = false;
  }

  /**
   * Monitor current track for when it's about to end
   */
  private startTrackEndMonitoring(): void {
    this.stopTrackEndMonitoring();

    if (!this.currentTrack) return;

    // Check every 100ms if we should trigger the onTrackEnd callback
    this.trackEndCheckInterval = window.setInterval(() => {
      if (!this.currentTrack) {
        this.stopTrackEndMonitoring();
        return;
      }

      const audio = this.currentTrack.audio;
      const timeRemaining = audio.duration - audio.currentTime;

      // Trigger onTrackEnd when track is about to finish
      // (accounting for crossfade duration)
      if (timeRemaining <= 0.1 && !this.isTransitioning) {
        this.stopTrackEndMonitoring();
        this.onTrackEnd?.();
      }
    }, 100);
  }

  /**
   * Stop monitoring for track end
   */
  private stopTrackEndMonitoring(): void {
    if (this.trackEndCheckInterval !== null) {
      clearInterval(this.trackEndCheckInterval);
      this.trackEndCheckInterval = null;
    }
  }

  /**
   * Stop and cleanup current track
   */
  private stopCurrentTrack(): void {
    this.stopTrackEndMonitoring();
    this.cancelCrossfade();

    if (this.currentTrack) {
      this.cleanupTrack(this.currentTrack);
      this.currentTrack = null;
    }
  }

  /**
   * Cleanup an audio track and release resources
   */
  private cleanupTrack(track: AudioTrack): void {
    try {
      track.audio.pause();
      track.audio.src = '';
      track.source.disconnect();
      track.gainNode.disconnect();
    } catch (error) {
      // Ignore cleanup errors
    }
  }

  /**
   * Pause current playback
   */
  pause(): void {
    if (this.currentTrack) {
      this.currentTrack.audio.pause();
    }
    if (this.nextTrack) {
      this.nextTrack.audio.pause();
    }
  }

  /**
   * Resume playback
   */
  async resume(): Promise<void> {
    await this.initAudioContext();
    if (this.currentTrack) {
      await this.currentTrack.audio.play();
    }
  }

  /**
   * Set master volume (0-1)
   */
  setVolume(volume: number): void {
    this.masterVolume = Math.max(0, Math.min(1, volume));

    if (this.currentTrack && !this.isTransitioning) {
      this.currentTrack.gainNode.gain.value = this.masterVolume;
    }
  }

  /**
   * Get master volume
   */
  getVolume(): number {
    return this.masterVolume;
  }

  /**
   * Set crossfade duration in milliseconds
   */
  setCrossfadeDuration(ms: number): void {
    this.crossfadeDuration = Math.max(0, ms);
  }

  /**
   * Get crossfade duration in milliseconds
   */
  getCrossfadeDuration(): number {
    return this.crossfadeDuration;
  }

  /**
   * Check if currently playing
   */
  isPlaying(): boolean {
    return this.currentTrack !== null && !this.currentTrack.audio.paused;
  }

  /**
   * Check if currently transitioning between tracks
   */
  isCrossfading(): boolean {
    return this.isTransitioning;
  }

  /**
   * Get current playback time
   */
  getCurrentTime(): number {
    return this.currentTrack?.audio.currentTime ?? 0;
  }

  /**
   * Get current track duration
   */
  getDuration(): number {
    return this.currentTrack?.audio.duration ?? 0;
  }

  /**
   * Seek to a specific time
   */
  seek(time: number): void {
    if (this.currentTrack) {
      this.currentTrack.audio.currentTime = Math.max(0, Math.min(time, this.getDuration()));
    }
  }

  /**
   * Cleanup all resources
   */
  cleanup(): void {
    this.stopCurrentTrack();

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

// Export a singleton instance for simple usage
let defaultInstance: AudioCrossfader | null = null;

export function getAudioCrossfader(options?: CrossfadeOptions): AudioCrossfader {
  if (!defaultInstance) {
    defaultInstance = new AudioCrossfader(options);
  }
  return defaultInstance;
}

export function resetAudioCrossfader(): void {
  if (defaultInstance) {
    defaultInstance.cleanup();
    defaultInstance = null;
  }
}
