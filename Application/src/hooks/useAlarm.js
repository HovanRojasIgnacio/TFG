import { useState, useRef } from 'react';
import { ALARM_MIN_INTERVAL_MS } from '../config';

/** Alarma sonora sintetizada con Web Audio API; se activa con un gesto del usuario. */
export function useAlarm() {
  const [soundEnabled, setSoundEnabled] = useState(false);
  const audioCtxRef = useRef(null);
  const lastBeepRef = useRef(0);

  // Inicializa o reanuda AudioContext tras interacción
  const initAudio = () => {
    if (!audioCtxRef.current) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioCtxRef.current = new AudioCtx();
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }
  };

  const beep = () => {
    if (!audioCtxRef.current) return;
    const now = Date.now();
    if (now - lastBeepRef.current < ALARM_MIN_INTERVAL_MS) return;
    lastBeepRef.current = now;
    try {
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.25);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      osc.stop(ctx.currentTime + 0.25);
    } catch {
      // Ignorar fallos puntuales de buffer de audio
    }
  };

  const toggleSound = (e) => {
    const enabled = e.target.checked;
    if (enabled) initAudio();
    setSoundEnabled(enabled);
  };

  return { soundEnabled, toggleSound, beep };
}
