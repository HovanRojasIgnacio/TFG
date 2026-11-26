import { useState, useRef, useEffect } from 'react';
import { ALARM_MIN_INTERVAL_MS } from '../config';

// Un pitido corto en `ctx`
function playBeep(ctx) {
  // El navegador puede suspender el audio por su cuenta: se reanuda para que la alarma no quede muda
  if (ctx.state !== 'running') ctx.resume().catch(() => {});
  try {
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
}

/**
 * Alarma sonora sintetizada con Web Audio API: mientras `alarm` sea true y el sonido esté activado, suena un
 * pitido cada ALARM_MIN_INTERVAL_MS. El sonido se activa con un gesto del usuario (lo exige el navegador).
 */
export function useAlarm(alarm) {
  const [soundEnabled, setSoundEnabled] = useState(false);
  const audioCtxRef = useRef(null);
  const lastBeepRef = useRef(0);

  useEffect(() => {
    if (!alarm || !soundEnabled) return;
    let timer;
    const ring = () => {
      playBeep(audioCtxRef.current);
      lastBeepRef.current = Date.now();
      timer = setTimeout(ring, ALARM_MIN_INTERVAL_MS);
    };
    // Si la infracción desaparece y vuelve enseguida, el pitido nuevo no se pega al anterior
    timer = setTimeout(ring, Math.max(0, lastBeepRef.current + ALARM_MIN_INTERVAL_MS - Date.now()));
    return () => clearTimeout(timer);
  }, [alarm, soundEnabled]);

  // El AudioContext se crea (o se reanuda) en el gesto del usuario que activa el sonido
  const toggleSound = (e) => {
    const enabled = e.target.checked;
    if (enabled) {
      if (!audioCtxRef.current) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        audioCtxRef.current = new AudioCtx();
      }
      if (audioCtxRef.current.state === 'suspended') audioCtxRef.current.resume().catch(() => {});
    }
    setSoundEnabled(enabled);
  };

  return { soundEnabled, toggleSound };
}
