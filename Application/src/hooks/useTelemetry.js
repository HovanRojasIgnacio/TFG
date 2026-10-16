import { useState, useEffect, useRef } from 'react';
import { API_BASE, POLL_INTERVAL_MS, DEFAULT_CONF, REQUEST_TIMEOUT_MS } from '../config';

const INITIAL_TELEMETRY = {
  fps: 0,
  device: "CONECTANDO...",
  weights_format: "DETECTANDO...",
  active_detections: { helmet: 0, no_helmet: 0, face: 0 },
  safety_score: 100,
  conf_threshold: DEFAULT_CONF,
  infractions: [],
  camera: null,
  camera_error: null
};

/**
 * Consulta /api/stats periódicamente mientras `isActive` sea true.
 * Cada consulta espera a que termine la anterior, así no se acumulan peticiones si el
 * backend va lento ni una respuesta antigua pisa a una más reciente.
 * `onStats(data)` se llama con cada respuesta (siempre con la última versión de la función).
 * `online` es null hasta la primera respuesta y false si la última consulta falló.
 */
export function useTelemetry(isActive, onStats) {
  const [telemetry, setTelemetry] = useState(INITIAL_TELEMETRY);
  const [online, setOnline] = useState(null);

  const onStatsRef = useRef(onStats);
  useEffect(() => {
    onStatsRef.current = onStats;
  });

  useEffect(() => {
    if (!isActive) return;

    const controller = new AbortController();
    let timer;

    const poll = async () => {
      try {
        // Además de cancelarse al desmontar, cada consulta caduca: si la red cae sin cerrar la conexión,
        // la petición quedaría pendiente y el sondeo (que espera a la anterior) se pararía sin avisar
        const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(REQUEST_TIMEOUT_MS)]);
        const res = await fetch(`${API_BASE}/api/stats`, { signal });
        if (!res.ok) throw new Error("Error en servidor");
        const data = await res.json();
        setTelemetry(prev => ({
          ...prev,
          ...data,
          active_detections: data.active_detections || prev.active_detections,
          infractions: Array.isArray(data.infractions) ? data.infractions : prev.infractions
        }));
        setOnline(true);
        onStatsRef.current?.(data);
      } catch {
        if (controller.signal.aborted) return;
        setOnline(false);  // El backend puede estar arrancando o reconectando
      }
      if (!controller.signal.aborted) {
        timer = setTimeout(poll, POLL_INTERVAL_MS);
      }
    };
    poll();

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [isActive]);

  return [telemetry, setTelemetry, online];
}
