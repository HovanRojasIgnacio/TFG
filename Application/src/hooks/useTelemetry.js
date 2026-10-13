import { useState, useEffect, useRef } from 'react';
import { API_BASE, POLL_INTERVAL_MS, DEFAULT_CONF } from '../config';

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
 * `onStats(data)` se llama con cada respuesta (siempre con la última versión de la función).
 */
export function useTelemetry(isActive, onStats) {
  const [telemetry, setTelemetry] = useState(INITIAL_TELEMETRY);

  const onStatsRef = useRef(onStats);
  useEffect(() => {
    onStatsRef.current = onStats;
  });

  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(() => {
      fetch(`${API_BASE}/api/stats`)
        .then(res => {
          if (!res.ok) throw new Error("Error en servidor");
          return res.json();
        })
        .then(data => {
          setTelemetry(prev => ({
            ...prev,
            ...data,
            active_detections: data.active_detections || prev.active_detections,
            infractions: Array.isArray(data.infractions) ? data.infractions : prev.infractions
          }));
          onStatsRef.current?.(data);
        })
        .catch(() => {
          // El backend puede estar reconectando
        });
    }, POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [isActive]);

  return [telemetry, setTelemetry];
}
