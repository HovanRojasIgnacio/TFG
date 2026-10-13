import { useEffect, useRef } from 'react';
import { API_BASE, CONF_DEBOUNCE_MS } from '../config';

/**
 * Manejador del deslizador de confianza: actualiza la interfaz al instante
 * y envía el valor al backend solo cuando el usuario deja de moverlo.
 */
export function useConfidence(setTelemetry) {
  const timerRef = useRef(null);

  useEffect(() => () => clearTimeout(timerRef.current), []);

  return (e) => {
    const value = parseFloat(e.target.value);
    setTelemetry(prev => ({ ...prev, conf_threshold: value }));

    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      fetch(`${API_BASE}/api/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conf: value })
      }).catch(console.error);
    }, CONF_DEBOUNCE_MS);
  };
}
