import { useState, useEffect, useRef } from 'react';
import { API_BASE, CONF_DEBOUNCE_MS, REQUEST_TIMEOUT_MS } from '../config';

/**
 * Manejador del deslizador de confianza: actualiza la interfaz al instante
 * y envía el valor al backend solo cuando el usuario deja de moverlo.
 * Mientras el valor no está confirmado por el backend, `pendingConf` tiene prioridad
 * sobre el que llega en el sondeo de /api/stats (así el deslizador no salta hacia atrás).
 */
export function useConfidence(setTelemetry) {
  const [pendingConf, setPendingConf] = useState(null);
  const timerRef = useRef(null);

  useEffect(() => () => clearTimeout(timerRef.current), []);

  const onChange = (e) => {
    const value = parseFloat(e.target.value);
    setPendingConf(value);

    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/settings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ conf: value }),
          signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
        });
        if (response.ok) {
          const data = await response.json();
          setTelemetry(prev => ({ ...prev, conf_threshold: data.conf_threshold }));
        }
      } catch (error) {
        console.error(error);
      }
      // Solo se libera si el usuario no ha movido el deslizador desde este envío
      setPendingConf(current => (current === value ? null : current));
    }, CONF_DEBOUNCE_MS);
  };

  return { pendingConf, onChange };
}
