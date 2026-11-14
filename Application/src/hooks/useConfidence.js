import { useState, useEffect, useRef } from 'react';
import { CONF_DEBOUNCE_MS } from '../config';
import { postJson } from '../lib/api';

/**
 * Manejador del deslizador de confianza: actualiza la interfaz al instante
 * y envía el valor al backend solo cuando el usuario deja de moverlo.
 * Mientras el valor no está confirmado por el backend, `pendingConf` tiene prioridad
 * sobre el que llega en el sondeo de /api/stats (así el deslizador no salta hacia atrás).
 */
export function useConfidence(setTelemetry) {
  const [pendingConf, setPendingConf] = useState(null);
  const [error, setError] = useState(null);
  const timerRef = useRef(null);

  useEffect(() => () => clearTimeout(timerRef.current), []);

  const onChange = (e) => {
    const value = parseFloat(e.target.value);
    setPendingConf(value);
    setError(null);

    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      try {
        const { ok, data } = await postJson("/api/settings", { conf: value });
        if (ok) {
          setTelemetry(prev => ({ ...prev, conf_threshold: data.conf_threshold }));
          setError(null);
        } else {
          setError(typeof data.detail === "string" ? data.detail : "No se pudo cambiar el umbral");
        }
      } catch {
        setError("No se pudo cambiar el umbral: sin conexión con el servidor");
      }
      // Solo se libera si el usuario no ha movido el deslizador desde este envío
      setPendingConf(current => (current === value ? null : current));
    }, CONF_DEBOUNCE_MS);
  };

  return { pendingConf, onChange, error };
}
