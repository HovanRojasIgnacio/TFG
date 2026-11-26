import { useState, useEffect, useCallback, useRef } from 'react';
import { POLL_INTERVAL_MS } from '../config';
import { getJson } from '../lib/api';

/**
 * Consulta /api/stats periódicamente mientras `isActive` sea true. Es el único que escribe la telemetría: tras
 * un cambio (cámara, umbral), quien lo hizo llama a `refresh()` para no esperar a la consulta siguiente.
 *
 * Cada consulta del sondeo espera a que termine la anterior, así no se acumulan peticiones si el backend va
 * lento, y una respuesta más antigua que la última aplicada (refresh puede adelantarse al sondeo) se descarta.
 * `telemetry` es null hasta la primera respuesta. `online` es null hasta la primera respuesta desde que se
 * inicia o reanuda la monitorización (los datos de antes de la pausa no son actuales: no deben hacer sonar la
 * alarma de una infracción antigua) y false si la última consulta falló.
 */
export function useTelemetry(isActive) {
  const [telemetry, setTelemetry] = useState(null);
  const [online, setOnline] = useState(null);
  const requests = useRef({ sent: 0, applied: 0 });

  // Al reanudar, los datos de antes de la pausa dejan de contar como actuales (ajuste de estado durante el render,
  // el patrón que recomienda React)
  const [wasActive, setWasActive] = useState(isActive);
  if (isActive !== wasActive) {
    setWasActive(isActive);
    if (isActive) setOnline(null);
  }

  const refresh = useCallback(async (signal) => {
    const request = ++requests.current.sent;
    try {
      // Además de cancelarse al desmontar, cada consulta caduca (lo hace getJson): si la red cae sin cerrar la
      // conexión, la petición quedaría pendiente y el sondeo se pararía sin avisar
      const data = await getJson("/api/stats", { signal });
      if (request < requests.current.applied) return;
      requests.current.applied = request;
      setTelemetry(data);
      setOnline(true);
    } catch {
      if (signal?.aborted || request < requests.current.applied) return;
      setOnline(false);  // El backend puede estar arrancando o reconectando
    }
  }, []);

  useEffect(() => {
    if (!isActive) return;
    const controller = new AbortController();
    let timer;
    const poll = async () => {
      await refresh(controller.signal);
      if (!controller.signal.aborted) timer = setTimeout(poll, POLL_INTERVAL_MS);
    };
    poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [isActive, refresh]);

  return { telemetry, online, refresh };
}
