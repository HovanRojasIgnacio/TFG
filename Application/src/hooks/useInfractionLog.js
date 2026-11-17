import { useState, useEffect } from 'react';
import { RECENT_INFRACTIONS } from '../config';
import { getJson } from '../lib/api';

/**
 * Registro de infracciones (hasta 100). Se pide a /api/infractions solo cuando cambia: cuando /api/stats trae
 * otro `lastId` (una infracción nueva) u otro `total` (la limpieza borró alguna), y cuando el backend vuelve a
 * estar en línea. La galería muestra las RECENT_INFRACTIONS últimas o, con showAll, todas.
 */
export function useInfractionLog(lastId, total, backendOnline) {
  const [entries, setEntries] = useState([]);
  const [showingAll, setShowingAll] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (backendOnline !== true) return;
    let cancelled = false;  // Una respuesta tardía no debe pisar a la de una petición posterior
    getJson("/api/infractions")
      .then(data => {
        if (cancelled) return;
        setEntries(Array.isArray(data.infractions) ? data.infractions : []);
        setError(null);
      })
      .catch(() => {
        if (!cancelled) setError("No se pudo cargar el registro de infracciones");
      });
    return () => { cancelled = true; };
  }, [lastId, total, backendOnline]);

  return {
    infractions: showingAll ? entries : entries.slice(-RECENT_INFRACTIONS),
    total: entries.length,
    showingAll,
    error,
    showAll: () => setShowingAll(true),
    showRecent: () => setShowingAll(false)
  };
}
