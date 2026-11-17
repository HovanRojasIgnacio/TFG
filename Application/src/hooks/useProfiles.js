import { useState, useEffect } from 'react';
import { getJson, postJson } from '../lib/api';

/**
 * Modos de detección (perfiles) y cambio de modo.
 * La lista de modos instalados se pide a /api/profiles cuando el backend está en línea; el modo actual, el que
 * se está cargando y el error de una carga fallida llegan con la telemetría (`current`, `loading`, `error`).
 */
export function useProfiles(backendOnline, current, loading, error) {
  const [profiles, setProfiles] = useState([]);
  // Error del último intento de cambio (petición rechazada o sin conexión); el de una carga fallida en el
  // backend llega en `error`
  const [selectError, setSelectError] = useState(null);

  useEffect(() => {
    if (backendOnline !== true) return;
    let cancelled = false;
    getJson("/api/profiles")
      .then(data => {
        if (!cancelled && Array.isArray(data.profiles)) setProfiles(data.profiles);
      })
      .catch(() => {});  // Se vuelve a pedir cuando useTelemetry vea el backend en línea
    return () => { cancelled = true; };
  }, [backendOnline]);

  const selectProfile = async (id) => {
    setSelectError(null);
    try {
      const { ok, data } = await postJson("/api/profiles/select", { id });
      if (!ok) {
        setSelectError(typeof data.detail === "string" ? data.detail : "No se pudo cambiar el modo");
      }
    } catch {
      setSelectError("Error de conexión al cambiar el modo de detección");
    }
  };

  return { profiles, current, loading, error, selectError, selectProfile };
}
