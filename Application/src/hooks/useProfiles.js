import { useState, useEffect, useCallback } from 'react';
import { getJson, postJson } from '../lib/api';

/** Devuelve el estado de /api/profiles, o null si el backend no responde. */
async function fetchProfiles() {
  try {
    const data = await getJson("/api/profiles");
    return {
      profiles: Array.isArray(data.profiles) ? data.profiles : [],
      current: data.current ?? null,
      loading: data.loading ?? null,
      error: data.error ?? null
    };
  } catch {
    return null;  // Sin backend o con un error del servidor: se vuelve a pedir cuando useTelemetry lo vea en línea
  }
}

/**
 * Modos de detección (perfiles) del backend y cambio de modo.
 * Se piden al montar, cuando el backend vuelve a estar en línea y cada vez que cambia
 * `telemetryLoading` (el campo profile_loading de /api/stats): así se recoge el modo nuevo
 * en cuanto el backend termina de cargarlo, sin un sondeo propio.
 */
export function useProfiles(backendOnline, telemetryLoading) {
  const [state, setState] = useState({ profiles: [], current: null, loading: null, error: null });
  // Error del último intento de cambio (petición rechazada o sin conexión); el de una carga
  // fallida en el backend llega en state.error
  const [selectError, setSelectError] = useState(null);

  const loadProfiles = useCallback(() => fetchProfiles().then(data => data && setState(data)), []);

  useEffect(() => {
    if (backendOnline === false) return;
    let cancelled = false;  // Una respuesta tardía no debe pisar a la de una consulta posterior
    fetchProfiles().then(data => {
      if (data && !cancelled) setState(data);
    });
    return () => { cancelled = true; };
  }, [backendOnline, telemetryLoading]);

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
    loadProfiles();
  };

  return { ...state, selectError, selectProfile };
}
