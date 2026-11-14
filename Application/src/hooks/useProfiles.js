import { useState, useEffect, useCallback, useRef } from 'react';
import { PROFILE_POLL_MS } from '../config';
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
 * `telemetryLoading` (el campo profile_loading de /api/stats). Además, mientras se carga un modo se
 * consulta cada PROFILE_POLL_MS hasta que termina: la telemetría puede estar en pausa o no llegar a ver
 * el cambio, y sin esto el selector se quedaría bloqueado.
 */
export function useProfiles(backendOnline, telemetryLoading) {
  const [state, setState] = useState({ profiles: [], current: null, loading: null, error: null });
  // Error del último intento de cambio (petición rechazada o sin conexión); el de una carga
  // fallida en el backend llega en state.error
  const [selectError, setSelectError] = useState(null);

  const wasLoading = useRef(false);

  // Aplica una respuesta de /api/profiles. Al terminar una carga se borra el error del intento de cambio
  // (por ejemplo el 409 «Ya se está cargando…»): si no, taparía o confundiría lo que pase después.
  const applyProfiles = useCallback((data) => {
    if (data.loading === null && wasLoading.current) setSelectError(null);
    wasLoading.current = data.loading !== null;
    setState(data);
  }, []);

  const loadProfiles = useCallback(() => fetchProfiles().then(data => data && applyProfiles(data)), [applyProfiles]);

  useEffect(() => {
    if (backendOnline === false) return;
    let cancelled = false;  // Una respuesta tardía no debe pisar a la de una consulta posterior
    fetchProfiles().then(data => {
      if (data && !cancelled) applyProfiles(data);
    });
    return () => { cancelled = true; };
  }, [backendOnline, telemetryLoading, applyProfiles]);

  // Mientras hay un modo cargándose se vuelve a consultar hasta que `loading` sea null
  useEffect(() => {
    if (state.loading === null) return;
    let cancelled = false;
    let timer;
    const poll = async () => {
      const data = await fetchProfiles();
      if (cancelled) return;
      if (data) applyProfiles(data);
      timer = setTimeout(poll, PROFILE_POLL_MS);
    };
    timer = setTimeout(poll, PROFILE_POLL_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [state.loading, applyProfiles]);

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
