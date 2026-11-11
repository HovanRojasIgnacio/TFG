import { useState, useEffect, useCallback } from 'react';
import { API_BASE, REQUEST_TIMEOUT_MS, CAMERA_SELECT_TIMEOUT_MS } from '../config';

/**
 * Lista de cámaras locales y selección de fuente de vídeo (local o IP).
 * `onSelected` se llama cuando el backend acepta el cambio de cámara.
 */
export function useCameras(setTelemetry, onSelected) {
  const [availableCameras, setAvailableCameras] = useState([]);
  const [testingCamera, setTestingCamera] = useState(false);
  // Error del último intento de cambio de cámara. Va aparte de telemetry.camera_error, que el
  // sondeo de /api/stats sobrescribe cada 500 ms con el estado del backend.
  const [selectError, setSelectError] = useState(null);

  // Cámaras disponibles y cámara actual. Se piden al montar y cada vez que se abre el diálogo: si el
  // backend aún no estaba en marcha al cargar el panel, la lista no se quedaría vacía para siempre.
  const loadCameras = useCallback(() => {
    fetch(`${API_BASE}/api/cameras`, { signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) })
      .then(res => {
        if (!res.ok) throw new Error(`Error ${res.status} al pedir las cámaras`);
        return res.json();
      })
      .then(data => {
        setAvailableCameras(data.available_local || []);
        if (data.current) {
          setTelemetry(prev => ({ ...prev, camera: data.current, camera_error: data.error }));
        }
      })
      .catch(console.error);
  }, [setTelemetry]);

  useEffect(loadCameras, [loadCameras]);

  // Mientras el backend prueba la cámara (local o IP) no se puede lanzar otra prueba
  const requestSelect = async (camera, networkErrorMessage) => {
    setSelectError(null);
    setTestingCamera(true);
    try {
      const response = await fetch(`${API_BASE}/api/cameras/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(camera),
        signal: AbortSignal.timeout(CAMERA_SELECT_TIMEOUT_MS)
      });
      if (response.ok) {
        const data = await response.json();
        setTelemetry(prev => ({ ...prev, camera: data.camera, camera_error: null }));
        onSelected();
        return;
      }
      const error = await response.json();
      setSelectError(typeof error.detail === "string" ? error.detail : "No se pudo cambiar la cámara");
    } catch {
      setSelectError(networkErrorMessage);
    } finally {
      setTestingCamera(false);
    }
  };

  const selectLocalCamera = (index) => {
    const camera = availableCameras.find(c => c.index === index);
    if (camera) {
      requestSelect({ type: "local", index: camera.index, name: camera.name }, "Error de conexión al cambiar cámara");
    }
  };

  const selectIpCamera = (url, name) => {
    if (!url.trim()) return;
    requestSelect(
      { type: "ip", url: url.trim(), name: name.trim() || "Cámara IP" },
      "Error de conexión al probar cámara IP"
    );
  };

  const refreshCameras = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/cameras?refresh=true`, {
        signal: AbortSignal.timeout(CAMERA_SELECT_TIMEOUT_MS)  // Abre cada cámara para comprobarla: tarda
      });
      if (!response.ok) throw new Error(`Error ${response.status}`);
      const data = await response.json();
      setAvailableCameras(data.available_local || []);
    } catch (error) {
      console.error("Error refreshing cameras:", error);
    }
  };

  const clearSelectError = () => setSelectError(null);

  return {
    availableCameras,
    testingCamera,
    selectError,
    clearSelectError,
    loadCameras,
    selectLocalCamera,
    selectIpCamera,
    refreshCameras
  };
}
