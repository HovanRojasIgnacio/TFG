import { useState, useEffect, useCallback } from 'react';
import { API_BASE } from '../config';

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
    fetch(`${API_BASE}/api/cameras`)
      .then(res => res.json())
      .then(data => {
        setAvailableCameras(data.available_local || []);
        if (data.current) {
          setTelemetry(prev => ({ ...prev, camera: data.current, camera_error: data.error }));
        }
      })
      .catch(console.error);
  }, [setTelemetry]);

  useEffect(loadCameras, [loadCameras]);

  const requestSelect = async (camera, networkErrorMessage) => {
    setSelectError(null);
    try {
      const response = await fetch(`${API_BASE}/api/cameras/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(camera)
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
    }
  };

  const selectLocalCamera = (index) => {
    const camera = availableCameras.find(c => c.index === index);
    if (camera) {
      requestSelect({ type: "local", index: camera.index, name: camera.name }, "Error de conexión al cambiar cámara");
    }
  };

  const selectIpCamera = async (url, name) => {
    if (!url.trim()) return;
    setTestingCamera(true);
    try {
      await requestSelect(
        { type: "ip", url: url.trim(), name: name.trim() || "Cámara IP" },
        "Error de conexión al probar cámara IP"
      );
    } finally {
      setTestingCamera(false);
    }
  };

  const refreshCameras = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/cameras?refresh=true`);
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
