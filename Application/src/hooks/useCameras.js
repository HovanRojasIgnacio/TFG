import { useState, useEffect, useCallback } from 'react';
import { CAMERA_SELECT_TIMEOUT_MS } from '../config';
import { getJson, postJson } from '../lib/api';

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
    getJson("/api/cameras")
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
      const { ok, data } = await postJson("/api/cameras/select", camera, { timeoutMs: CAMERA_SELECT_TIMEOUT_MS });
      if (ok) {
        setTelemetry(prev => ({ ...prev, camera: data.camera, camera_error: null }));
        onSelected();
        return;
      }
      setSelectError(typeof data.detail === "string" ? data.detail : "No se pudo cambiar la cámara");
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
    setTestingCamera(true);
    try {
      // Abre cada cámara para comprobarla: tarda
      const data = await getJson("/api/cameras?refresh=true", { timeoutMs: CAMERA_SELECT_TIMEOUT_MS });
      setAvailableCameras(data.available_local || []);
    } catch {
      setSelectError("No se pudo actualizar la lista de cámaras");
    } finally {
      setTestingCamera(false);
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
