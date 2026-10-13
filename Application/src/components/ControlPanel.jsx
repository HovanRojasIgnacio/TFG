import { DEFAULT_CONF } from '../config';

function getCameraDisplayName(camera) {
  if (!camera) return "Cámara no seleccionada";
  if (camera.type === "local") {
    return camera.name || `Cámara Local ${camera.index}`;
  }
  return camera.name || camera.url;
}

export default function ControlPanel({
  confThreshold,
  onConfidenceChange,
  soundEnabled,
  onToggleSound,
  camera,
  cameraError,
  onOpenCameraModal
}) {
  const conf = confThreshold ?? DEFAULT_CONF;

  return (
    <div className="card config-card">
      <h3>Consola de Control</h3>
      <div className="divider" />

      <div className="control-group">
        <div className="slider-header">
          <span className="control-label">Umbral de Confianza (Conf):</span>
          <span className="slider-value">{conf.toFixed(3)}</span>
        </div>
        <input
          type="range"
          min="0.10"
          max="0.95"
          step="0.025"
          value={conf}
          onChange={onConfidenceChange}
          className="modern-slider"
        />
        <p className="slider-desc">Ajusta la exigencia probabilística de las predicciones.</p>
      </div>

      <div className="control-row">
        <span className="control-label">Alarma Sonora Reactiva:</span>
        <label className="switch">
          <input
            type="checkbox"
            checked={soundEnabled}
            onChange={onToggleSound}
          />
          <span className="slider-switch"></span>
        </label>
      </div>

      <div className="control-group">
        <div className="control-row">
          <span className="control-label">Fuente de Vídeo:</span>
          <button
            className="camera-select-btn"
            onClick={onOpenCameraModal}
          >
            <span className="camera-icon">📷</span>
            <span>{getCameraDisplayName(camera)}</span>
            <span className="camera-chevron">▼</span>
          </button>
        </div>
        {cameraError && (
          <div className="camera-error">
            ⚠ {cameraError}
          </div>
        )}
      </div>
    </div>
  );
}
