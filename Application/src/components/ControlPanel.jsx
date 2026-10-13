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
          <label className="control-label" htmlFor="confidence-slider">Umbral de Confianza (Conf):</label>
          <span className="slider-value">{conf.toFixed(3)}</span>
        </div>
        <input
          id="confidence-slider"
          aria-describedby="confidence-help"
          type="range"
          min="0.10"
          max="0.95"
          step="0.025"
          value={conf}
          onChange={onConfidenceChange}
          className="modern-slider"
        />
        <p className="slider-desc" id="confidence-help">Ajusta la exigencia probabilística de las predicciones.</p>
      </div>

      <div className="control-row">
        <label className="control-label" htmlFor="alarm-toggle">Alarma Sonora Reactiva:</label>
        <label className="switch">
          <input
            id="alarm-toggle"
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
