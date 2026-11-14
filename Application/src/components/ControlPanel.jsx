import { DEFAULT_CONF } from '../config';
import { getCameraDisplayName } from '../lib/telemetry';
import ErrorMessage from './ErrorMessage';

export default function ControlPanel({
  confThreshold,
  onConfidenceChange,
  soundEnabled,
  onToggleSound,
  camera,
  cameraError,
  confidenceError,
  onOpenCameraModal,
  profiles,
  onSelectProfile
}) {
  const conf = confThreshold ?? DEFAULT_CONF;
  const loadingProfile = profiles.loading
    ? profiles.profiles.find(p => p.id === profiles.loading)?.name ?? profiles.loading
    : null;
  const profileError = profiles.error || profiles.selectError;

  return (
    <section className="card config-card" aria-labelledby="console-title">
      <h2 id="console-title">Consola de Control</h2>
      <div className="divider" />

      <div className="control-group">
        <label className="control-label" htmlFor="profile-select">Modo de detección:</label>
        <select
          id="profile-select"
          className="mode-select"
          aria-describedby="profile-help"
          value={profiles.loading ?? profiles.current ?? ""}
          disabled={!!profiles.loading || profiles.profiles.length === 0}
          onChange={(e) => onSelectProfile(e.target.value)}
        >
          {profiles.profiles.length === 0 && <option value="">Sin conexión</option>}
          {profiles.profiles.map(p => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
        <p className="slider-desc" id="profile-help" aria-live="polite">
          {loadingProfile
            ? `Cargando modelo «${loadingProfile}»… (la primera vez tarda alrededor de un minuto)`
            : "Elementos de protección que se comprueban."}
        </p>
        {profileError && <ErrorMessage>{profileError}</ErrorMessage>}
      </div>

      <div className="control-group">
        <div className="slider-header">
          <label className="control-label" htmlFor="confidence-slider">Umbral de Confianza:</label>
          <span className="slider-value">{conf.toFixed(3)}</span>
        </div>
        <input
          id="confidence-slider"
          aria-describedby="confidence-help"
          aria-valuetext={`${Math.round(conf * 100)} por ciento`}
          type="range"
          min="0.10"
          max="0.95"
          step="0.025"
          value={conf}
          onChange={onConfidenceChange}
          className="modern-slider"
        />
        <p className="slider-desc" id="confidence-help">
          Más alto: menos detecciones, pero más seguras. Más bajo: más detecciones, con más errores.
        </p>
        {confidenceError && <ErrorMessage>{confidenceError}</ErrorMessage>}
      </div>

      <div className="control-group">
        <div className="control-row">
          <label className="control-label" htmlFor="alarm-toggle">Alarma Sonora Reactiva:</label>
          <label className="switch">
            <input
              id="alarm-toggle"
              type="checkbox"
              aria-describedby="alarm-help"
              checked={soundEnabled}
              onChange={onToggleSound}
            />
            <span className="slider-switch"></span>
          </label>
        </div>
        <p className="slider-desc" id="alarm-help">Suena un tono mientras falta un elemento de protección.</p>
      </div>

      <div className="control-group">
        <div className="control-row">
          <span className="control-label" id="camera-label">Fuente de Vídeo:</span>
          <button
            className="camera-select-btn"
            aria-describedby="camera-label"
            aria-haspopup="dialog"
            onClick={onOpenCameraModal}
          >
            <span className="camera-icon" aria-hidden="true">📷</span>
            <span>{getCameraDisplayName(camera)}</span>
            <span className="camera-chevron" aria-hidden="true">▼</span>
          </button>
        </div>
        {cameraError && <ErrorMessage role="status">{cameraError}</ErrorMessage>}
      </div>
    </section>
  );
}
