import ErrorMessage from './ErrorMessage';

// Paso del deslizador; los límites (profiles.CONF_MIN y CONF_MAX) llegan del servidor en conf_range
const CONF_STEP = 0.025;

export default function ControlPanel({
  confThreshold,
  confRange,
  onConfidenceChange,
  soundEnabled,
  onToggleSound,
  camera,
  cameraError,
  confidenceError,
  onOpenCameraModal,
  profiles,
  currentProfile,
  loadingProfile,
  profileError,
  onSelectProfile,
  profilesPaused
}) {
  // Hasta la primera respuesta del servidor no se conocen el umbral ni sus límites
  const confKnown = confThreshold != null && confRange != null;
  const loadingName = loadingProfile ? profiles.find(p => p.id === loadingProfile)?.name ?? loadingProfile : null;

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
          value={loadingProfile ?? currentProfile ?? ""}
          disabled={profilesPaused || !!loadingProfile || profiles.length === 0}
          onChange={(e) => onSelectProfile(e.target.value)}
        >
          {profiles.length === 0 && <option value="">Sin conexión</option>}
          {profiles.map(p => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
        <p className="slider-desc" id="profile-help" aria-live="polite">
          {/* En pausa no llega la telemetría, que es la que dice cuándo termina la carga de un modo */}
          {profilesPaused
            ? "Reanuda la monitorización para cambiar de modo."
            : loadingName
              ? `Cargando modelo «${loadingName}»… (la primera vez tarda alrededor de un minuto)`
              : "Elementos de protección que se comprueban."}
        </p>
        {profileError && <ErrorMessage>{profileError}</ErrorMessage>}
      </div>

      <div className="control-group">
        <div className="slider-header">
          <label className="control-label" htmlFor="confidence-slider">Umbral de Confianza:</label>
          <span className="slider-value">{confKnown ? confThreshold.toFixed(3) : "—"}</span>
        </div>
        <input
          id="confidence-slider"
          aria-describedby="confidence-help"
          aria-valuetext={confKnown ? `${Math.round(confThreshold * 100)} por ciento` : "Sin datos del servidor"}
          type="range"
          min={confRange?.[0]}
          max={confRange?.[1]}
          step={CONF_STEP}
          value={confThreshold ?? ""}
          disabled={!confKnown}
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
            <span>{camera?.name ?? "Cámara no seleccionada"}</span>
            <span className="camera-chevron" aria-hidden="true">▼</span>
          </button>
        </div>
        {cameraError && <ErrorMessage role="status">{cameraError}</ErrorMessage>}
      </div>
    </section>
  );
}
