import { useState, useEffect, useRef } from 'react';
import './App.css';

// Configurable con VITE_API_BASE (por ejemplo en un archivo .env); por defecto, el backend local
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const ALARM_MIN_INTERVAL_MS = 1500;
const CONF_DEBOUNCE_MS = 250;
const STREAM_RETRY_MS = 3000;

function App() {
  const [isActive, setIsActive] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [streamError, setStreamError] = useState(false);
  const [activeModalImg, setActiveModalImg] = useState(null);
  const [cameraModalOpen, setCameraModalOpen] = useState(false);
  const [availableCameras, setAvailableCameras] = useState([]);
  const [ipCameraUrl, setIpCameraUrl] = useState("");
  const [ipCameraName, setIpCameraName] = useState("");
  const [testingCamera, setTestingCamera] = useState(false);

  const [telemetry, setTelemetry] = useState({
    fps: 0,
    device: "CONECTANDO...",
    weights_format: "DETECTANDO...",
    active_detections: { helmet: 0, no_helmet: 0, face: 0 },
    safety_score: 100,
    conf_threshold: 0.325,
    infractions: [],
    camera: null,
    camera_error: null
  });

  const audioCtxRef = useRef(null);
  const lastBeepRef = useRef(0);
  const modalRef = useRef(null);
  const confTimerRef = useRef(null);

  useEffect(() => () => clearTimeout(confTimerRef.current), []);

  // Si el stream falla (por ejemplo, el backend aún arranca), se reintenta solo mientras la monitorización esté activa
  useEffect(() => {
    if (!streamError || !isActive) return;
    const timer = setTimeout(() => setStreamError(false), STREAM_RETRY_MS);
    return () => clearTimeout(timer);
  }, [streamError, isActive]);

  // Modal abierto: el foco entra en el diálogo, Tab queda atrapado dentro, Escape lo cierra
  // y al cerrarse el foco vuelve al elemento que lo abrió
  useEffect(() => {
    if (!activeModalImg && !cameraModalOpen) return;
    const opener = document.activeElement;
    const modal = modalRef.current;
    modal?.focus();

    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        setActiveModalImg(null);
        setCameraModalOpen(false);
        return;
      }
      if (e.key !== 'Tab' || !modal) return;

      const focusable = modal.querySelectorAll(
        'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href]'
      );
      if (focusable.length === 0) {
        e.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && (document.activeElement === first || document.activeElement === modal)) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      opener?.focus?.();
    };
  }, [activeModalImg, cameraModalOpen]);

  // Inicializa o reanuda AudioContext tras interacción
  const initAudio = () => {
    if (!audioCtxRef.current) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioCtxRef.current = new AudioCtx();
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }
  };

  const triggerBeepAlarm = () => {
    if (!audioCtxRef.current) return;
    const now = Date.now();
    if (now - lastBeepRef.current < ALARM_MIN_INTERVAL_MS) return;
    lastBeepRef.current = now;
    try {
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.25);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      osc.stop(ctx.currentTime + 0.25);
    } catch {
      // Ignorar fallos puntuales de buffer de audio
    }
  };

  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(() => {
      fetch(`${API_BASE}/api/stats`)
        .then(res => {
          if (!res.ok) throw new Error("Error en servidor");
          return res.json();
        })
        .then(data => {
          setTelemetry(prev => ({
            ...prev,
            ...data,
            active_detections: data.active_detections || prev.active_detections,
            infractions: Array.isArray(data.infractions) ? data.infractions : prev.infractions
          }));

          if (soundEnabled && data.active_detections?.no_helmet > 0) {
            triggerBeepAlarm();
          }
        })
        .catch(() => {
          // El backend puede estar reconectando
        });
    }, 500);

    return () => clearInterval(interval);
  }, [isActive, soundEnabled]);

  // Fetch available cameras on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/cameras`)
      .then(res => res.json())
      .then(data => {
        setAvailableCameras(data.available_local || []);
        if (data.current) {
          setTelemetry(prev => ({ ...prev, camera: data.current, camera_error: data.error }));
        }
      })
      .catch(console.error);
  }, []);

  const handleConfidenceChange = (e) => {
    const value = parseFloat(e.target.value);
    setTelemetry(prev => ({ ...prev, conf_threshold: value }));

    // Se envía al backend solo cuando el usuario deja de mover el deslizador
    clearTimeout(confTimerRef.current);
    confTimerRef.current = setTimeout(() => {
      fetch(`${API_BASE}/api/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conf: value })
      }).catch(console.error);
    }, CONF_DEBOUNCE_MS);
  };

  const toggleSound = (e) => {
    const enabled = e.target.checked;
    if (enabled) initAudio();
    setSoundEnabled(enabled);
  };

  const selectCamera = async (camera) => {
    try {
      const response = await fetch(`${API_BASE}/api/cameras/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(camera)
      });
      if (response.ok) {
        const data = await response.json();
        setTelemetry(prev => ({ ...prev, camera: data.camera, camera_error: null }));
        setCameraModalOpen(false);
      } else {
        const error = await response.json();
        setTelemetry(prev => ({ ...prev, camera_error: error.detail }));
      }
    } catch {
      setTelemetry(prev => ({ ...prev, camera_error: "Error de conexión al cambiar cámara" }));
    }
  };

  const handleLocalCameraSelect = (index) => {
    const camera = availableCameras.find(c => c.index === index);
    if (camera) {
      selectCamera({ type: "local", index: camera.index, name: camera.name });
    }
  };

  const handleIpCameraTest = async () => {
    if (!ipCameraUrl.trim()) return;
    setTestingCamera(true);
    try {
      const response = await fetch(`${API_BASE}/api/cameras/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          type: "ip", 
          url: ipCameraUrl.trim(), 
          name: ipCameraName.trim() || "Cámara IP" 
        })
      });
      if (response.ok) {
        const data = await response.json();
        setTelemetry(prev => ({ ...prev, camera: data.camera, camera_error: null }));
        setIpCameraUrl("");
        setIpCameraName("");
        setCameraModalOpen(false);
      } else {
        const error = await response.json();
        setTelemetry(prev => ({ ...prev, camera_error: error.detail }));
      }
    } catch {
      setTelemetry(prev => ({ ...prev, camera_error: "Error de conexión al probar cámara IP" }));
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

  const getSafetyColor = (score) => {
    if (score >= 90) return "#10b981";
    if (score >= 60) return "#f59e0b";
    return "#ef4444";
  };

  const getCameraDisplayName = () => {
    if (!telemetry.camera) return "Cámara no seleccionada";
    if (telemetry.camera.type === "local") {
      return telemetry.camera.name || `Cámara Local ${telemetry.camera.index}`;
    }
    return telemetry.camera.name || telemetry.camera.url;
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-logo">
          <span className="shield-icon">🛡️</span>
          <div>
            <h1>GuardWatch AI</h1>
            <p className="subtitle">Inspección de EPIs en Tiempo Real — YOLO26s</p>
          </div>
        </div>
        <div className="header-badge">TRABAJO FIN DE GRADO</div>
      </header>

      <main className="main-content">
        <div className={`card video-card ${telemetry.active_detections?.no_helmet > 0 ? 'border-alarm' : ''}`}>
          <div className="card-header">
            <div className="header-left">
              <span className={`pulse-dot ${isActive && !streamError ? 'online' : 'offline'}`}></span>
              <h3>Vídeo Monitorizado en Vivo</h3>
            </div>
            <div className="header-right">
              <span className="hardware-pill">{telemetry.device}</span>
              <span className="format-pill">{telemetry.weights_format}</span>
            </div>
          </div>

          <div className="video-viewport">
            {isActive && !streamError ? (
              <img
                src={`${API_BASE}/api/stream`}
                alt="Señal de vídeo"
                className="stream-img"
                onError={() => setStreamError(true)}
              />
            ) : (
              <div className="viewport-placeholder">
                <span className="placeholder-icon">📷</span>
                <p>{streamError ? "Error al conectar con la señal del servidor. Reintentando automáticamente…" : "La monitorización está en pausa."}</p>
                {streamError && (
                  <button className="retry-btn" onClick={() => setStreamError(false)}>
                    Reintentar conexión
                  </button>
                )}
              </div>
            )}
          </div>

          <div className="card-footer">
            <button
              className={`control-btn ${isActive ? 'danger' : 'success'}`}
              onClick={() => {
                setIsActive(!isActive);
                setStreamError(false);
              }}
            >
              {isActive ? 'Suspender Monitoreo' : 'Iniciar Monitoreo'}
            </button>
          </div>
        </div>

        <div className="right-panel">
          <div className="card config-card">
            <h3>Consola de Control</h3>
            <div className="divider" />

            <div className="control-group">
              <div className="slider-header">
                <span className="control-label">Umbral de Confianza (Conf):</span>
                <span className="slider-value">{(telemetry.conf_threshold ?? 0.325).toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.95"
                step="0.025"
                value={telemetry.conf_threshold ?? 0.325}
                onChange={handleConfidenceChange}
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
                  onChange={toggleSound}
                />
                <span className="slider-switch"></span>
              </label>
            </div>

            <div className="control-group">
              <div className="control-row">
                <span className="control-label">Fuente de Vídeo:</span>
                <button 
                  className="camera-select-btn"
                  onClick={() => setCameraModalOpen(true)}
                >
                  <span className="camera-icon">📷</span>
                  <span>{getCameraDisplayName()}</span>
                  <span className="camera-chevron">▼</span>
                </button>
              </div>
              {telemetry.camera_error && (
                <div className="camera-error">
                  ⚠ {telemetry.camera_error}
                </div>
              )}
            </div>
          </div>

          <div className="card stats-card">
            <h3>Métricas de Inferencia</h3>
            <div className="divider" />

            <div className="metrics-grid">
              <div className="metric-box">
                <span className="metric-title">FPS Inferencia</span>
                <span className="metric-value text-emerald">{isActive ? telemetry.fps : 0}</span>
              </div>
              <div className="metric-box">
                <span className="metric-title">Cumplimiento EPI</span>
                <span className="metric-value" style={{ color: getSafetyColor(telemetry.safety_score) }}>
                  {telemetry.safety_score}%
                </span>
              </div>
            </div>

            <div className="counter-list">
              <div className="counter-item border-green">
                <span>Cascos Correctos:</span>
                <span className="counter-val text-green">{telemetry.active_detections?.helmet ?? 0}</span>
              </div>
              <div className="counter-item border-red">
                <span>Infracciones Críticas:</span>
                <span className="counter-val text-red">{telemetry.active_detections?.no_helmet ?? 0}</span>
              </div>
              <div className="counter-item border-orange">
                <span>Rostros Neutrales:</span>
                <span className="counter-val text-orange">{telemetry.active_detections?.face ?? 0}</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      <section className="card infractions-section">
        <div className="section-header">
          <h3>Registro de Infracciones Guardadas (Snapshots)</h3>
          <span className="log-count">{(telemetry.infractions || []).length} Guardadas</span>
        </div>
        <div className="divider" />

        {(!telemetry.infractions || telemetry.infractions.length === 0) ? (
          <div className="no-infractions">
            <p>✅ Ninguna infracción registrada en la sesión actual. Cumplimiento normativo estricto.</p>
          </div>
        ) : (
          <div className="infractions-grid">
            {telemetry.infractions.slice().reverse().map((inf) => (
              <div className="infraction-item-card" key={inf.id}>
                <div className="item-thumbnail">
                  <img src={inf.image_url} alt="Evidencia de infracción" loading="lazy" />
                  <span className="item-badge-time">{inf.timestamp}</span>
                </div>
                <div className="item-info">
                  <div className="info-top">
                    <span className="info-conf">Conf: {(inf.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <p>{inf.details}</p>
                  <button
                    className="view-btn"
                    onClick={() => setActiveModalImg(inf.image_url)}
                  >
                    Ver Evidencia
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {activeModalImg && (
        <div className="modal-overlay" onClick={() => setActiveModalImg(null)}>
          <div className="modal-content" ref={modalRef} tabIndex={-1} role="dialog" aria-modal="true" aria-label="Evidencia de infracción" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="modal-close" aria-label="Cerrar" onClick={() => setActiveModalImg(null)}>×</button>
            <img src={activeModalImg} alt="Evidencia Detallada" />
            <div className="modal-meta">Evidencia Forense — Captura de fotograma procesada por el motor de inferencia</div>
          </div>
        </div>
      )}

      {cameraModalOpen && (
        <div className="modal-overlay" onClick={() => setCameraModalOpen(false)}>
          <div className="modal-content camera-modal" ref={modalRef} tabIndex={-1} role="dialog" aria-modal="true" aria-label="Seleccionar fuente de vídeo" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="modal-close" aria-label="Cerrar" onClick={() => setCameraModalOpen(false)}>×</button>
            <h3>Seleccionar Fuente de Vídeo</h3>
            
            <div className="camera-tab">
              <button 
                className="tab-btn"
                onClick={() => refreshCameras()}
              >
                🔄 Actualizar cámaras locales
              </button>
            </div>

            <div className="camera-section">
              <h4>Cámaras Locales (USB/Integradas)</h4>
              {availableCameras.length === 0 ? (
                <p className="no-cameras">No se detectaron cámaras locales. Conecte una cámara USB o use una cámara IP.</p>
              ) : (
                <div className="camera-list">
                  {availableCameras.map((cam) => (
                    <button
                      key={cam.index}
                      className={`camera-option ${telemetry.camera?.type === 'local' && telemetry.camera?.index === cam.index ? 'active' : ''}`}
                      onClick={() => handleLocalCameraSelect(cam.index)}
                    >
                      <span className="camera-option-name">{cam.name}</span>
                      <span className="camera-option-detail">Índice: {cam.index} • {cam.width}x{cam.height}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="camera-divider">o</div>

            <div className="camera-section">
              <h4>Cámara IP / Móvil (IP Webcam, DroidCam, etc.)</h4>
              <p className="camera-section-desc">
                Ingrese la URL de streaming de su móvil. Ejemplos:
                <br />
                <code>http://192.168.1.100:8080/video</code> (IP Webcam)
                <br />
                <code>http://192.168.1.100:4747/video</code> (DroidCam)
              </p>
              <div className="ip-camera-form">
                <div className="form-group">
                  <label>Nombre:</label>
                  <input
                    type="text"
                    value={ipCameraName}
                    onChange={(e) => setIpCameraName(e.target.value)}
                    placeholder="Mi Móvil"
                  />
                </div>
                <div className="form-group">
                  <label>URL de Streaming:</label>
                  <input
                    type="text"
                    value={ipCameraUrl}
                    onChange={(e) => setIpCameraUrl(e.target.value)}
                    placeholder="http://192.168.x.x:port/video"
                  />
                </div>
                <button 
                  className="control-btn primary"
                  onClick={handleIpCameraTest}
                  disabled={testingCamera || !ipCameraUrl.trim()}
                >
                  {testingCamera ? 'Probando...' : 'Conectar y Usar'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <footer className="app-footer">
        <p>Sistema de videovigilancia adaptado para control de accesos e inspección de equipos de protección individual en obra.</p>
      </footer>
    </div>
  );
}

export default App;