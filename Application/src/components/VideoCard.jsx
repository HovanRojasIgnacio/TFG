import { useState, useEffect } from 'react';
import { API_BASE, STREAM_RETRY_MS } from '../config';

export default function VideoCard({ isActive, onToggleActive, hasInfraction, device, weightsFormat, backendOnline }) {
  const [streamError, setStreamError] = useState(false);

  // Si el backend se cae, el navegador no avisa con onError ni onLoad: la imagen MJPEG se queda en el
  // último frame. Cuando el backend vuelve a responder se monta una imagen nueva con otra URL, que abre
  // otra conexión. (Ajuste de estado durante el render, el patrón que recomienda React.)
  const [streamKey, setStreamKey] = useState(0);
  const [lastOnline, setLastOnline] = useState(backendOnline);
  if (backendOnline !== lastOnline) {
    setLastOnline(backendOnline);
    if (backendOnline && lastOnline === false) setStreamKey(k => k + 1);
  }

  // Si el stream falla (por ejemplo, el backend aún arranca), se reintenta solo mientras la monitorización esté activa
  useEffect(() => {
    if (!streamError || !isActive) return;
    const timer = setTimeout(() => setStreamError(false), STREAM_RETRY_MS);
    return () => clearTimeout(timer);
  }, [streamError, isActive]);

  return (
    <section className={`card video-card ${hasInfraction ? 'border-alarm' : ''}`} aria-labelledby="video-title">
      <div className="card-header">
        <div className="header-left">
          <span className={`pulse-dot ${isActive && !streamError ? 'online' : 'offline'}`} aria-hidden="true"></span>
          <h2 id="video-title">Vídeo Monitorizado en Vivo</h2>
          <span className="sr-only">{isActive && !streamError ? '(en directo)' : '(sin vídeo)'}</span>
        </div>
        <div className="header-right">
          <span className="hardware-pill" title="Dispositivo que ejecuta el modelo">
            <span className="sr-only">Dispositivo: </span>{device}
          </span>
          <span className="format-pill" title="Formato del modelo">
            <span className="sr-only">Formato del modelo: </span>{weightsFormat}
          </span>
        </div>
      </div>

      <div className="video-viewport">
        {isActive && !streamError ? (
          <img
            key={streamKey}
            src={`${API_BASE}/api/stream?c=${streamKey}`}
            alt="Vídeo en directo de la cámara, con un recuadro en cada detección"
            className="stream-img"
            onError={() => setStreamError(true)}
          />
        ) : (
          <div className="viewport-placeholder">
            <span className="placeholder-icon" aria-hidden="true">📷</span>
            <p role="status">{streamError ? "Error al conectar con la señal del servidor. Reintentando automáticamente…" : "La monitorización está en pausa."}</p>
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
            onToggleActive();
            setStreamError(false);
          }}
        >
          {isActive ? 'Suspender Monitoreo' : 'Iniciar Monitoreo'}
        </button>
      </div>
    </section>
  );
}
