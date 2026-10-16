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
    <div className={`card video-card ${hasInfraction ? 'border-alarm' : ''}`}>
      <div className="card-header">
        <div className="header-left">
          <span className={`pulse-dot ${isActive && !streamError ? 'online' : 'offline'}`}></span>
          <h3>Vídeo Monitorizado en Vivo</h3>
        </div>
        <div className="header-right">
          <span className="hardware-pill">{device}</span>
          <span className="format-pill">{weightsFormat}</span>
        </div>
      </div>

      <div className="video-viewport">
        {isActive && !streamError ? (
          <img
            key={streamKey}
            src={`${API_BASE}/api/stream?c=${streamKey}`}
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
            onToggleActive();
            setStreamError(false);
          }}
        >
          {isActive ? 'Suspender Monitoreo' : 'Iniciar Monitoreo'}
        </button>
      </div>
    </div>
  );
}
