import { useState } from 'react';
import Modal from './Modal';

export default function CameraModal({
  cameras,
  currentCamera,
  testing,
  error,
  onRefresh,
  onSelectLocal,
  onSelectIp,
  onClose
}) {
  const [ipCameraUrl, setIpCameraUrl] = useState("");
  const [ipCameraName, setIpCameraName] = useState("");

  return (
    <Modal label="Seleccionar fuente de vídeo" className="camera-modal" onClose={onClose}>
      <h3>Seleccionar Fuente de Vídeo</h3>

      {error && (
        <div className="camera-error" role="alert">
          ⚠ {error}
        </div>
      )}

      <div className="camera-tab">
        <button
          className="tab-btn"
          onClick={onRefresh}
        >
          🔄 Actualizar cámaras locales
        </button>
      </div>

      <div className="camera-section">
        <h4>Cámaras Locales (USB/Integradas)</h4>
        {cameras.length === 0 ? (
          <p className="no-cameras">No se detectaron cámaras locales. Conecte una cámara USB o use una cámara IP.</p>
        ) : (
          <div className="camera-list">
            {cameras.map((cam) => (
              <button
                key={cam.index}
                className={`camera-option ${currentCamera?.type === 'local' && currentCamera?.index === cam.index ? 'active' : ''}`}
                onClick={() => onSelectLocal(cam.index)}
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
            onClick={() => onSelectIp(ipCameraUrl, ipCameraName)}
            disabled={testing || !ipCameraUrl.trim()}
          >
            {testing ? 'Probando...' : 'Conectar y Usar'}
          </button>
        </div>
      </div>
    </Modal>
  );
}
