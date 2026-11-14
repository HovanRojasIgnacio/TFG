import { useState } from 'react';
import Modal from './Modal';
import ErrorMessage from './ErrorMessage';

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
  const isCurrent = (cam) => currentCamera?.type === 'local' && currentCamera?.index === cam.index;

  return (
    <Modal label="Seleccionar fuente de vídeo" className="camera-modal" onClose={onClose}>
      <h2>Seleccionar Fuente de Vídeo</h2>

      {error && <ErrorMessage>{error}</ErrorMessage>}

      {testing && <p className="slider-desc" role="status">Comprobando las cámaras…</p>}

      <div className="camera-tab">
        <button
          className="tab-btn"
          onClick={onRefresh}
          disabled={testing}
        >
          <span aria-hidden="true">🔄 </span>Actualizar cámaras locales
        </button>
      </div>

      <div className="camera-section">
        <h3>
          Cámaras Locales (<abbr title="Universal Serial Bus, el conector habitual de las cámaras web">USB</abbr>/Integradas)
        </h3>
        {cameras.length === 0 ? (
          <p className="no-cameras">No se ha encontrado ninguna cámara local. Conecte una cámara USB o use una cámara IP.</p>
        ) : (
          <div className="camera-list">
            {cameras.map((cam) => (
              <button
                key={cam.index}
                className={`camera-option ${isCurrent(cam) ? 'active' : ''}`}
                aria-pressed={isCurrent(cam)}
                disabled={testing}
                onClick={() => onSelectLocal(cam.index)}
              >
                <span className="camera-option-name">{cam.name}</span>
                <span className="camera-option-detail">Número {cam.index}, {cam.width} × {cam.height} píxeles</span>
                {isCurrent(cam) && <span className="camera-option-current">En uso</span>}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="camera-divider"><span>o</span></div>

      <div className="camera-section">
        <h3>
          Cámara <abbr title="Internet Protocol: cámara conectada por la red">IP</abbr> / Móvil (<span lang="en">IP Webcam</span>, <span lang="en">DroidCam</span>, etc.)
        </h3>
        <p className="camera-section-desc" id="ip-camera-help">
          Escriba la dirección (<abbr title="Uniform Resource Locator: dirección de Internet">URL</abbr>) que muestra
          la aplicación de cámara de su móvil. Ejemplos:
          <br />
          <code>http://192.168.1.100:8080/video</code> (IP Webcam)
          <br />
          <code>http://192.168.1.100:4747/video</code> (DroidCam)
        </p>
        <div className="ip-camera-form">
          <div className="form-group">
            <label htmlFor="ip-camera-name">Nombre:</label>
            <input
              id="ip-camera-name"
              type="text"
              autoComplete="off"
              value={ipCameraName}
              onChange={(e) => setIpCameraName(e.target.value)}
              placeholder="Mi Móvil"
            />
          </div>
          <div className="form-group">
            <label htmlFor="ip-camera-url">Dirección (URL) de la cámara:</label>
            <input
              id="ip-camera-url"
              type="text"
              aria-describedby="ip-camera-help"
              autoComplete="off"
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
            {testing ? 'Probando…' : 'Conectar y Usar'}
          </button>
        </div>
      </div>
    </Modal>
  );
}
