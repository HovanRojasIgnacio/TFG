import React, { useState } from 'react';
import './App.css';

function App() {
  const [isActive, setIsActive] = useState(true);

  // URL que apunta al servidor local de FastAPI
  const streamUrl = "http://localhost:8000/api/stream";

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Panel de Vigilancia de EPIs</h1>
        <p className="subtitle">Visualizador Web en Tiempo Real — Modelo YOLO26s</p>
      </header>

      <main className="main-content">
        {/* Tarjeta de Vídeo */}
        <div className="card video-card">
          <div className="card-header">
            <h3>Feed de Cámara en Vivo</h3>
            <span className={`status-indicator ${isActive ? 'active' : 'inactive'}`}>
              {isActive ? '● TRANSMITIENDO' : '○ SISTEMA APAGADO'}
            </span>
          </div>

          <div className="video-viewport">
            {isActive ? (
              <img 
                src={streamUrl} 
                alt="Detección en tiempo real" 
                className="stream-img"
              />
            ) : (
              <div className="viewport-placeholder">
                <p>La señal de vídeo está pausada.</p>
              </div>
            )}
          </div>

          <div className="card-footer">
            <button 
              className={`control-btn ${isActive ? 'danger' : 'success'}`}
              onClick={() => setIsActive(!isActive)}
            >
              {isActive ? 'Detener Inspección' : 'Iniciar Inspección'}
            </button>
          </div>
        </div>

        {/* Tarjeta de Datos e Información de tu TFG */}
        <div className="card telemetry-card">
          <h3>Configuración del Sistema</h3>
          <div className="divider" />
          
          <div className="metric-group">
            <div className="metric-row">
              <span className="label">Aceleración de Hardware:</span>
              <span className="value highlight">RTX 5060 (CUDA)</span>
            </div>
            <div className="metric-row">
              <span className="label">Umbral de Confianza (Conf):</span>
              <span className="value">0.325</span>
            </div>
            <div className="metric-row">
              <span className="label">Modo de Inferencia:</span>
              <span className="value">FP16 (Half Precision)</span>
            </div>
          </div>

          <div className="legend-box">
            <h4>Código de Colores de Riesgo</h4>
            <ul>
              <li><span className="dot red"></span> **Rojo**: Infracción detectada (Sin Casco / Cabeza descubierta)</li>
              <li><span className="dot green"></span> **Verde**: Protección adecuada (Casco detectado)</li>
              <li><span className="dot orange"></span> **Naranja**: Elementos neutros analizados</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;