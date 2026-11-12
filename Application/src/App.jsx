import { useState } from 'react';
import './App.css';
import { hasAlarm } from './lib/telemetry';
import { useTelemetry } from './hooks/useTelemetry';
import { useAlarm } from './hooks/useAlarm';
import { useCameras } from './hooks/useCameras';
import { useConfidence } from './hooks/useConfidence';
import { useProfiles } from './hooks/useProfiles';
import VideoCard from './components/VideoCard';
import ControlPanel from './components/ControlPanel';
import MetricsCard from './components/MetricsCard';
import InfractionGallery from './components/InfractionGallery';
import EvidenceModal from './components/EvidenceModal';
import CameraModal from './components/CameraModal';

function App() {
  const [isActive, setIsActive] = useState(true);
  const [activeModalImg, setActiveModalImg] = useState(null);
  const [cameraModalOpen, setCameraModalOpen] = useState(false);

  const { soundEnabled, toggleSound, beep } = useAlarm();

  const [telemetry, setTelemetry, backendOnline] = useTelemetry(isActive, (data) => {
    if (soundEnabled && hasAlarm(data)) {
      beep();
    }
  });

  const confidence = useConfidence(setTelemetry);
  const cameras = useCameras(setTelemetry, () => setCameraModalOpen(false));
  const profiles = useProfiles(backendOnline, telemetry.profile_loading);

  return (
    <div className="app-container">
      <a className="skip-link" href="#contenido">Saltar al contenido</a>
      <header className="app-header">
        <div className="header-logo">
          <span className="shield-icon" aria-hidden="true">🛡️</span>
          <div>
            <h1 lang="en">GuardWatch AI</h1>
            <p className="subtitle">
              Inspección de <abbr title="Equipos de Protección Individual">EPIs</abbr> en Tiempo Real — YOLO26s
            </p>
          </div>
        </div>
        <div className="header-badge">TRABAJO FIN DE GRADO</div>
      </header>

      {/* Aviso para lectores de pantalla cuando aparece o desaparece una infracción (WCAG 4.1.3) */}
      <p className="sr-only" role="status">
        {!isActive
          ? "Monitorización en pausa."
          : hasAlarm(telemetry)
            ? "Atención: falta un elemento de protección en la imagen."
            : "Sin infracciones en la imagen."}
      </p>

      {isActive && backendOnline === false && (
        <div className="connection-banner" role="alert">
          <span aria-hidden="true">⚠ </span>Sin conexión con el servidor. Los datos pueden no estar al día. Se está reintentando…
        </div>
      )}

      <main className="main-content" id="contenido" tabIndex={-1}>
        <VideoCard
          isActive={isActive}
          onToggleActive={() => setIsActive(!isActive)}
          hasInfraction={isActive && hasAlarm(telemetry)}
          device={telemetry.device}
          weightsFormat={telemetry.weights_format}
          backendOnline={backendOnline}
        />

        <div className="right-panel">
          <ControlPanel
            confThreshold={confidence.pendingConf ?? telemetry.conf_threshold}
            onConfidenceChange={confidence.onChange}
            soundEnabled={soundEnabled}
            onToggleSound={toggleSound}
            camera={telemetry.camera}
            cameraError={telemetry.camera_error}
            onOpenCameraModal={() => {
              cameras.clearSelectError();
              cameras.loadCameras();
              setCameraModalOpen(true);
            }}
            profiles={profiles}
            onSelectProfile={profiles.selectProfile}
          />
          <MetricsCard
            isActive={isActive}
            fps={telemetry.fps}
            safetyScore={telemetry.safety_score}
            items={telemetry.items}
            itemNames={telemetry.profile?.items}
            neutral={telemetry.neutral}
          />
        </div>
      </main>

      <InfractionGallery infractions={telemetry.infractions} onView={setActiveModalImg} />

      {activeModalImg && (
        <EvidenceModal imageUrl={activeModalImg} onClose={() => setActiveModalImg(null)} />
      )}

      {cameraModalOpen && (
        <CameraModal
          cameras={cameras.availableCameras}
          currentCamera={telemetry.camera}
          testing={cameras.testingCamera}
          error={cameras.selectError}
          onRefresh={cameras.refreshCameras}
          onSelectLocal={cameras.selectLocalCamera}
          onSelectIp={cameras.selectIpCamera}
          onClose={() => setCameraModalOpen(false)}
        />
      )}

      <footer className="app-footer">
        <p>Sistema de videovigilancia que comprueba el uso de los equipos de protección individual en una obra.</p>
      </footer>
    </div>
  );
}

export default App;
