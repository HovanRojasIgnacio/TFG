import { useState } from 'react';
import './App.css';
import { useTelemetry } from './hooks/useTelemetry';
import { useAlarm } from './hooks/useAlarm';
import { useCameras } from './hooks/useCameras';
import { useConfidence } from './hooks/useConfidence';
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

  const [telemetry, setTelemetry] = useTelemetry(isActive, (data) => {
    if (soundEnabled && data.active_detections?.no_helmet > 0) {
      beep();
    }
  });

  const confidence = useConfidence(setTelemetry);
  const cameras = useCameras(setTelemetry, () => setCameraModalOpen(false));

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
        <VideoCard
          isActive={isActive}
          onToggleActive={() => setIsActive(!isActive)}
          hasInfraction={telemetry.active_detections?.no_helmet > 0}
          device={telemetry.device}
          weightsFormat={telemetry.weights_format}
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
              setCameraModalOpen(true);
            }}
          />
          <MetricsCard
            isActive={isActive}
            fps={telemetry.fps}
            safetyScore={telemetry.safety_score}
            detections={telemetry.active_detections}
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
        <p>Sistema de videovigilancia adaptado para control de accesos e inspección de equipos de protección individual en obra.</p>
      </footer>
    </div>
  );
}

export default App;
