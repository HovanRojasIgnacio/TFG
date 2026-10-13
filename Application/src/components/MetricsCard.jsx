function getSafetyColor(score) {
  if (score >= 90) return "#10b981";
  if (score >= 60) return "#f59e0b";
  return "#ef4444";
}

export default function MetricsCard({ isActive, fps, safetyScore, detections }) {
  return (
    <div className="card stats-card">
      <h3>Métricas de Inferencia</h3>
      <div className="divider" />

      <div className="metrics-grid">
        <div className="metric-box">
          <span className="metric-title">FPS Inferencia</span>
          <span className="metric-value text-emerald">{isActive ? fps : 0}</span>
        </div>
        <div className="metric-box">
          <span className="metric-title">Cumplimiento EPI</span>
          <span className="metric-value" style={{ color: getSafetyColor(safetyScore) }}>
            {safetyScore}%
          </span>
        </div>
      </div>

      <div className="counter-list">
        <div className="counter-item border-green">
          <span>Cascos Correctos:</span>
          <span className="counter-val text-green">{detections?.helmet ?? 0}</span>
        </div>
        <div className="counter-item border-red">
          <span>Infracciones Críticas:</span>
          <span className="counter-val text-red">{detections?.no_helmet ?? 0}</span>
        </div>
        <div className="counter-item border-orange">
          <span>Rostros Neutrales:</span>
          <span className="counter-val text-orange">{detections?.face ?? 0}</span>
        </div>
      </div>
    </div>
  );
}
