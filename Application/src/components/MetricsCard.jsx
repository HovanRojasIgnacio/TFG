function getSafetyColor(score) {
  if (score >= 90) return "#10b981";
  if (score >= 60) return "#f59e0b";
  return "#ef4444";
}

export default function MetricsCard({ isActive, fps, safetyScore, items, itemNames, neutral }) {
  // Una fila por elemento de protección del modo actual (casco, chaleco...), en el orden del perfil
  const rows = Object.entries(items || {});

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
        {rows.map(([item, count]) => (
          <div className={`counter-item ${count.missing > 0 ? "border-red" : "border-green"}`} key={item}>
            <span>{itemNames?.[item] ?? item}:</span>
            <span>
              <span className="counter-val text-green">{count.ok}</span> correctos
              {" / "}
              <span className="counter-val text-red">{count.missing}</span> sin
            </span>
          </div>
        ))}
        <div className="counter-item border-orange">
          <span>Detecciones Neutrales:</span>
          <span className="counter-val text-orange">{neutral ?? 0}</span>
        </div>
      </div>
    </div>
  );
}
