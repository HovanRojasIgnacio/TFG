import { getSafetyColor } from '../lib/telemetry';

export default function MetricsCard({ isActive, fps, safetyScore, items, itemNames, neutral }) {
  // Una fila por elemento de protección del modo actual (casco, chaleco...), en el orden del perfil
  const rows = Object.entries(items || {});

  return (
    <section className="card stats-card" aria-labelledby="metrics-title">
      <h2 id="metrics-title">Métricas de Inferencia</h2>
      <div className="divider" />

      {!isActive && (
        <p className="slider-desc paused-note">En pausa: estos datos son los de antes de suspender y no se actualizan.</p>
      )}

      <div className="metrics-grid">
        <div className="metric-box">
          <span className="metric-title">
            <abbr title="Fotogramas por segundo">FPS</abbr> Inferencia
          </span>
          <span className="metric-value text-emerald">{isActive ? fps : 0}</span>
        </div>
        <div className="metric-box">
          <span className="metric-title">
            Cumplimiento <abbr title="Equipos de Protección Individual">EPI</abbr>
          </span>
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
    </section>
  );
}
