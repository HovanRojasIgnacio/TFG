export default function InfractionGallery({ infractions, onView }) {
  const list = infractions || [];

  return (
    <section className="card infractions-section">
      <div className="section-header">
        <h3>Registro de Infracciones Guardadas (Snapshots)</h3>
        <span className="log-count">{list.length} Guardadas</span>
      </div>
      <div className="divider" />

      {list.length === 0 ? (
        <div className="no-infractions">
          <p>✅ Ninguna infracción registrada en la sesión actual. Cumplimiento normativo estricto.</p>
        </div>
      ) : (
        <div className="infractions-grid">
          {list.slice().reverse().map((inf) => (
            <div className="infraction-item-card" key={inf.id}>
              <div className="item-thumbnail">
                <img src={inf.image_url} alt="Evidencia de infracción" loading="lazy" />
                <span className="item-badge-time">{inf.timestamp}</span>
              </div>
              <div className="item-info">
                <div className="info-top">
                  {inf.profile && <span className="info-profile">{inf.profile}</span>}
                  <span className="info-conf">Conf: {(inf.confidence * 100).toFixed(0)}%</span>
                </div>
                <p>{inf.details}</p>
                <button
                  className="view-btn"
                  onClick={() => onView(inf.image_url)}
                >
                  Ver Evidencia
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
