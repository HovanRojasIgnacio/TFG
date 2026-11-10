export default function InfractionGallery({ infractions, onView }) {
  const list = infractions || [];

  return (
    <section className="card infractions-section" aria-labelledby="gallery-title">
      <div className="section-header">
        <h2 id="gallery-title">Registro de Infracciones Guardadas</h2>
        <span className="log-count">{list.length} Guardadas</span>
      </div>
      <div className="divider" />

      {list.length === 0 ? (
        <div className="no-infractions">
          <p><span aria-hidden="true">✅ </span>Ninguna infracción registrada en la sesión actual.</p>
        </div>
      ) : (
        <div className="infractions-grid">
          {list.slice().reverse().map((inf) => (
            <div className="infraction-item-card" key={inf.id}>
              <div className="item-thumbnail">
                <img src={inf.image_url} alt={`Imagen de la infracción de las ${inf.timestamp}`} loading="lazy" />
                <span className="item-badge-time">{inf.timestamp}</span>
              </div>
              <div className="item-info">
                <div className="info-top">
                  {inf.profile && <span className="info-profile">{inf.profile}</span>}
                  <span className="info-conf">Confianza: {(inf.confidence * 100).toFixed(0)} %</span>
                </div>
                <p>{inf.details}</p>
                <button
                  className="view-btn"
                  aria-label={`Ver Evidencia de las ${inf.timestamp}`}
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
