import { infractionTime, galleryCount, localDate } from '../lib/telemetry';
import ErrorMessage from './ErrorMessage';

export default function InfractionGallery({
  infractions, total, showingAll, error, onShowAll, onShowRecent, onView
}) {
  const list = infractions || [];
  // El registro guarda más infracciones de las que se muestran al principio: se pueden ver todas
  const hasOlder = !showingAll && (total ?? 0) > list.length;
  // El registro se conserva entre reinicios: la fecha se muestra en las infracciones que no son de hoy
  const today = localDate();
  const when = (inf) => infractionTime(inf, today);

  return (
    <section className="card infractions-section" aria-labelledby="gallery-title">
      <div className="section-header">
        <h2 id="gallery-title">Registro de Infracciones Guardadas</h2>
        {/* Al principio se muestran las RECENT_INFRACTIONS últimas de las 100 que conserva el backend */}
        {list.length > 0 && (
          <span className="log-count">{galleryCount(list.length, showingAll)}</span>
        )}
      </div>
      <div className="divider" />

      {(hasOlder || showingAll) && (
        <div className="log-actions">
          {showingAll ? (
            <button className="tab-btn log-toggle" onClick={onShowRecent}>Ver solo las más recientes</button>
          ) : (
            <button className="tab-btn log-toggle" onClick={onShowAll}>{`Ver las ${total} guardadas`}</button>
          )}
        </div>
      )}
      {error && <ErrorMessage>{error}</ErrorMessage>}

      {list.length === 0 ? (
        <div className="no-infractions">
          <p><span aria-hidden="true">✅ </span>Ninguna infracción registrada.</p>
        </div>
      ) : (
        <div className="infractions-grid">
          {list.slice().reverse().map((inf) => (
            <div className="infraction-item-card" key={inf.id}>
              <div className="item-thumbnail">
                <img src={inf.image_url} alt={`Imagen de la infracción de las ${when(inf)}`} loading="lazy" />
                <span className="item-badge-time">{when(inf)}</span>
              </div>
              <div className="item-info">
                <div className="info-top">
                  {inf.profile && <span className="info-profile">{inf.profile}</span>}
                  <span className="info-conf">Confianza: {(inf.confidence * 100).toFixed(0)} %</span>
                </div>
                <p>{inf.details}</p>
                <button
                  className="view-btn"
                  aria-label={`Ver Evidencia de las ${when(inf)}`}
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
