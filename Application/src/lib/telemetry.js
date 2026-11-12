// Funciones puras del panel: lo que se muestra a partir de la telemetría, sin React ni el navegador.
// Tienen pruebas en telemetry.test.js (npm test).

/**
 * La alarma (sonido, borde rojo y aviso) responde a la infracción confirmada en varios fotogramas, no a un
 * fotograma suelto. Un backend anterior no envía ese campo: entonces vale cualquier infracción.
 */
export function hasAlarm(data) {
  return Boolean(data.violation_confirmed ?? data.violations > 0);
}

/** Color del porcentaje de cumplimiento, con contraste suficiente sobre la tarjeta (los de App.css). */
export function getSafetyColor(score) {
  if (score >= 90) return "#34d399";
  if (score >= 60) return "#fbbf24";
  return "#fecaca";
}

/** Nombre de la cámara que se muestra en el botón de la fuente de vídeo. */
export function getCameraDisplayName(camera) {
  if (!camera) return "Cámara no seleccionada";
  if (camera.type === "local") {
    return camera.name || `Cámara Local ${camera.index}`;
  }
  return camera.name || camera.url;
}

/** Fecha de hoy como AAAA-MM-DD en hora local (el formato del campo `date` de una infracción). */
export function localDate(now = new Date()) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

/** Momento de una infracción: solo la hora si es de hoy; con la fecha si es de otro día. */
export function infractionTime(inf, today = localDate()) {
  return inf.date && inf.date !== today ? `${inf.date} ${inf.timestamp}` : inf.timestamp;
}

/** Texto del contador de la galería, que muestra como mucho las 12 últimas infracciones. */
export function galleryCount(n) {
  if (n <= 0) return "";
  return n === 1 ? "La más reciente" : `Las ${n} más recientes`;
}
