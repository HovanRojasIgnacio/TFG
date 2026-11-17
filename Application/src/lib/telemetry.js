// Funciones puras del panel: lo que se muestra a partir de la telemetría, sin React ni el navegador.
// Tienen pruebas en telemetry.test.js (npm test).

/**
 * La alarma (sonido, borde rojo y aviso) responde a la infracción confirmada en varios fotogramas, no a un
 * fotograma suelto.
 */
export function hasAlarm(data) {
  return data.violation_confirmed === true;
}

/** Color del porcentaje de cumplimiento, con contraste suficiente sobre la tarjeta (los de App.css). */
export function getSafetyColor(score) {
  if (score >= 90) return "#34d399";
  if (score >= 60) return "#fbbf24";
  return "#fecaca";
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

/**
 * Texto del contador de la galería: las últimas infracciones (como mucho 12) o, con `all`, el registro
 * completo que conserva el servidor.
 */
export function galleryCount(n, all = false) {
  if (n <= 0) return "";
  if (all) return n === 1 ? "La única guardada" : `Las ${n} guardadas`;
  return n === 1 ? "La más reciente" : `Las ${n} más recientes`;
}
