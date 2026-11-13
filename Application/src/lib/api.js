// Peticiones al backend: dirección, tiempo máximo y JSON en un solo sitio.
import { API_BASE, REQUEST_TIMEOUT_MS } from '../config.js';

// Cancela la petición al pasar `timeoutMs` y, si se da, también cuando se cancele `signal`
function withTimeout(timeoutMs, signal) {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

/** Devuelve el JSON de la respuesta. Lanza un error si falla la red, caduca o el estado no es 2xx. */
export async function getJson(path, { timeoutMs = REQUEST_TIMEOUT_MS, signal } = {}) {
  const res = await fetch(`${API_BASE}${path}`, { signal: withTimeout(timeoutMs, signal) });
  if (!res.ok) throw new Error(`Error ${res.status} en ${path}`);
  return res.json();
}

/**
 * Envía `body` como JSON y devuelve { ok, status, data }; `data` es el JSON de la respuesta
 * ({} si no es JSON). Un estado de error (400, 409…) no lanza: solo falla la red o el tiempo.
 */
export async function postJson(path, body, { timeoutMs = REQUEST_TIMEOUT_MS } = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: withTimeout(timeoutMs)
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}
