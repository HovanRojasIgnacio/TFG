// Configurable con VITE_API_BASE (por ejemplo en un archivo .env); por defecto, el backend local
export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export const POLL_INTERVAL_MS = 500;       // Frecuencia de consulta de /api/stats
export const ALARM_MIN_INTERVAL_MS = 1500; // Tiempo mínimo entre pitidos de la alarma
export const CONF_DEBOUNCE_MS = 250;       // Espera tras mover el deslizador antes de enviar el umbral
export const STREAM_RETRY_MS = 3000;       // Espera antes de reintentar el stream tras un error
export const REQUEST_TIMEOUT_MS = 3000;    // Tiempo máximo de una petición a la API (sin él, una conexión
                                           // caída sin aviso dejaría la petición pendiente durante minutos)
export const DEFAULT_CONF = 0.325;
