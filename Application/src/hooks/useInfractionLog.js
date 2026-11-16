import { useState } from 'react';
import { getJson } from '../lib/api';

/**
 * Registro completo de infracciones (hasta 100). La telemetría solo trae las 12 últimas; el resto se pide
 * a /api/infractions cuando el usuario quiere verlo. `all` es null mientras se muestran solo las recientes.
 * La lista completa es la del momento en que se pidió: no se actualiza sola.
 */
export function useInfractionLog() {
  const [all, setAll] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const showAll = async () => {
    setError(null);
    setLoading(true);
    try {
      const data = await getJson("/api/infractions");
      setAll(Array.isArray(data.infractions) ? data.infractions : []);
    } catch {
      setError("No se pudo cargar el registro completo");
    } finally {
      setLoading(false);
    }
  };

  const showRecent = () => {
    setError(null);
    setAll(null);
  };

  return { all, loading, error, showAll, showRecent };
}
