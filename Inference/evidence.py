"""Evidencias: capturas de las infracciones y registro (infractions.json) que sobrevive a los reinicios."""
import json
import logging
import os
import time
from typing import Dict, List
import profiles

logger = logging.getLogger("guardwatch")

current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
infractions_dir = os.path.join(static_dir, "infractions")
os.makedirs(infractions_dir, exist_ok=True)

# Días que se conservan las capturas de infracciones. La Ley Orgánica 3/2018 (art. 22) fija un máximo de un
# mes para las imágenes de videovigilancia salvo que se necesiten para acreditar un hecho.
SNAPSHOT_RETENTION_DAYS = float(os.environ.get("GUARDWATCH_SNAPSHOT_DAYS", "30"))
LOG_SIZE = 100  # Infracciones que conserva el registro
# El registro se guarda fuera de /static: contiene los nombres de las capturas, que no deben poder adivinarse
LOG_FILE = os.path.join(current_dir, "infractions.json")


def purge_old_snapshots() -> set:
    """Borra las capturas con más de SNAPSHOT_RETENTION_DAYS días y devuelve sus nombres.

    El registro solo conserva las últimas infracciones, así que sin esta limpieza las capturas que ya no están
    en él se quedarían en el disco para siempre.
    """
    limit = time.time() - SNAPSHOT_RETENTION_DAYS * 86400
    removed = set()
    os.makedirs(infractions_dir, exist_ok=True)  # Por si alguien borró la carpeta con el servidor en marcha
    for entry in os.scandir(infractions_dir):
        try:
            if entry.is_file() and entry.name.lower().endswith(".jpg") and entry.stat().st_mtime < limit:
                os.remove(entry.path)
                removed.add(entry.name)
        except OSError as e:  # Borrada o bloqueada por otro proceso entre el listado y el borrado
            logger.warning("No se pudo borrar la captura %s: %s", entry.name, e)
    if removed:
        logger.info("Borradas %d capturas con más de %g días", len(removed), SNAPSHOT_RETENTION_DAYS)
    return removed


def load_log() -> List[Dict]:
    """Lee el registro de infracciones guardado y descarta las entradas cuya captura ya no existe."""
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError("no es una lista")
        return [e for e in entries if isinstance(e, dict) and os.path.isfile(
            os.path.join(infractions_dir, os.path.basename(str(e.get("image_path", "")))))][-LOG_SIZE:]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, AttributeError, RecursionError) as e:
        logger.warning("No se pudo leer el registro de infracciones %s: %s", LOG_FILE, e)
        return []


def save_log(entries) -> None:
    """Guarda el registro de infracciones (se escribe aparte y se sustituye, para no dejarlo a medias)."""
    try:
        tmp = LOG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(list(entries), f, ensure_ascii=False)
        os.replace(tmp, LOG_FILE)
    except OSError as e:  # Se ejecuta en el hilo de vídeo: un fallo de disco no debe terminarlo
        logger.error("No se pudo guardar el registro de infracciones: %s", e)


def missing_details(profile: profiles.Profile, items: Dict[str, Dict[str, int]]) -> str:
    """Texto de una infracción con los elementos que faltan, p. ej. "Falta: casco ×1, chaleco ×2"."""
    parts = [f"{profile.items[item].lower()} ×{c['missing']}" for item, c in items.items() if c["missing"]]
    return "Falta: " + ", ".join(parts)
