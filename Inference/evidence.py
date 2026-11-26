"""Evidencias: capturas de las infracciones y registro (infractions.json) que sobrevive a los reinicios.

EvidenceRecorder decide cuándo se guarda una captura, la añade al registro, borra la que sale de él y hace la
limpieza periódica de las capturas antiguas.
"""
import json
import logging
import os
import secrets
import time
from collections import deque
from typing import Dict, List, Mapping, NamedTuple, Optional
import cv2
import profiles

logger = logging.getLogger("guardwatch")

current_dir = os.path.dirname(os.path.abspath(__file__))
INFRACTIONS_DIR = os.path.join(current_dir, "static", "infractions")  # Las capturas (las sirve execute.py)
# El registro se guarda fuera de /static: contiene los nombres de las capturas, que no deben poder adivinarse
LOG_FILE = os.path.join(current_dir, "infractions.json")
LOG_SIZE = 100  # Infracciones que conserva el registro
SAVE_COOLDOWN = 4.0  # Segundos mínimos entre dos capturas de infracciones
# Mientras la misma infracción sigue en la imagen se guarda una captura cada REPEAT_COOLDOWN segundos: una cada
# SAVE_COOLDOWN llenaba el registro (100 entradas) en menos de siete minutos y borraba todas las anteriores
REPEAT_COOLDOWN = 60.0
PURGE_INTERVAL = 3600.0  # Cada cuánto se revisan las capturas antiguas mientras el servidor está en marcha (s)


class Infraction(NamedTuple):
    """Una entrada del registro: así se guarda en infractions.json y así la devuelve la API."""
    id: int             # Milisegundos desde 1970: único y creciente
    date: str           # AAAA-MM-DD
    timestamp: str      # HH:MM:SS
    confidence: float   # La de la infracción más fiable del frame
    image: str          # Nombre del archivo de la captura, en la carpeta del EvidenceRecorder
    profile: str        # Nombre del modo con el que se detectó
    details: str        # Lo que faltaba, p. ej. "Falta: casco ×1"

    @classmethod
    def from_json(cls, entry: Mapping) -> Optional["Infraction"]:
        """La entrada leída de infractions.json, o None si no es una infracción."""
        try:
            entry = dict(entry)
            # Los registros anteriores guardaban la dirección de la captura (/static/infractions/<nombre>)
            entry.setdefault("image", os.path.basename(str(entry.get("image_path", ""))))
            return cls(**{field: entry[field] for field in cls._fields})
        except (TypeError, ValueError, KeyError):
            return None


def missing_details(profile: profiles.Profile, items: Dict[str, Dict[str, int]]) -> str:
    """Texto de una infracción con los elementos que faltan, p. ej. "Falta: casco ×1, chaleco ×2"."""
    parts = [f"{profile.items[item].lower()} ×{c['missing']}" for item, c in items.items() if c["missing"]]
    return "Falta: " + ", ".join(parts)


class EvidenceRecorder:
    """Capturas de las infracciones confirmadas, en `folder`, y su registro (las últimas log_size), en `log_file`.

    Lo usa solo el hilo de vídeo; la API lee `log` (una deque: leerla mientras se añade una entrada es seguro).
    """

    def __init__(self, folder: str, log_file: str, retention_days: float, log_size: int = LOG_SIZE):
        self.folder = folder
        self.log_file = log_file
        self.retention_days = retention_days
        self._purge_old_snapshots()
        # Las infracciones de ejecuciones anteriores siguen en el panel
        self.log = deque(self._load_log(), maxlen=log_size)
        self.save_cooldown = SAVE_COOLDOWN
        self.repeat_cooldown = REPEAT_COOLDOWN
        self._last_save = 0.0
        self._episode_recorded = False  # La infracción que sigue en la imagen ya tiene su captura
        self._undeleted = set()         # Capturas que salieron del registro y no se pudieron borrar: se reintenta
        self._last_purge = time.time()

    def update(self, frame, profile: profiles.Profile, counts, confirmed: bool, now: float) -> None:
        """Guarda una captura si toca: solo de infracciones confirmadas y de un frame en el que se ven.

        La primera captura de una infracción espera save_cooldown desde la anterior; mientras la misma infracción
        sigue en la imagen, las siguientes esperan repeat_cooldown.
        """
        if not confirmed:
            self._episode_recorded = False
            return
        cooldown = self.repeat_cooldown if self._episode_recorded else self.save_cooldown
        if counts.violations > 0 and now - self._last_save > cooldown:
            self._record(frame, profile, counts, now)
            self._episode_recorded = True

    def reset_episode(self) -> None:
        """La siguiente infracción es nueva (otra cámara, otro modo o se perdió la imagen)."""
        self._episode_recorded = False

    def purge_if_due(self) -> None:
        if time.time() - self._last_purge > PURGE_INTERVAL:
            self.purge()

    def purge(self) -> None:
        """Borra las capturas caducadas y quita del registro las entradas cuya imagen ya no existe."""
        self._last_purge = time.time()
        self._retry_undeleted()
        try:
            removed = self._purge_old_snapshots()
        except OSError as e:
            # Se ejecuta en el único hilo de vídeo: un fallo de disco no debe terminarlo
            logger.error("No se pudieron revisar las capturas antiguas: %s", e)
            return
        if removed:
            self.log = deque((i for i in self.log if i.image not in removed), maxlen=self.log.maxlen)
            self._save_log()

    def _purge_old_snapshots(self) -> set:
        """Borra las capturas con más de retention_days días y devuelve sus nombres.

        El registro solo conserva las últimas infracciones, así que sin esta limpieza las capturas que ya no están
        en él se quedarían en el disco para siempre.
        """
        limit = time.time() - self.retention_days * 86400
        removed = set()
        os.makedirs(self.folder, exist_ok=True)  # Al arrancar o si alguien la borró con el servidor en marcha
        for entry in os.scandir(self.folder):
            try:
                if entry.is_file() and entry.name.lower().endswith(".jpg") and entry.stat().st_mtime < limit:
                    os.remove(entry.path)
                    removed.add(entry.name)
            except OSError as e:  # Borrada o bloqueada por otro proceso entre el listado y el borrado
                logger.warning("No se pudo borrar la captura %s: %s", entry.name, e)
        if removed:
            logger.info("Borradas %d capturas con más de %g días", len(removed), self.retention_days)
        return removed

    def _load_log(self) -> List[Infraction]:
        """Lee el registro de infracciones guardado y descarta las entradas cuya captura ya no existe."""
        try:
            with open(self.log_file, encoding="utf-8") as f:
                entries = json.load(f)
            if not isinstance(entries, list):
                raise ValueError("no es una lista")
            infractions = (Infraction.from_json(e) for e in entries)
            return [i for i in infractions if i is not None and os.path.isfile(os.path.join(self.folder, i.image))]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, AttributeError, RecursionError) as e:
            logger.warning("No se pudo leer el registro de infracciones %s: %s", self.log_file, e)
            return []

    def _save_log(self) -> None:
        """Guarda el registro de infracciones (se escribe aparte y se sustituye, para no dejarlo a medias)."""
        try:
            tmp = self.log_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump([i._asdict() for i in self.log], f, ensure_ascii=False)
            os.replace(tmp, self.log_file)
        except OSError as e:  # Se ejecuta en el hilo de vídeo: un fallo de disco no debe terminarlo
            logger.error("No se pudo guardar el registro de infracciones: %s", e)

    def _retry_undeleted(self) -> None:
        """Reintenta borrar las capturas que salieron del registro y seguían abiertas en otro programa."""
        for name in list(self._undeleted):
            try:
                os.remove(os.path.join(self.folder, name))
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.debug("La captura %s sigue sin poder borrarse: %s", name, e)
                continue
            self._undeleted.discard(name)

    def _record(self, frame, profile: profiles.Profile, counts, now: float) -> None:
        """Guarda la captura y la añade al registro (si está lleno, borra la más antigua)."""
        # Las capturas se sirven sin autenticación: el sufijo aleatorio impide adivinar su dirección
        filename = f"inf_{int(now)}_{secrets.token_hex(8)}.jpg"
        filepath = os.path.join(self.folder, filename)
        self._last_save = now
        if not cv2.imwrite(filepath, frame):
            # Disco lleno o carpeta sin permisos: sin imagen no se registra una entrada con la evidencia rota
            logger.error("No se pudo guardar la captura de la infracción en %s", filepath)
            return
        # El registro guarda las últimas LOG_SIZE; la imagen de la que sale se borra del disco
        if len(self.log) == self.log.maxlen:
            evicted = self.log[0].image
            try:
                os.remove(os.path.join(self.folder, evicted))
            except OSError as e:
                logger.warning("No se pudo borrar la captura %s: %s", evicted, e)
                self._undeleted.add(evicted)

        self.log.append(Infraction(
            id=int(now * 1000),
            date=time.strftime("%Y-%m-%d", time.localtime(now)),
            timestamp=time.strftime("%H:%M:%S", time.localtime(now)),
            confidence=round(counts.highest_violation, 2),
            image=filename,
            profile=profile.name,
            details=missing_details(profile, counts.items),
        ))
        self._save_log()
