"""Perfiles de detección: cada modo (casco, casco y chaleco...) es una carpeta, no código.

Inference/profiles/<id>/ contiene los pesos (best.pt; el motor TensorRT se genera ahí mismo) y profile.json, que
dice qué significa cada clase del modelo:

    {
      "name": "Casco y chaleco",
      "items": {"helmet": "Casco", "vest": "Chaleco"},
      "classes": {
        "Helmet":   {"item": "helmet", "state": "ok"},
        "NoHelmet": {"item": "helmet", "state": "missing"},
        "face":     {"state": "neutral", "label": "Rostro"},
        "person":   {"state": "ignore"}
      },
      "source": "dataset, licencia y atribución"
    }

weights (opcional) dice de dónde descargar best.pt si falta en la carpeta: {"url": "https://...", "sha256":
"..."}; los pesos no están en git (véase runtime.ensure_weights).

state: ok (lleva el elemento), missing (infracción), neutral (se muestra pero no cuenta para el cumplimiento) o
ignore (ni se dibuja ni se cuenta). level (opcional, "head" o "person") marca las clases de un modelo que ve al
mismo operario en varios niveles, para contarlo una sola vez (véase counting.py). conf (opcional) es el umbral
de confianza recomendado para el modelo.
"""
import json
import logging
import os
import re
from typing import Dict, Mapping, NamedTuple, Optional

logger = logging.getLogger("guardwatch")

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
STATES = ("ok", "missing", "neutral", "ignore")
LEVELS = {"head": 0, "person": 1}
CONF_MIN, CONF_MAX = 0.1, 0.95  # Intervalo del umbral de confianza (el del deslizador del panel)
DEFAULT_CONF = 0.325  # Umbral de un modo cuyo profile.json no recomienda ninguno


class ClassRule(NamedTuple):
    state: str
    item: Optional[str] = None
    level: Optional[int] = None   # 0 = cabeza, 1 = persona, None = sin jerarquía
    label: Optional[str] = None   # Texto del recuadro (por defecto, el del elemento o el nombre de la clase)


class Profile(NamedTuple):
    id: str
    name: str
    items: Dict[str, str]          # id del elemento -> nombre para el panel
    classes: Dict[str, ClassRule]
    source: str
    conf: Optional[float]
    base_dir: str
    weights_url: Optional[str] = None     # Dónde descargar best.pt si falta (véase runtime.ensure_weights)
    weights_sha256: Optional[str] = None  # SHA-256 que tiene que tener el archivo descargado

    def rule(self, class_name: str) -> ClassRule:
        return self.classes[class_name]

    def box_label(self, class_name: str) -> Optional[str]:
        """Texto del recuadro de una clase ("OK: Casco", "PELIGRO: Sin Casco", "Rostro"); None si no se dibuja."""
        rule = self.classes[class_name]
        if rule.state == "ignore":
            return None
        if rule.state == "neutral":
            return rule.label or class_name
        label = rule.label or self.items[rule.item]
        return f"OK: {label}" if rule.state == "ok" else f"PELIGRO: Sin {label}"

    def summary(self) -> Dict[str, object]:
        """Lo que el panel necesita del modo (en /api/profiles y /api/stats)."""
        return {"id": self.id, "name": self.name, "items": self.items}

    @property
    def threshold(self) -> float:
        """Umbral con el que se usa el modo: el recomendado en profile.json o, si no hay, DEFAULT_CONF."""
        return self.conf or DEFAULT_CONF


def discover(profiles_dir: str = PROFILES_DIR) -> Dict[str, str]:
    """Perfiles instalados: {id: carpeta}. Solo cuentan las carpetas con profile.json."""
    if not os.path.isdir(profiles_dir):
        return {}
    return {entry.name: entry.path for entry in sorted(os.scandir(profiles_dir), key=lambda e: e.name)
            if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "profile.json"))}


def parse(profile_id: str, data: Mapping, base_dir: str = "") -> Profile:
    """Construye y valida un perfil a partir del contenido de profile.json."""
    items = dict(data.get("items") or {})
    if not items:
        raise ValueError(f"Perfil {profile_id}: 'items' está vacío")
    classes = {}
    for name, spec in (data.get("classes") or {}).items():
        state = spec.get("state")
        if state not in STATES:
            raise ValueError(f"Perfil {profile_id}, clase {name}: state debe ser uno de {STATES}")
        item = spec.get("item")
        if state in ("ok", "missing") and item not in items:
            raise ValueError(f"Perfil {profile_id}, clase {name}: item '{item}' no está en 'items'")
        level = spec.get("level")
        if level is not None and level not in LEVELS:
            raise ValueError(f"Perfil {profile_id}, clase {name}: level debe ser 'head' o 'person'")
        classes[name] = ClassRule(state, item if state in ("ok", "missing") else None,
                                  LEVELS.get(level), spec.get("label"))
    if not classes:
        raise ValueError(f"Perfil {profile_id}: 'classes' está vacío")
    conf = data.get("conf")
    if conf is not None:
        # Fuera del intervalo del deslizador del panel el modo dejaría de detectar (o lo detectaría todo) sin aviso
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not CONF_MIN <= conf <= CONF_MAX:
            raise ValueError(f"Perfil {profile_id}: conf debe ser un número entre {CONF_MIN} y {CONF_MAX}")
        conf = float(conf)
    weights = data.get("weights") or {}
    url, sha256 = weights.get("url"), weights.get("sha256")
    if weights and not (isinstance(url, str) and url.startswith("https://")
                        and isinstance(sha256, str) and re.fullmatch(r"[0-9a-f]{64}", sha256)):
        raise ValueError(f"Perfil {profile_id}: weights necesita una url https:// y el sha256 del archivo")
    return Profile(profile_id, data.get("name") or profile_id, items, classes, data.get("source", ""),
                   conf, base_dir, url, sha256)


def load(profile_id: str, profiles_dir: str = PROFILES_DIR) -> Profile:
    """Lee profiles/<id>/profile.json. El id tiene que ser una carpeta instalada (nunca una ruta)."""
    found = discover(profiles_dir)
    if profile_id not in found:
        raise KeyError(f"Perfil desconocido: {profile_id!r}. Disponibles: {', '.join(found) or 'ninguno'}")
    return _read(profile_id, found[profile_id])


def load_all(profiles_dir: str = PROFILES_DIR) -> Dict[str, Profile]:
    """Los perfiles instalados que se pueden leer, por id. Uno roto se registra y se omite: no oculta los demás."""
    loaded = {}
    for profile_id, base_dir in discover(profiles_dir).items():
        try:
            loaded[profile_id] = _read(profile_id, base_dir)
        except Exception as e:
            logger.warning("Perfil %s no válido: %s", profile_id, e)
    return loaded


def _read(profile_id: str, base_dir: str) -> Profile:
    with open(os.path.join(base_dir, "profile.json"), encoding="utf-8") as f:
        return parse(profile_id, json.load(f), base_dir)


def check_model(profile: Profile, model_names: Mapping[int, str]) -> None:
    """Comprueba que el perfil describe todas las clases del modelo (evita unos pesos equivocados en la carpeta)."""
    missing = [n for n in model_names.values() if n not in profile.classes]
    if missing:
        raise ValueError(f"Perfil {profile.id}: el modelo tiene clases que profile.json no describe: "
                         f"{', '.join(missing)}")
