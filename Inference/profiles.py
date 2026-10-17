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

state: ok (lleva el elemento), missing (infracción), neutral (se muestra pero no cuenta para el cumplimiento) o
ignore (ni se dibuja ni se cuenta). level (opcional, "head" o "person") marca las clases de un modelo que ve al
mismo operario en varios niveles, para contarlo una sola vez (véase counting.py). conf (opcional) es el umbral
de confianza recomendado para el modelo.
"""
import json
import os
from typing import Dict, Mapping, NamedTuple, Optional

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
STATES = ("ok", "missing", "neutral", "ignore")
LEVELS = {"head": 0, "person": 1}


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

    def rule(self, class_name: str) -> ClassRule:
        return self.classes[class_name]


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
    return Profile(profile_id, data.get("name") or profile_id, items, classes, data.get("source", ""),
                   float(conf) if conf is not None else None, base_dir)


def load(profile_id: str, profiles_dir: str = PROFILES_DIR) -> Profile:
    """Lee profiles/<id>/profile.json. El id tiene que ser una carpeta instalada (nunca una ruta)."""
    found = discover(profiles_dir)
    if profile_id not in found:
        raise KeyError(f"Perfil desconocido: {profile_id!r}. Disponibles: {', '.join(found) or 'ninguno'}")
    base_dir = found[profile_id]
    with open(os.path.join(base_dir, "profile.json"), encoding="utf-8") as f:
        return parse(profile_id, json.load(f), base_dir)


def check_model(profile: Profile, model_names: Mapping[int, str]) -> None:
    """Comprueba que el perfil describe todas las clases del modelo (evita unos pesos equivocados en la carpeta)."""
    missing = [n for n in model_names.values() if n not in profile.classes]
    if missing:
        raise ValueError(f"Perfil {profile.id}: el modelo tiene clases que profile.json no describe: "
                         f"{', '.join(missing)}")
