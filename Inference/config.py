"""Lo que se configura sin tocar el código: las variables de entorno GUARDWATCH_* (véase el README).

Se leen una sola vez, en execute.py, con Settings.from_env(); los demás módulos reciben los valores. La excepción
es GUARDWATCH_NO_VENV, que venv_bootstrap lee antes de importar nada (decide con qué Python se ejecuta el servidor).
"""
import math
import os
from typing import FrozenSet, Mapping, NamedTuple, Tuple


def retention_days(raw: str) -> float:
    """Valida GUARDWATCH_SNAPSHOT_DAYS: un número de días mayor que cero.

    Con 0 o un valor negativo la limpieza borraría todas las capturas, incluidas las recién guardadas, y con
    "nan" no borraría ninguna.
    """
    try:
        days = float(raw)
    except ValueError:
        days = math.nan
    if not (math.isfinite(days) and days > 0):
        raise ValueError(f"GUARDWATCH_SNAPSHOT_DAYS={raw!r} no es válido: debe ser un número de días mayor "
                         "que 0 (por ejemplo, 30)")
    return days


def _names(raw: str) -> Tuple[str, ...]:
    """Una lista separada por comas, sin espacios ni elementos vacíos."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


class Settings(NamedTuple):
    profile: str               # GUARDWATCH_PROFILE: modo con el que arranca (carpeta de Inference/profiles)
    # GUARDWATCH_SNAPSHOT_DAYS: días que se conservan las capturas. La Ley Orgánica 3/2018 (art. 22) fija un
    # máximo de un mes para las imágenes de videovigilancia salvo que se necesiten para acreditar un hecho
    snapshot_days: float
    origins: Tuple[str, ...]   # GUARDWATCH_ORIGINS: otras direcciones desde las que se sirve el panel (CORS)
    hosts: FrozenSet[str]      # GUARDWATCH_HOSTS: otros nombres con los que se llama a este equipo (cabecera Host)
    listen: str                # GUARDWATCH_HOST: dirección en la que escucha (127.0.0.1: solo este equipo)

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        """Lee y valida las variables; ValueError si alguna no es válida (el servidor no arranca)."""
        return cls(profile=env.get("GUARDWATCH_PROFILE", "casco"),
                   snapshot_days=retention_days(env.get("GUARDWATCH_SNAPSHOT_DAYS", "30")),
                   origins=tuple(o.rstrip("/") for o in _names(env.get("GUARDWATCH_ORIGINS", ""))),
                   hosts=frozenset(h.lower() for h in _names(env.get("GUARDWATCH_HOSTS", ""))),
                   listen=env.get("GUARDWATCH_HOST", "0.0.0.0"))
