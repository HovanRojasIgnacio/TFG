"""Agrupa las clases del modelo en las categorías del panel y cuenta cada operario una sola vez.

El modelo desplegado marca a un mismo operario en varios niveles: la persona (person_with_helmet,
person_no_helmet), la cabeza (head_with_helmet, head) y el propio casco (helmet). Si se contara cada
recuadro, un operario con casco sumaría tres "cascos correctos". Por eso solo se cuentan los recuadros
del nivel más detallado presente en el frame: la cabeza, o la persona si no se ha detectado ninguna
cabeza. Con un modelo sin esa jerarquía (por ejemplo, clases helmet / no_helmet) se cuenta cada recuadro.
"""
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

CATEGORIES = ("helmet", "no_helmet", "face")


@lru_cache(maxsize=None)
def class_category(class_name: str) -> str:
    """Categoría del panel de una clase: no_helmet (infracción), helmet (correcto) o face (neutral)."""
    name = class_name.lower()
    if "no_helmet" in name or name == "head":
        return "no_helmet"
    if "with_helmet" in name or name == "helmet":
        return "helmet"
    return "face"


@lru_cache(maxsize=None)
def worker_level(class_name: str):
    """Nivel de la clase en la jerarquía del operario: 0 = cabeza, 1 = persona, None = otro objeto."""
    name = class_name.lower()
    if name == "head" or name.startswith("head_"):
        return 0
    if name.startswith("person_"):
        return 1
    return None


def counted_flags(class_names: Sequence[str]) -> List[bool]:
    """Indica, para cada recuadro del frame, si cuenta en los contadores (un recuadro por operario)."""
    levels = [lvl for lvl in map(worker_level, class_names) if lvl is not None]
    level = min(levels) if levels else None
    flags = []
    for name in class_names:
        if class_category(name) == "face":
            flags.append(not name.lower().startswith("person"))  # "person" sola no es un rostro
        else:
            flags.append(level is None or worker_level(name) == level)
    return flags


def count(class_names: Sequence[str], confs: Sequence[float]) -> Tuple[Dict[str, int], float, List[bool]]:
    """Cuenta operarios con casco, sin casco y rostros.

    Devuelve los contadores, la confianza máxima de una infracción y qué recuadros han contado.
    """
    counts = {c: 0 for c in CATEGORIES}
    highest_infraction = 0.0
    flags = counted_flags(class_names)
    for name, conf, counted in zip(class_names, confs, flags):
        if not counted:
            continue
        category = class_category(name)
        counts[category] += 1
        if category == "no_helmet":
            highest_infraction = max(highest_infraction, conf)
    return counts, highest_infraction, flags
