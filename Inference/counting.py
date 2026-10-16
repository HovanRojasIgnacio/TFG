"""Agrupa las clases del modelo en las categorías del panel y cuenta cada operario una sola vez.

El modelo desplegado marca a un mismo operario en varios niveles: la persona (person_with_helmet,
person_no_helmet), la cabeza (head_with_helmet, head) y el propio casco (helmet). Si se contara cada
recuadro, un operario con casco sumaría tres "cascos correctos". Por eso se cuenta cada cabeza y, además,
cada persona en cuyo recuadro no hay ninguna cabeza detectada (un operario lejano o de espaldas del que
solo se ve el cuerpo). El casco suelto no cuenta: ya lo representa la cabeza. Con un modelo sin esa
jerarquía (por ejemplo, clases helmet / no_helmet) se cuenta cada recuadro.
"""
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

CATEGORIES = ("helmet", "no_helmet", "face")
Box = Sequence[float]  # x1, y1, x2, y2


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


def _contains(outer: Box, inner: Box) -> bool:
    """True si el centro del recuadro `inner` está dentro de `outer` (la cabeza está en esa persona)."""
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _iou(a: Box, b: Box) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


DUPLICATE_IOU = 0.5  # Dos recuadros de persona que se solapan más que esto son el mismo operario


def counted_flags(class_names: Sequence[str], boxes: Optional[Sequence[Box]] = None) -> List[bool]:
    """Indica, para cada recuadro del frame, si cuenta en los contadores (un recuadro por operario).

    Con las coordenadas (`boxes`), cada operario se identifica por su posición: cuentan todas las cabezas y
    las personas sin ninguna cabeza dentro, salvo que se solapen con otra persona ya representada (el modelo
    a veces marca dos veces a la misma persona). Sin coordenadas se cuenta solo el nivel más detallado presente.
    """
    levels = [worker_level(n) for n in class_names]
    hierarchy = any(lvl is not None for lvl in levels)
    heads = [boxes[i] for i, lvl in enumerate(levels) if lvl == 0] if boxes is not None else []
    frame_level = min((lvl for lvl in levels if lvl is not None), default=None)
    # Personas ya representadas: las que contienen una cabeza y las que se van contando
    claimed = [boxes[i] for i, lvl in enumerate(levels)
               if lvl == 1 and any(_contains(boxes[i], h) for h in heads)] if boxes is not None else []
    flags = []
    for i, (name, lvl) in enumerate(zip(class_names, levels)):
        if class_category(name) == "face":
            flags.append(not name.lower().startswith("person"))  # "person" sola no es un rostro
        elif not hierarchy:
            flags.append(True)                                 # modelo sin jerarquía: cada recuadro
        elif lvl == 0:
            flags.append(True)                                 # cada cabeza es un operario
        elif lvl == 1:
            if boxes is None:
                flags.append(frame_level == 1)
            elif any(_contains(boxes[i], h) for h in heads) or                     any(_iou(boxes[i], c) > DUPLICATE_IOU for c in claimed):
                flags.append(False)                            # ya la representa su cabeza u otra persona
            else:
                flags.append(True)
                claimed.append(boxes[i])
        else:
            flags.append(False)                                # casco suelto: ya cuenta su cabeza
    return flags


def count(class_names: Sequence[str], confs: Sequence[float],
          boxes: Optional[Sequence[Box]] = None) -> Tuple[Dict[str, int], float, List[bool]]:
    """Cuenta operarios con casco, sin casco y rostros.

    Devuelve los contadores, la confianza máxima de una infracción y qué recuadros han contado.
    """
    counts = {c: 0 for c in CATEGORIES}
    highest_infraction = 0.0
    flags = counted_flags(class_names, boxes)
    for name, conf, counted in zip(class_names, confs, flags):
        if not counted:
            continue
        category = class_category(name)
        counts[category] += 1
        if category == "no_helmet":
            highest_infraction = max(highest_infraction, conf)
    return counts, highest_infraction, flags
