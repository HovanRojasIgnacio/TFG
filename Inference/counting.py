"""Cuenta, para cada elemento de protección del perfil, cuántos se llevan y cuántos faltan.

Qué significa cada clase lo dice el perfil (profiles.py). Algunos modelos marcan a un mismo operario en varios
niveles: el de casco desplegado ve la persona (person_with_helmet, person_no_helmet), la cabeza
(head_with_helmet, head) y el propio casco (helmet). Si se contara cada recuadro, un operario con casco sumaría
tres "cascos correctos". Por eso, cuando el perfil asigna niveles, se cuenta cada cabeza y, además, cada persona
en cuyo recuadro no hay ninguna cabeza detectada (un operario lejano o de espaldas del que solo se ve el cuerpo);
las clases sin nivel (el casco suelto) no cuentan, porque ya las representa la cabeza. Si en el frame no hay
ninguna clase con nivel (por ejemplo, un modelo Helmet / NoHelmet / Vest / NoVest) se cuenta cada recuadro.
"""
from typing import Dict, List, NamedTuple, Optional, Sequence

from profiles import Profile

Box = Sequence[float]  # x1, y1, x2, y2


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


def counted_flags(profile: Profile, class_names: Sequence[str],
                  boxes: Optional[Sequence[Box]] = None) -> List[bool]:
    """Indica, para cada recuadro del frame, si cuenta en los contadores (un recuadro por operario).

    Con las coordenadas (`boxes`), cada operario se identifica por su posición: cuentan todas las cabezas y
    las personas sin ninguna cabeza dentro, salvo que se solapen con otra persona ya representada (el modelo
    a veces marca dos veces a la misma persona). Sin coordenadas se cuenta solo el nivel más detallado presente.
    """
    rules = [profile.rule(n) for n in class_names]
    levels = [r.level for r in rules]
    hierarchy = any(lvl is not None for lvl in levels)
    heads = [boxes[i] for i, lvl in enumerate(levels) if lvl == 0] if boxes is not None else []
    frame_level = min((lvl for lvl in levels if lvl is not None), default=None)
    # Personas ya representadas: las que contienen una cabeza y las que se van contando
    claimed = [boxes[i] for i, lvl in enumerate(levels)
               if lvl == 1 and any(_contains(boxes[i], h) for h in heads)] if boxes is not None else []
    flags = []
    for i, (rule, lvl) in enumerate(zip(rules, levels)):
        if rule.state in ("neutral", "ignore"):
            flags.append(rule.state == "neutral")              # neutral se muestra; ignore no
        elif not hierarchy:
            flags.append(True)                                 # sin jerarquía: cada recuadro
        elif lvl == 0:
            flags.append(True)                                 # cada cabeza es un operario
        elif lvl == 1:
            if boxes is None:
                flags.append(frame_level == 1)
            elif any(_contains(boxes[i], h) for h in heads) or \
                    any(_iou(boxes[i], c) > DUPLICATE_IOU for c in claimed):
                flags.append(False)                            # ya la representa su cabeza u otra persona
            else:
                flags.append(True)
                claimed.append(boxes[i])
        else:
            flags.append(False)                                # casco suelto: ya cuenta su cabeza
    return flags


class Counts(NamedTuple):
    items: Dict[str, Dict[str, int]]   # {elemento: {"ok": n, "missing": n}}
    neutral: int
    highest_violation: float           # Confianza máxima de una infracción (0 si no hay)
    flags: List[bool]                  # Qué recuadros han contado (son los que se dibujan)

    @property
    def violations(self) -> int:
        return sum(c["missing"] for c in self.items.values())

    @property
    def safety_score(self) -> int:
        """Porcentaje de elementos llevados sobre los contados; 100 si no se ha contado ninguno."""
        ok = sum(c["ok"] for c in self.items.values())
        total = ok + self.violations
        return int(ok / total * 100) if total > 0 else 100


def empty_items(profile: Profile) -> Dict[str, Dict[str, int]]:
    return {item: {"ok": 0, "missing": 0} for item in profile.items}


def count(profile: Profile, class_names: Sequence[str], confs: Sequence[float],
          boxes: Optional[Sequence[Box]] = None) -> Counts:
    """Cuenta los elementos llevados y los que faltan, los recuadros neutrales y la confianza de la infracción."""
    items = empty_items(profile)
    neutral = 0
    highest = 0.0
    flags = counted_flags(profile, class_names, boxes)
    for name, conf, counted in zip(class_names, confs, flags):
        if not counted:
            continue
        rule = profile.rule(name)
        if rule.state == "neutral":
            neutral += 1
            continue
        items[rule.item][rule.state] += 1
        if rule.state == "missing":
            highest = max(highest, conf)
    return Counts(items, neutral, highest, flags)
