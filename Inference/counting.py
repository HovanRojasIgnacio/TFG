"""Cuenta, para cada elemento de protección del perfil, cuántos se llevan y cuántos faltan.

Qué significa cada clase lo dice el perfil (profiles.py). Algunos modelos marcan a un mismo operario en varios
niveles: el de casco desplegado ve la persona (person_with_helmet, person_no_helmet), la cabeza
(head_with_helmet, head) y el propio casco (helmet). Si se contara cada recuadro, un operario con casco sumaría
tres "cascos correctos". Por eso, cuando el perfil asigna niveles, se cuenta cada cabeza y, además, cada persona
en cuyo recuadro no hay ninguna cabeza detectada (un operario lejano o de espaldas del que solo se ve el cuerpo);
las clases sin nivel (el casco suelto) no cuentan, porque ya las representa la cabeza. Si el perfil no asigna
niveles a ninguna clase (por ejemplo, un modelo Helmet / NoHelmet / Vest / NoVest) se cuenta cada recuadro,
salvo cuando el modelo se contradice: si marca el mismo sitio como "lleva" y como "falta" el mismo elemento (la
supresión de no máximos es por clase y no lo evita), solo cuenta el recuadro con más confianza.
"""
from typing import Dict, List, NamedTuple, Sequence

from profiles import Profile

Box = Sequence[float]  # x1, y1, x2, y2


def _contains(outer: Box, inner: Box) -> bool:
    """True si el centro del recuadro `inner` está dentro de `outer` (la cabeza está en esa persona)."""
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _head_distance(person: Box, head: Box) -> float:
    """Distancia del centro de la cabeza al centro del borde superior de la persona (donde debería estar)."""
    cx, cy = (head[0] + head[2]) / 2, (head[1] + head[3]) / 2
    return abs(cx - (person[0] + person[2]) / 2) + abs(cy - person[1])


def iou(a: Box, b: Box) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


DUPLICATE_IOU = 0.5  # Dos recuadros de persona que se solapan más que esto son el mismo operario
# Una cabeza solo representa a una persona si su centro está en esta fracción superior del recuadro: la cabeza de
# un vecino situada a media altura de otra persona no debe ocultarla
HEAD_ZONE = 0.4


def _head_zone_persons(boxes: Sequence[Box], persons: List[int], head: int) -> List[int]:
    """Personas cuya zona alta (HEAD_ZONE) contiene el centro de la cabeza `head`."""
    head_cy = (boxes[head][1] + boxes[head][3]) / 2
    return [i for i in persons if _contains(boxes[i], boxes[head])
            and head_cy <= boxes[i][1] + HEAD_ZONE * (boxes[i][3] - boxes[i][1])]


def _represented_persons(rules, boxes: Sequence[Box]) -> set:
    """Índices de las personas a las que representa alguna cabeza (su operario ya cuenta por la cabeza)."""
    # Una cabeza representa a la persona del borde superior más cercano; a las demás que la contienen solo si
    # dicen lo mismo que ella (el modelo ha marcado dos veces al mismo operario). Una persona sin casco no
    # desaparece por la cabeza con casco de otra.
    persons = [i for i, r in enumerate(rules) if r.level == 1]
    represented = set()
    for head in (j for j, r in enumerate(rules) if r.level == 0):
        inside = _head_zone_persons(boxes, persons, head)
        if not inside:
            continue
        owner = min(inside, key=lambda i: _head_distance(boxes[i], boxes[head]))
        represented.update(i for i in inside if i == owner or rules[i].state == rules[head].state)
    return represented


def _claim_person(i: int, boxes: Sequence[Box], represented: set, claimed: List[Box]) -> bool:
    """True si la persona `i` es un operario nuevo (y lo reclama); False si ya la representa otro recuadro."""
    if i in represented or any(iou(boxes[i], c) > DUPLICATE_IOU for c in claimed):
        return False
    claimed.append(boxes[i])
    return True


def _flags_with_levels(rules, boxes: Sequence[Box]) -> List[bool]:
    """Perfil con niveles: cuentan las cabezas y las personas que no representa ninguna cabeza ni otra persona."""
    represented = _represented_persons(rules, boxes)
    claimed = [boxes[i] for i in sorted(represented)]  # Personas ya representadas o ya contadas
    flags = []
    for i, rule in enumerate(rules):
        if rule.state in ("neutral", "ignore"):
            flags.append(rule.state == "neutral")      # neutral se muestra; ignore no
        elif rule.level == 1:
            flags.append(_claim_person(i, boxes, represented, claimed))
        else:
            flags.append(rule.level == 0)              # cada cabeza es un operario; el casco suelto, no
    return flags


def counted_flags(profile: Profile, class_names: Sequence[str],
                  boxes: Sequence[Box], confs: Sequence[float]) -> List[bool]:
    """Indica, para cada recuadro del frame, si cuenta en los contadores (un recuadro por operario).

    Cada operario se identifica por su posición. Con niveles en el perfil cuentan todas las cabezas y las
    personas a las que no representa ninguna cabeza, salvo que se solapen con otra persona ya representada
    (el modelo a veces marca dos veces a la misma persona). Sin niveles cuenta cada recuadro no neutral
    ni ignorado, salvo las contradicciones.
    """
    rules = [profile.rule(n) for n in class_names]
    # La jerarquía la decide el perfil, no el frame: un casco suelto no es un operario aunque sea lo único visible
    if any(r.level is not None for r in profile.classes.values()):
        return _flags_with_levels(rules, boxes)
    flags = [r.state != "ignore" for r in rules]  # cada recuadro cuenta; el neutral se muestra
    _drop_contradictions(rules, boxes, confs, flags)
    return flags


def _drop_contradictions(rules, boxes: Sequence[Box], confs: Sequence[float], flags: List[bool]) -> None:
    """De dos recuadros solapados que dicen lo contrario del mismo elemento, deja de contar el menos fiable."""
    # De más a menos confiable: un recuadro que sigue contando anula a los menos confiables que lo contradicen, y
    # uno ya anulado no anula a ninguno (el resultado no depende del orden en que llegan los recuadros)
    for i in sorted(range(len(rules)), key=lambda k: -confs[k]):
        if not flags[i]:
            continue
        for j in range(len(rules)):
            if j != i and flags[j] and confs[j] <= confs[i] and rules[i].item is not None \
                    and rules[i].item == rules[j].item and rules[i].state != rules[j].state \
                    and iou(boxes[i], boxes[j]) > DUPLICATE_IOU:
                flags[j] = False


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
          boxes: Sequence[Box]) -> Counts:
    """Cuenta los elementos llevados y los que faltan, los recuadros neutrales y la confianza de la infracción."""
    items = empty_items(profile)
    neutral = 0
    highest = 0.0
    flags = counted_flags(profile, class_names, boxes, confs)
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
