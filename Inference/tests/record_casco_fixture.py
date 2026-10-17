"""Graba las detecciones del modelo de casco y los contadores que les da counting.count().

Sirve de referencia para comprobar que el perfil "casco" cuenta exactamente igual que el código anterior a los
perfiles (test_counting.py). Se ejecuta una vez, con imágenes sin anotar (no con las capturas de infracciones,
que ya llevan los recuadros dibujados):

    python Inference/tests/record_casco_fixture.py --images ruta/a/imagenes --weights Inference/best.pt
"""
import argparse
import glob
import json
import os
import random
import sys

INFERENCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, INFERENCE_DIR)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import venv_bootstrap
if __name__ == "__main__":
    venv_bootstrap.ensure_venv()
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from ultralytics import YOLO

import counting

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    parser = argparse.ArgumentParser(description="Graba el fixture de regresión del perfil casco")
    parser.add_argument("--images", required=True, help="carpeta con imágenes sin anotar")
    parser.add_argument("--weights", required=True, help="best.pt del modelo de casco")
    parser.add_argument("--count", type=int, default=60, help="imágenes elegidas al azar (semilla fija)")
    parser.add_argument("--conf", type=float, default=0.325)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "fixtures", "casco_detections.json"))
    args = parser.parse_args()

    images = sorted(p for p in glob.glob(os.path.join(args.images, "**", "*"), recursive=True)
                    if p.lower().endswith(IMAGE_EXTENSIONS))
    random.Random(0).shuffle(images)
    model = YOLO(args.weights, task="detect")
    frames = []
    for path in images[:args.count]:
        boxes = model.predict(path, device="cuda", conf=args.conf, verbose=False)[0].boxes
        coords, confs = boxes.xyxy.int().tolist(), boxes.conf.tolist()
        names = [model.names[c] for c in boxes.cls.int().tolist()]
        counts, highest, flags = counting.count(names, confs, coords)
        # También sin coordenadas: count() admite boxes=None y entonces cuenta solo el nivel más detallado
        counts_nb, _, flags_nb = counting.count(names, confs)
        frames.append({"names": names, "confs": confs, "boxes": coords,
                       "counts": counts, "highest": highest, "flags": flags,
                       "counts_no_boxes": counts_nb, "flags_no_boxes": flags_nb})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model_names": list(model.names.values()), "conf": args.conf, "frames": frames}, f)
    with_hierarchy = sum(any(n.startswith(("head", "person_")) for n in fr["names"]) for fr in frames)
    print(f"{len(frames)} frames ({with_hierarchy} con cabezas o personas) -> {args.out}")


if __name__ == "__main__":
    main()
