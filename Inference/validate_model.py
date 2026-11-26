"""Compara la precisión de best.pt (PyTorch, referencia) con la del motor TensorRT (best.engine) de un modo.

Hay que ejecutarlo con tu conjunto de validación real antes de citar cifras de precisión:

    python validate_model.py --data ruta/a/yolo_dataset/dataset.yaml
    python validate_model.py --data ruta/a/dataset.yaml --split test
    python validate_model.py --images ruta/a/carpeta_de_imagenes      # solo concordancia, sin etiquetas
    python validate_model.py --profile casco_chaleco --data ruta/a/dataset.yaml --split test

--profile Modo de detección (carpeta de Inference/profiles); por defecto, casco.

--data    Ejecuta la validación de Ultralytics con ambos modelos y muestra precisión, recall, mAP50 y
          mAP50-95 (global y por clase). Debe ser el dataset con el que se entrenó best.pt.
--images  Compara las detecciones del motor con las de best.pt (FP32) en cada imagen y cuenta
          detecciones coincidentes, perdidas, añadidas y cambios en la decisión "hay infracción".
"""
import argparse
import os
import venv_bootstrap
venv_bootstrap.setup(__name__ == "__main__")

import counting
import profiles
from runtime import Model, build_engine, engine_path, ensure_engine, pt_path, require_cuda
from split_dataset import images_in


def load_models(base_dir):
    """best.pt en FP32, para que sea una referencia exacta, y el motor actual (lleva su precisión dentro)."""
    require_cuda()
    pt, engine = pt_path(base_dir), engine_path(base_dir)
    if not os.path.exists(pt):
        raise SystemExit(f"Falta {pt}: es la referencia de la comparación.")
    if not ensure_engine(base_dir, build_engine):
        raise SystemExit("No hay un best.engine actual y no se pudo generar (¿falta requirements-tensorrt.txt?).")
    return Model.open(pt, half=False), Model.open(engine, half=False)


def validate(model, data, split):
    # batch=1 en ambos para comparar con la misma geometría de imagen (el motor dinámico se generó con batch 1)
    return model.yolo.val(data=data, split=split, imgsz=640, batch=1, device=0, plots=False, verbose=False)


def report_map(pt_model, engine_model, data, split):
    results = {"best.pt": validate(pt_model, data, split), "best.engine": validate(engine_model, data, split)}
    print(f"\nValidación en '{split}' de {data}\n")
    print(f"{'modelo':14} {'precisión':>10} {'recall':>8} {'mAP50':>8} {'mAP50-95':>9}")
    for name, r in results.items():
        print(f"{name:14} {r.box.mp:>10.3f} {r.box.mr:>8.3f} {r.box.map50:>8.3f} {r.box.map:>9.3f}")
    a, b = results["best.pt"].box, results["best.engine"].box
    print(f"{'diferencia':14} {b.mp - a.mp:>+10.3f} {b.mr - a.mr:>+8.3f} {b.map50 - a.map50:>+8.3f} {b.map - a.map:>+9.3f}")

    names = pt_model.names
    print(f"\nmAP50-95 por clase:\n{'clase':22} {'best.pt':>8} {'engine':>8} {'dif':>8}")
    for i, cls in enumerate(a.ap_class_index):
        j = list(b.ap_class_index).index(cls) if cls in list(b.ap_class_index) else None
        pt_v = a.maps[cls]
        en_v = b.maps[cls] if j is not None else float("nan")
        print(f"{names[int(cls)]:22} {pt_v:>8.3f} {en_v:>8.3f} {en_v - pt_v:>+8.3f}")


def _has_violation(profile, detections):
    # Misma regla que el servidor: cada operario cuenta una vez (véase counting.py)
    return counting.count(profile, detections).violations > 0


def report_agreement(profile, pt_model, engine_model, folder):
    images = images_in(folder, recursive=True)
    if not images:
        raise SystemExit(f"No hay imágenes en {folder}")
    threshold = profile.threshold  # El mismo umbral con el que arranca el servidor en este modo
    matched = missed = extra = flips = 0
    conf_diffs = []
    total_ref = 0
    for path in images:
        ref = pt_model.detect(path, threshold)  # best.pt en FP32 (referencia), con la inferencia del servidor
        out = engine_model.detect(path, threshold)
        total_ref += len(ref.names)
        used = set()
        for cls, conf, box in zip(*ref):
            best = None
            for i, (c2, conf2, box2) in enumerate(zip(*out)):
                if i in used or c2 != cls:
                    continue
                v = counting.iou(box, box2)
                if v >= 0.5 and (best is None or v > best[0]):
                    best = (v, i, conf2)
            if best:
                matched += 1
                used.add(best[1])
                conf_diffs.append(abs(conf - best[2]))
            else:
                missed += 1
        extra += len(out.names) - len(used)
        flips += _has_violation(profile, ref) != _has_violation(profile, out)
    mean_diff = sum(conf_diffs) / max(len(conf_diffs), 1)
    print(f"\nConcordancia del motor frente a best.pt (FP32) en {len(images)} imágenes de {folder}")
    print(f"  detecciones de referencia : {total_ref}")
    print(f"  coincidentes              : {matched}")
    print(f"  perdidas por el motor     : {missed}")
    print(f"  añadidas por el motor     : {extra}")
    print(f"  diferencia media de conf. : {mean_diff:.3f}")
    print(f"  cambios de decisión 'infracción': {flips} de {len(images)} imágenes")


def main():
    parser = argparse.ArgumentParser(description="Compara best.pt con best.engine")
    parser.add_argument("--profile", default="casco", help="modo de detección (carpeta de Inference/profiles)")
    parser.add_argument("--data", help="dataset.yaml con el que se entrenó el modelo (calcula mAP)")
    parser.add_argument("--split", default="val", help="partición del dataset: val o test")
    parser.add_argument("--images", help="carpeta de imágenes para comparar detecciones (sin etiquetas)")
    args = parser.parse_args()
    if not args.data and not args.images:
        parser.error("indica --data y/o --images")

    profile = profiles.load(args.profile)
    pt_model, engine_model = load_models(profile.base_dir)
    profiles.check_model(profile, pt_model.names)
    print(f"Modo: {profile.name} ({profile.base_dir})")
    if args.data:
        report_map(pt_model, engine_model, args.data, args.split)
    if args.images:
        report_agreement(profile, pt_model, engine_model, args.images)


if __name__ == "__main__":
    main()
