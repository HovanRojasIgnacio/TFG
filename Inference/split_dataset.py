"""Separa una parte de las imágenes de entrenamiento como conjunto de prueba, sin modificar el dataset.

Muchos datasets de Roboflow solo traen train y valid; si valid se usa para elegir la mejor época, hace falta
otro conjunto, nunca visto, para medir la precisión final. El script escribe listas de imágenes y un
dataset.yaml que las usa (Ultralytics encuentra cada etiqueta cambiando /images/ por /labels/):

    python split_dataset.py --dataset ruta/al/dataset --names Helmet NoHelmet NoVest Vest --out ruta/salida
"""
import argparse
import glob
import os
import random

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def images_in(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMAGE_EXTENSIONS))


def main():
    parser = argparse.ArgumentParser(description="Separa un conjunto de prueba de las imágenes de entrenamiento")
    parser.add_argument("--dataset", required=True, help="carpeta con train/images y valid/images")
    parser.add_argument("--names", nargs="+", required=True, help="nombres de las clases en orden")
    parser.add_argument("--out", required=True, help="carpeta donde escribir las listas y el dataset.yaml")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0, help="semilla: la misma separación en cada ejecución")
    args = parser.parse_args()

    root = os.path.abspath(args.dataset)
    train = images_in(os.path.join(root, "train", "images"))
    valid = images_in(os.path.join(root, "valid", "images"))
    if not train or not valid:
        raise SystemExit(f"No se encontraron imágenes en {root}/train/images y {root}/valid/images")
    random.Random(args.seed).shuffle(train)
    n_test = round(len(train) * args.test_fraction)
    test, train = sorted(train[:n_test]), sorted(train[n_test:])

    os.makedirs(args.out, exist_ok=True)
    lists = {"train": train, "val": valid, "test": test}
    for split, paths in lists.items():
        with open(os.path.join(args.out, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(p.replace("\\", "/") for p in paths) + "\n")
    out = os.path.abspath(args.out).replace("\\", "/")
    with open(os.path.join(args.out, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {out}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n\nnames:\n")
        f.writelines(f"  {i}: {name}\n" for i, name in enumerate(args.names))
    print(f"train {len(train)} | val {len(valid)} | test {len(test)} -> {os.path.join(args.out, 'dataset.yaml')}")


if __name__ == "__main__":
    main()
