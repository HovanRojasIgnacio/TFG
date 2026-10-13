"""Mide la latencia y los FPS de inferencia del modelo (sin cámara ni servidor).

Uso:
    python benchmark.py                      # frames sintéticos 640x480
    python benchmark.py --source video.mp4   # frames de un vídeo
    python benchmark.py --frames 300 --warmup 20
"""
import argparse
import os
# Ultralytics no debe instalar paquetes por su cuenta con pip (puede modificar el Python global)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from runtime import select_runtime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

current_dir = os.path.dirname(os.path.abspath(__file__))


def frame_source(source, total: int):
    """Genera 'total' frames: de un vídeo (en bucle) o sintéticos."""
    if source:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el vídeo: {source}")
        produced = 0
        while produced < total:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            produced += 1
            yield frame
        cap.release()
    else:
        rng = np.random.default_rng(0)
        for _ in range(total):
            yield rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Benchmark de inferencia YOLO")
    parser.add_argument("--source", help="Ruta a un vídeo (por defecto, frames sintéticos)")
    parser.add_argument("--frames", type=int, default=200, help="Frames medidos")
    parser.add_argument("--warmup", type=int, default=10, help="Frames de calentamiento (no medidos)")
    parser.add_argument("--conf", type=float, default=0.325)
    args = parser.parse_args()

    # Misma selección que execute.py: GPU siempre que sea posible
    weights, device, half = select_runtime(current_dir)
    model = YOLO(weights, task="detect")
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "-"
    print(f"Modelo: {os.path.basename(weights)} | Dispositivo: {device} ({gpu_name}) | half={half}")

    latencies = []
    for i, frame in enumerate(frame_source(args.source, args.warmup + args.frames)):
        start = time.perf_counter()
        model.predict(frame, device=device, conf=args.conf, half=half, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000
        if i >= args.warmup:
            latencies.append(elapsed)

    lat = np.array(latencies)
    print(f"Frames medidos: {len(lat)}")
    print(f"Latencia media: {lat.mean():.1f} ms | p50: {np.percentile(lat, 50):.1f} ms | "
          f"p95: {np.percentile(lat, 95):.1f} ms | máx: {lat.max():.1f} ms")
    print(f"FPS (1 / latencia media): {1000 / lat.mean():.1f}")
    print("Nota: mide solo model.predict(); no incluye captura, dibujado ni codificación JPEG.")


if __name__ == "__main__":
    main()
