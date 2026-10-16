"""Mide la latencia y los FPS de inferencia del modelo (sin cámara ni servidor).

Uso:
    python benchmark.py                      # frames sintéticos 640x480
    python benchmark.py --source video.mp4   # frames de un vídeo
    python benchmark.py --frames 300 --warmup 20
"""
import argparse
import os
# Igual que en execute.py: debe fijarse antes de importar torch, numpy u OpenCV
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import venv_bootstrap
if __name__ == "__main__":
    venv_bootstrap.ensure_venv()  # Usa Inference/.venv (PyTorch con CUDA) aunque se lance con otro Python
# Ultralytics no debe instalar paquetes por su cuenta con pip (puede modificar el Python global)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
import time

import cv2
import numpy as np
import torch

from runtime import load_model, precision_kwargs

current_dir = os.path.dirname(os.path.abspath(__file__))


def frame_source(source, total: int):
    """Genera 'total' frames: de un vídeo (en bucle) o sintéticos."""
    if source:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el vídeo: {source}")
        produced = 0
        read_since_rewind = 0
        while produced < total:
            ok, frame = cap.read()
            if not ok:
                if read_since_rewind == 0:
                    # Recién rebobinado y sigue sin dar frames: el vídeo no se puede decodificar
                    cap.release()
                    raise RuntimeError(f"No se pudo leer ningún frame del vídeo: {source}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_since_rewind = 0
                continue
            read_since_rewind += 1
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

    # Misma carga que execute.py (solo GPU; genera el motor TensorRT si falta)
    model, (weights, device, half) = load_model(current_dir)
    print(f"Modelo: {os.path.basename(weights)} | Dispositivo: {device} ({torch.cuda.get_device_name(0)}) | half={half}")

    latencies = []
    for i, frame in enumerate(frame_source(args.source, args.warmup + args.frames)):
        start = time.perf_counter()
        model.predict(frame, device=device, conf=args.conf, verbose=False, **precision_kwargs(half))
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
