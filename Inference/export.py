"""Genera el motor TensorRT (best.engine) de un modo de detección en su misma carpeta, antes de arrancar el servidor.

    python export.py                          # modo casco
    python export.py --profile casco_chaleco  # genera el motor antes de elegir el modo desde el panel
    python export.py --onnx                   # además, best.onnx (el servidor no lo usa; para otras herramientas)

El servidor genera el motor por su cuenta la primera vez que carga un modo (alrededor de un minuto, con menos FPS
mientras tanto); hacerlo antes con este script deja el cambio de modo en unos segundos.
"""
import argparse
import os
import venv_bootstrap
venv_bootstrap.setup(__name__ == "__main__")
import profiles
from runtime import build_engine, ensure_weights, require_cuda


def export_profile(profile_id: str, onnx: bool = False) -> None:
    profile = profiles.load(profile_id)
    base_dir = profile.base_dir
    ensure_weights(base_dir, profile.weights_url, profile.weights_sha256)
    weights_path = os.path.join(base_dir, "best.pt")
    if not os.path.exists(weights_path):
        raise SystemExit(f"[ERROR]: No se encontró 'best.pt' en {base_dir}")
    require_cuda()

    if onnx:
        from ultralytics import YOLO
        print("--- EXPORTANDO A ONNX ---")
        print(f"[OK]: {YOLO(weights_path).export(format='onnx', dynamic=True, simplify=True)}")

    # Mismo procedimiento que usa el servidor cuando falta best.engine (formas dinámicas, FP16)
    print("--- GENERANDO EL MOTOR TENSORRT ---")
    if build_engine(base_dir):
        print(f"[OK]: {os.path.join(base_dir, 'best.engine')}")
    else:
        print("[AVISO]: No se pudo generar el motor TensorRT (¿falta pip install -r requirements-tensorrt.txt?). "
              "El servidor usará best.pt en la GPU.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera el motor TensorRT de un modo de detección")
    parser.add_argument("--profile", default="casco", help="modo de detección (carpeta de Inference/profiles)")
    parser.add_argument("--onnx", action="store_true", help="exportar también best.onnx")
    args = parser.parse_args()
    export_profile(args.profile, args.onnx)
