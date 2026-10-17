"""Exporta los pesos de un modo de detección a ONNX y a TensorRT (best.engine), en su misma carpeta.

    python export.py                          # modo casco
    python export.py --profile casco_chaleco  # genera el motor antes de elegir el modo desde el panel

El servidor genera el motor por su cuenta la primera vez que carga un modo (alrededor de un minuto, con menos
FPS mientras tanto); hacerlo antes con este script deja el cambio de modo en unos segundos.
"""
import argparse
import os
import venv_bootstrap
if __name__ == "__main__":
    venv_bootstrap.ensure_venv()  # Usa Inference/.venv (PyTorch con CUDA y TensorRT) aunque se lance con otro Python
# Ultralytics no debe instalar paquetes por su cuenta con pip (puede modificar el Python global)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
from ultralytics import YOLO
import profiles
from runtime import build_engine, require_cuda

def exportar_modelo(profile_id):
    current_dir = profiles.load(profile_id).base_dir
    weights_path = os.path.join(current_dir, "best.pt")

    if not os.path.exists(weights_path):
        print(f"[ERROR]: No se encontró el archivo 'best.pt' en {current_dir}")
        return

    require_cuda()

    print("[SISTEMA]: Cargando modelo PyTorch original...")
    model = YOLO(weights_path)

    print("\n--- EXPORTANDO A FORMATO ONNX ---")
    onnx_path = model.export(format="onnx", dynamic=True, simplify=True)
    print(f"[OK]: Modelo ONNX guardado en: {onnx_path}")

    # Mismo procedimiento que usa el servidor cuando falta best.engine (formas dinámicas, FP16)
    print("\n--- EXPORTANDO A FORMATO TENSORRT (.engine) ---")
    if build_engine(current_dir):
        print(f"[OK]: Modelo TensorRT optimizado guardado en: {os.path.join(current_dir, 'best.engine')}")
    else:
        print("\n[AVISO]: No se pudo exportar a TensorRT (¿falta pip install -r requirements-tensorrt.txt?).")
        print("El servidor usará best.pt en la GPU.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta un modo de detección a ONNX y TensorRT")
    parser.add_argument("--profile", default="casco", help="modo de detección (carpeta de Inference/profiles)")
    exportar_modelo(parser.parse_args().profile)
