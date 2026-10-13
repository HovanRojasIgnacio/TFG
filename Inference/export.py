import os
import venv_bootstrap
if __name__ == "__main__":
    venv_bootstrap.ensure_venv()  # Usa Inference/.venv (PyTorch con CUDA y TensorRT) aunque se lance con otro Python
# Ultralytics no debe instalar paquetes por su cuenta con pip (puede modificar el Python global)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
from ultralytics import YOLO
from runtime import build_engine, require_cuda

def exportar_modelo():
    current_dir = os.path.dirname(os.path.abspath(__file__))
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
    exportar_modelo()
