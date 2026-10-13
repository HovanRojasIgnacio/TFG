import os
from ultralytics import YOLO

def exportar_modelo():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "best.pt")
    
    if not os.path.exists(weights_path):
        print(f"[ERROR]: No se encontró el archivo 'best.pt' en {current_dir}")
        return

    print("[SISTEMA]: Cargando modelo PyTorch original...")
    model = YOLO(weights_path)

    print("\n--- EXPORTANDO A FORMATO ONNX ---")
    onnx_path = model.export(format="onnx", dynamic=True, simplify=True)
    print(f"[OK]: Modelo ONNX guardado en: {onnx_path}")

    print("\n--- EXPORTANDO A FORMATO TENSORRT (.engine) ---")
    try:
        engine_path = model.export(format="engine", device=0, half=True, dynamic=True)
        print(f"[OK]: Modelo TensorRT optimizado guardado en: {engine_path}")
    except Exception as e:
        print(f"\n[AVISO]: No se pudo exportar a TensorRT: {e}")
        print("El formato ONNX que acabamos de generar funcionará de maravilla.")

if __name__ == "__main__":
    exportar_modelo()