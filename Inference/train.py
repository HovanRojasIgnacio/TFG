"""Ajuste fino (fine-tuning) de YOLO26 para un modo de detección.

Uso:
    python train.py --data ruta/a/dataset.yaml --name casco_chaleco
    python train.py --data dataset.yaml --name prueba --model yolo26n.pt --epochs 50

Los resultados quedan en Inference/runs/<name>/ (pesos en weights/best.pt). Para usar el modelo en el
servidor, copia best.pt a Inference/profiles/<id>/ junto a su profile.json (véase el README).
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

from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo YOLO26 para un modo de detección")
    parser.add_argument("--data", required=True, help="dataset.yaml (formato Ultralytics)")
    parser.add_argument("--name", required=True, help="nombre del entrenamiento (carpeta en Inference/runs)")
    parser.add_argument("--model", default="yolo26s.pt", help="pesos preentrenados de partida")
    parser.add_argument("--epochs", type=int, default=100, help="épocas máximas")
    parser.add_argument("--patience", type=int, default=15, help="épocas sin mejora antes de parar")
    parser.add_argument("--batch", type=int, default=16, help="tamaño de lote (8 si falta memoria de GPU)")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,      # Parada temprana: se conservan los pesos de la mejor época
        imgsz=640,                   # Resolución de entrada (la del preentrenamiento en COCO)
        batch=args.batch,
        device=0,
        workers=4,
        optimizer="MuSGD",           # Optimizador híbrido SGD + Muon de YOLO26 (mejor que AdamW, capítulo 7)
        project=os.path.join(current_dir, "runs"),
        name=args.name,
        exist_ok=False,
        plots=True,
        # Aumento de datos
        mosaic=0.5,                  # Combina cuatro imágenes en una
        close_mosaic=10,             # Desactivado en las últimas 10 épocas
        scale=0.7,                   # Variación del tamaño de los objetos (operarios lejanos)
        fliplr=0.5,                  # Volteo horizontal
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Cambios de color, saturación y brillo (iluminación de obra)
    )
    print(f"\nPesos de la mejor época: {os.path.join(current_dir, 'runs', args.name, 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
