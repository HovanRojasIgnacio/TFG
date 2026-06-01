import os
# Evita conflictos de librerías duplicadas OpenMP en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI(title="YOLO26s Web Inference Engine")

# Permitir conexiones desde el puerto de React (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Configurar rutas y hardware de forma segura
current_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(current_dir, "best.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[SISTEMA]: Inicializando entorno en: {device.upper()}")

# 2. Cargar el modelo de forma global
model = YOLO(weights_path)

# Solo activar FP16 si hay GPU disponible
use_half = True if device == "cuda" else False

# 3. Warm-up de la GPU para evitar retrasos iniciales
print("[SISTEMA]: Realizando Warm-up de los tensores de vídeo...")
warmup_frame = np.zeros((480, 640, 3), dtype=np.uint8)
model.predict(warmup_frame, device=device, conf=0.325, half=use_half, verbose=False)
print("[SISTEMA]: Motor listo y optimizado.")

def gen_frames():
    """Generador continuo de frames procesados para streaming web."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR]: No se puede acceder a la webcam.")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Efecto espejo para comodidad del usuario
            frame = cv2.flip(frame, 1)

            # Inferencia YOLO con parámetros optimizados
            results = model.predict(frame, device=device, conf=0.325, half=use_half, verbose=False)
            boxes = results[0].boxes

            # Renderizado personalizado de las cajas delimitadoras
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if "no_helmet" in class_name.lower() or class_name.lower() == "head":
                    color = (0, 0, 255)  # ROJO
                    label = f"ALERTA: {class_name} ({conf:.2f})"
                elif "with_helmet" in class_name.lower() or class_name.lower() == "helmet":
                    color = (0, 255, 0)  # VERDE
                    label = f"OK: {class_name} ({conf:.2f})"
                else:
                    color = (255, 165, 0)  # NARANJA
                    label = f"{class_name} ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Codificar el resultado final en formato JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            # Enpaquetar frame en formato binario para el streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
    except Exception as e:
        print(f"[ERROR DURANTE STREAMING]: {e}")
    finally:
        cap.release()
        print("[SISTEMA]: Captura de vídeo liberada.")

@app.get("/api/stream")
def video_feed():
    """Endpoint principal de streaming de vídeo para el frontend."""
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    # Lanzamos el backend en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)