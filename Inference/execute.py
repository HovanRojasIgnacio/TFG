import cv2
import torch
from ultralytics import YOLO

def main():
    # 1. Configurar la ruta de tus mejores pesos entrenados
    # Revisa que esta ruta coincida exactamente con la de tu ordenador
    weights_path = r"C:\Users\Firecraft811\Desktop\Safety Helmet Wearing Dataset\runs\detect\TFG_PPE_Detection\YOLO26s_Base_Run\weights\best.pt"
    
    # 2. Detectar aceleración por hardware (Tu RTX 5060)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Lanzando entorno de inferencia en: {device.upper()}")

    # 3. Cargar el modelo YOLO26s
    model = YOLO(weights_path)

    # 4. Inicializar la cámara del ordenador
    # El índice '0' activa la webcam integrada. Si usas una webcam USB externa, prueba con '1'.
    cap = cv2.VideoCapture(0)
    
    # Forzar una resolución fluida para la captura de video
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: No se puede acceder a la webcam.")
        return

    print("\n--- ¡Sistema de Vigilancia Activo! ---")
    print("Presiona la tecla 'q' para cerrar la ventana del sistema.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame de la cámara.")
            break

        # Mirror effect (opcional: voltea la imagen para que actúe como un espejo natural)
        frame = cv2.flip(frame, 1)

        # 5. Ejecutar la predicción de YOLO26s con los parámetros óptimos de tu TFG
        # - conf=0.325: Extraído directamente del pico de tu curva F1 enviado anteriormente.
        # - half=True: Activa FP16 para maximizar los FPS en tu RTX 5060.
        results = model.predict(frame, device=device, conf=0.325, half=True, verbose=False)
        
        boxes = results[0].boxes

        # 6. Lógica de renderizado personalizado por colores
        for box in boxes:
            # Extraer coordenadas, confianza y ID de clase
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Definición dinámica del color según el riesgo (Formato BGR en OpenCV)
            if "no_helmet" in class_name.lower() or class_name.lower() == "head":
                color = (0, 0, 255)      # ROJO para infracciones o zonas desprotegidas
                label = f"ALERTA: {class_name} ({conf:.2f})"
            elif "with_helmet" in class_name.lower() or class_name.lower() == "helmet":
                color = (0, 255, 0)      # VERDE para EPI correcto colocado
                label = f"OK: {class_name} ({conf:.2f})"
            else:
                color = (255, 165, 0)    # NARANJA/AZUL para clases neutras (como 'face')
                label = f"{class_name} ({conf:.2f})"

            # Dibujar el rectángulo contenedor y la etiqueta de texto en el frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 7. Mostrar la interfaz en tiempo real
        cv2.imshow("TFG - Sistema de Detección de EPIs en Tiempo Real (YOLO26s)", frame)

        # Romper el bucle de video de forma segura si el usuario pulsa la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos de hardware
    cap.release()
    cv2.destroyAllWindows()
    print("Sistema cerrado correctamente.")

if __name__ == "__main__":
    main()