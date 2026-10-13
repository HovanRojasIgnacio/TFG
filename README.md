# TFG - Sistema de Vigilancia de EPIs en Tiempo Real (YOLO26s)

Este proyecto consiste en una aplicación web completa para el monitoreo en tiempo real del uso de cascos de seguridad (EPIs). El sistema utiliza un modelo personalizado **YOLO26s** optimizado para ejecutarse mediante aceleración por hardware en una GPU NVIDIA RTX 5060.

La arquitectura se divide en:
* **Backend (`Inference/`):** Servidor API de alto rendimiento con **FastAPI** que gestiona la webcam, la inferencia de IA y el streaming de vídeo.
* **Frontend (`Application/`):** Panel de control interactivo con interfaz oscura desarrollado en **React (Vite)**.

---

## 🛠️ Requisitos Previos

Antes de comenzar, asegúrate de tener instalado lo siguiente en tu sistema Windows:
1.  **Python 3.10 o superior** (Asegúrate de marcar "Add Python to PATH" en la instalación).
2.  **Node.js (Versión LTS)** (Para la gestión y ejecución de React).
3.  **CUDA Toolkit y cuDNN** (Altamente recomendado para habilitar el soporte de tu tarjeta gráfica RTX).

---

## 🚀 Guía de Inicio Rápido

Sigue estos pasos en orden para clonar, instalar y ejecutar todo el entorno en tu máquina local.

### 1. Clonar el repositorio
Abre una terminal (PowerShell o CMD) y ejecuta:
```bash
git clone https://github.com/HovanRojasIgnacio/TFG.git
cd TFG
```
### 2. Configurar y lanzar el Backend (FastAPI)

En la primera terminal de comandos, accede al directorio del backend para instalar las dependencias y arrancar el motor de Inteligencia Artificial:

```bash
cd Inference

# (Recomendado) entorno virtual propio
python -m venv .venv
.venv\Scripts\activate

# IMPORTANTE: instalar PyTorch con CUDA ANTES que el resto. El `pip install torch` normal
# instala la versión solo-CPU y toda la inferencia (y la carga del procesador) iría a la CPU.
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130

# Instalar el resto de librerías de Python
pip install -r requirements.txt

# Comprobar que la GPU está disponible (debe imprimir True y el nombre de la tarjeta)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# (Opcional, recomendado) Motor TensorRT: unas 3 veces más rápido que best.pt en FP16
# (4 ms frente a 11 ms por fotograma en una RTX 5060). Genera best.onnx y best.engine.
# El motor depende de tu GPU y de la versión de TensorRT, por eso no se sube a git.
pip install -r requirements-tensorrt.txt
python export.py
# Si no hay motor, el servidor usa best.pt en la GPU; si la exportación TensorRT falla se conserva el .onnx.

# (Opcional) Medir latencia y FPS de inferencia del modelo (sin cámara):
python benchmark.py

# Arrancar el servidor backend
python execute.py
```
El motor de IA se levantará y transmitirá el flujo de vídeo en: http://localhost:8000

### 3. Configurar y lanzar el Frontend (React)

Abre una segunda terminal (dejando la del backend activa), vuelve a la raíz del proyecto y ejecuta la interfaz web:

```bash
cd Application

# Instalar las dependencias de Node
npm install

# Lanzar el servidor de desarrollo de React
npm run dev
```
El frontend se levantará en el navegador en la dirección: http://localhost:5173

## Despliegue en GitHub Pages
La interfaz de usuario está configurada para compilarse y alojarse directamente de forma estática en GitHub. Si realizas cambios en el diseño de tu app de React y quieres actualizar la versión online, ejecuta desde la carpeta Application:

```Bash
npm run deploy
```
La plataforma se empaquetará de forma automática y los cambios se verán reflejados en tu enlace público definitivo:
 - https://hovanrojasignacio.github.io/TFG/

Para apuntar la interfaz a otro backend, define la variable `VITE_API_BASE` (por ejemplo en `Application/.env`: `VITE_API_BASE=http://192.168.1.50:8000`) antes de ejecutar `npm run dev` o `npm run build`.

Nota Técnica: Al acceder desde tu enlace público de GitHub Pages, la aplicación de React intentará conectarse con el motor local (localhost:8000). Por lo tanto, el recuadro de vídeo se activará únicamente si tienes encendido el backend localmente ejecutando python execute.py en tu PC.

## Código de Colores de Seguridad
El stream de vídeo renderiza las cajas de detección en tiempo real según los siguientes colores de riesgo:

 - 🔴 Rojo (ALERTA): Infracción de seguridad (Persona detectada sin casco: no_helmet o head).

 - 🟢 Verde (OK): Cumplimiento normativo (Casco colocado correctamente: with_helmet o helmet).

 - 🟠 Naranja: Elementos neutros analizados en el entorno (face).

## Seguridad y acceso a la red

Por defecto el backend escucha en `0.0.0.0:8000` con CORS abierto (`*`) y **sin autenticación**: cualquier equipo de la misma red puede ver el vídeo, leer el registro de infracciones y cambiar los ajustes. Es adecuado para una demostración en una red de confianza, pero no para una red compartida o pública.

Para limitar el acceso a este equipo, arranca el backend con:

```powershell
$env:GUARDWATCH_HOST = "127.0.0.1"
python execute.py
```

Las cámaras IP solo admiten URLs `http://`, `https://` o `rtsp://`.

## Cumplimiento EPI y evidencias

- **Cumplimiento EPI (%)** = cascos correctos / (cascos correctos + infracciones) × 100, calculado sobre las detecciones del fotograma actual. Los rostros neutros (`face`) no cuentan. Si no hay ninguna persona detectada, vale 100 %.
- **Infracciones:** cuando hay al menos una infracción se guarda una captura anotada como máximo cada 4 segundos en `Inference/static/infractions/`. El panel muestra las 12 últimas; el registro en memoria conserva 100 y se pierde al reiniciar el backend.
- **Espacio en disco:** las capturas no se borran automáticamente y crecen mientras haya infracciones. Elimina el contenido de `Inference/static/infractions/` cuando no necesites las evidencias.
