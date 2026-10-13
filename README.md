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

# Instalar las librerías necesarias de Python
pip install -r requirements.txt

# (Opcional) Si quieres optimizar tu best.pt a formatos ultrarrápidos (.engine / .onnx):
# Si la exportación a TensorRT falla, se conserva el .onnx y el servidor lo usará automáticamente.
python export.py

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