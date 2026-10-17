# TFG - Sistema de Vigilancia de EPIs en Tiempo Real (YOLO26s)

Este proyecto consiste en una aplicación web completa para el monitoreo en tiempo real del uso de cascos de seguridad y, opcionalmente, chalecos de alta visibilidad (EPIs). El sistema utiliza un modelo personalizado **YOLO26s** optimizado para ejecutarse mediante aceleración por hardware en una GPU NVIDIA RTX 5060.

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

# Entorno virtual: el servidor lo usa solo, aunque lo arranques con otro Python
python -m venv .venv
.venv\Scripts\activate

# IMPORTANTE: instalar PyTorch con CUDA ANTES que el resto. El `pip install torch` normal instala
# la versión solo-CPU, y el servidor NO arranca sin GPU NVIDIA (no existe modo CPU).
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130

# Instalar el resto de librerías de Python
pip install -r requirements.txt

# Comprobar que la GPU está disponible (debe imprimir True y el nombre de la tarjeta)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# (Recomendado) TensorRT: unas 3 veces más rápido que best.pt en FP16 (4 ms frente a 11 ms por fotograma
# en una RTX 5060). No hay que generar el motor a mano: al arrancar, si falta best.engine, o no es
# compatible con tu GPU o con tu versión de TensorRT, el servidor lo genera solo (aprox. 1 minuto).
# Sin TensorRT instalado, el servidor usa best.pt en la GPU.
pip install -r requirements-tensorrt.txt
# (Opcional) generar best.engine y best.onnx de un modo antes de arrancar (por defecto, casco):
python export.py
python export.py --profile casco_chaleco

# (Opcional) Medir latencia y FPS de inferencia del modelo (sin cámara):
python benchmark.py --profile casco

# (Opcional) Comparar la precisión de best.pt y del motor con TU conjunto de validación
# (el dataset.yaml con el que se entrenó; mAP, precisión y recall por clase):
python validate_model.py --profile casco --data ruta\a\yolo_dataset\dataset.yaml

# Arrancar el servidor backend (usa Inference/.venv automáticamente, con cualquier Python)
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
El frontend se levantará en el navegador en la dirección: http://localhost:5173/TFG/ (la ruta `/TFG/` viene de `base` en `vite.config.js`)

## Despliegue en GitHub Pages
La interfaz de usuario está configurada para compilarse y alojarse directamente de forma estática en GitHub. Si realizas cambios en el diseño de tu app de React y quieres actualizar la versión online, ejecuta desde la carpeta Application:

```Bash
npm run deploy
```
La plataforma se empaquetará de forma automática y los cambios se verán reflejados en tu enlace público definitivo:
 - https://hovanrojasignacio.github.io/TFG/

Para apuntar la interfaz a otro backend, define la variable `VITE_API_BASE` (por ejemplo en `Application/.env`: `VITE_API_BASE=http://192.168.1.50:8000`) antes de ejecutar `npm run dev` o `npm run build`. El backend solo acepta peticiones del navegador desde las direcciones del panel (`http://localhost:5173`, `http://localhost:4173` y la versión de GitHub Pages); si el panel se sirve desde otra dirección, añádela al arrancar el backend con `GUARDWATCH_ORIGINS` (por ejemplo, `$env:GUARDWATCH_ORIGINS = "http://192.168.1.50:5173"`).

Nota Técnica: Al acceder desde tu enlace público de GitHub Pages, la aplicación de React intentará conectarse con el motor local (localhost:8000). Por lo tanto, el recuadro de vídeo se activará únicamente si tienes encendido el backend localmente ejecutando python execute.py en tu PC.

## Modos de detección

El panel permite elegir, con el servidor en marcha, qué elementos de protección se comprueban:

| Modo | Carpeta | Comprueba | Modelo |
|---|---|---|---|
| Casco | `Inference/profiles/casco/` | casco | YOLO26s de siete clases (persona, cabeza, casco, rostro...) |
| Casco y chaleco | `Inference/profiles/casco_chaleco/` | casco y chaleco | YOLO26s entrenado con `train.py` sobre el dataset [20250731-PPE2286](https://universe.roboflow.com/ying2001/20250731-ppe2286-m0ega) (ying2001, Roboflow Universe, CC BY 4.0) |

Cada modo es una carpeta con los pesos (`best.pt`) y un `profile.json` que dice qué significa cada clase: a qué elemento se refiere (`item`), si se lleva (`ok`), falta (`missing`), es neutral (`neutral`, se dibuja pero no cuenta) o se ignora (`ignore`), y, en modelos que ven a un mismo operario como persona y como cabeza, su nivel (`level`: `head` o `person`) para contarlo una sola vez. El servidor comprueba al cargar un modo que `profile.json` describe todas las clases del modelo.

- **Modo al arrancar:** `casco`; se cambia con la variable de entorno `GUARDWATCH_PROFILE` (por ejemplo, `$env:GUARDWATCH_PROFILE = "casco_chaleco"`).
- **Cambio desde el panel:** el modelo nuevo se carga en segundo plano mientras el anterior sigue procesando vídeo. La primera vez que se usa un modo en un equipo se genera su motor TensorRT (alrededor de un minuto, con menos FPS mientras tanto); `python export.py --profile <modo>` lo genera antes. Al cambiar de modo se aplica el umbral de confianza recomendado en su `profile.json` (`conf`).
- **Añadir un modo:** entrena el modelo (`python train.py --data dataset.yaml --name <modo>`; si el dataset solo trae train y valid, `split_dataset.py` separa antes un conjunto de prueba), copia `Inference/runs/<modo>/weights/best.pt` a `Inference/profiles/<modo>/` y escribe su `profile.json` tomando como ejemplo los existentes. Aparece en el panel sin reiniciar el servidor.
- **API:** `GET /api/profiles` (modos, el actual y el que se está cargando) y `POST /api/profiles/select` con `{"id": "casco_chaleco"}` (responde 202 y carga en segundo plano; 400 si el modo no existe, 409 si ya se está cargando otro).

## Código de Colores de Seguridad
El stream de vídeo renderiza las cajas de detección en tiempo real según los siguientes colores de riesgo (el texto de cada caja lo da el `profile.json` del modo):

 - 🔴 Rojo (ALERTA): falta un elemento de protección (`PELIGRO: Sin Casco`, `PELIGRO: Sin Chaleco`).

 - 🟢 Verde (OK): el elemento se lleva correctamente (`OK: Casco`, `OK: Chaleco`).

 - 🟠 Naranja: detecciones neutrales, que no cuentan para el cumplimiento (`Rostro` en el modo casco).

## Seguridad y acceso a la red

Por defecto el backend escucha en `0.0.0.0:8000` **sin autenticación**. CORS solo admite las direcciones del panel, así que una página web cualquiera abierta en el navegador no puede usar la API, pero cualquier equipo de la misma red puede llamarla directamente: ver el vídeo, leer el registro de infracciones y cambiar los ajustes. Es adecuado para una demostración en una red de confianza, pero no para una red compartida o pública.

Para limitar el acceso a este equipo, arranca el backend con:

```powershell
$env:GUARDWATCH_HOST = "127.0.0.1"
python execute.py
```

Las cámaras IP solo admiten URLs `http://`, `https://` o `rtsp://`. Si la URL lleva usuario y contraseña (`rtsp://usuario:clave@host/...`) o parámetros como `pwd=`, la API, el panel y el registro los muestran como `***`. Las cámaras RTSP se leen por TCP; para usar UDP, define `OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;udp"` antes de arrancar el backend.

## Cumplimiento EPI y evidencias

- **Cumplimiento EPI (%)** = elementos llevados / (elementos llevados + elementos que faltan) × 100, sumando todos los elementos del modo (casco, chaleco) en el fotograma actual. En el modo casco cada operario cuenta una vez; en el modo casco y chaleco cuenta cada recuadro, así que un operario con casco y sin chaleco suma un elemento llevado y uno que falta (50 %). Las detecciones neutrales no cuentan. Si no se detecta ningún elemento, vale 100 %.
- **Infracciones:** cuando falta al menos un elemento se guarda una captura anotada como máximo cada 4 segundos en `Inference/static/infractions/`. El panel muestra las 12 últimas; el registro en memoria conserva 100 y se pierde al reiniciar el backend. Cuando una infracción sale de ese registro, su captura se borra del disco. Además, al arrancar y cada hora se borran las capturas con más de 30 días, incluidas las de ejecuciones anteriores (la Ley Orgánica 3/2018 fija un mes como máximo); el plazo se cambia con la variable de entorno `GUARDWATCH_SNAPSHOT_DAYS`.
- **Espacio en disco:** como mucho se acumulan las capturas de los últimos 30 días (o del plazo de `GUARDWATCH_SNAPSHOT_DAYS`); las más antiguas se borran solas. Puedes vaciar `Inference/static/infractions/` antes si no necesitas las evidencias; si borras la carpeta, el servidor la vuelve a crear.
