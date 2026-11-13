import os
import venv_bootstrap
venv_bootstrap.setup(__name__ == "__main__")
# RTSP por TCP: FFmpeg usa UDP por defecto y, por Wi-Fi, los paquetes perdidos dan frames
# corruptos o grises. Se puede cambiar con la variable de entorno (p. ej. "rtsp_transport;udp")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cameras
import evidence
import profiles
import video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("guardwatch")

pipeline = None  # VideoPipeline; se crea al arrancar el servidor (lifespan), no al importar el módulo


@asynccontextmanager
async def lifespan(_app):
    """Carga el modelo y arranca el hilo de vídeo al iniciar el servidor, y lo para al cerrarlo.

    Importar este módulo no carga el modelo ni abre la cámara, así que las pruebas pueden instalar antes su
    propio pipeline (con un modelo simulado y sin hilo de vídeo).
    """
    global pipeline
    if pipeline is None:
        pipeline = video.VideoPipeline(video.load_detector(STARTUP_PROFILE))
    yield
    pipeline.stop()


app = FastAPI(title="GuardWatch AI Engine", lifespan=lifespan)

# Solo el panel puede llamar a la API desde un navegador. Con "*", cualquier página web abierta por el
# operador podía cambiar la cámara o el umbral y leer las respuestas. GUARDWATCH_ORIGINS añade otros
# orígenes separados por comas (por ejemplo, el panel servido desde otro equipo de la red).
DASHBOARD_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173",   # npm run dev
                     "http://localhost:4173", "http://127.0.0.1:4173",   # npm run preview
                     "https://hovanrojasignacio.github.io"]               # versión publicada en GitHub Pages
extra_origins = [o.strip().rstrip("/") for o in os.environ.get("GUARDWATCH_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=DASHBOARD_ORIGINS + extra_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.mount("/static", StaticFiles(directory=evidence.static_dir), name="static")

# Modo de detección con el que arranca el servidor (carpeta de Inference/profiles)
STARTUP_PROFILE = os.environ.get("GUARDWATCH_PROFILE", "casco")
if STARTUP_PROFILE not in profiles.discover():
    raise SystemExit(f"GUARDWATCH_PROFILE={STARTUP_PROFILE!r} no es un perfil instalado. Disponibles: "
                     f"{', '.join(profiles.discover()) or 'ninguno'} (carpetas de {profiles.PROFILES_DIR})")


class SettingsUpdate(BaseModel):
    conf: float


class ProfileSelect(BaseModel):
    id: str


@app.post("/api/settings")
def update_settings(payload: SettingsUpdate):
    if 0.1 <= payload.conf <= 0.95:
        pipeline.conf_threshold = payload.conf
        pipeline.conf_set_by_user = True
        return {"status": "success", "conf_threshold": pipeline.conf_threshold}
    raise HTTPException(status_code=400, detail="El umbral debe estar entre 0.10 y 0.95")


@app.get("/api/cameras")
def get_cameras(refresh: bool = False):
    """Devuelve las cámaras disponibles y la seleccionada."""
    return pipeline.get_camera_info(force_refresh=refresh)


@app.post("/api/cameras/refresh")
def refresh_cameras():
    """Fuerza la actualización de las cámaras disponibles."""
    return pipeline.get_camera_info(force_refresh=True)


@app.post("/api/cameras/select")
def select_camera(payload: cameras.CameraSource):
    """Cambia a otra fuente de cámara."""
    if payload.type == "ip":
        # Solo protocolos de red: evita que OpenCV abra rutas de archivo locales u otros esquemas
        if not payload.url or not payload.url.strip().lower().startswith(cameras.ALLOWED_IP_SCHEMES):
            raise HTTPException(status_code=400,
                                detail="La URL de la cámara IP debe empezar por http://, https:// o rtsp://")
    elif payload.type == "local":
        if payload.index is None or payload.index < 0:
            raise HTTPException(status_code=400, detail="Índice de cámara local no válido")
    else:
        raise HTTPException(status_code=400, detail="Tipo de cámara no válido (use 'local' o 'ip')")

    if pipeline.set_camera(payload):
        return {"status": "success", "camera": pipeline.get_camera_info(probe=False)["current"]}
    raise HTTPException(status_code=400, detail="No se pudo conectar a la cámara seleccionada")


@app.get("/api/profiles")
def get_profiles():
    """Modos de detección instalados, el actual y el estado de la carga en curso."""
    available = []
    for profile_id in profiles.discover():
        try:
            p = profiles.load(profile_id)
        except Exception as e:  # Un profile.json roto no debe ocultar los demás modos
            logger.warning("Perfil %s no válido: %s", profile_id, e)
            continue
        available.append({"id": p.id, "name": p.name, "items": p.items})
    return {
        "profiles": available,
        "current": pipeline.detector.profile.id,
        "loading": pipeline.profile_loading,
        "error": pipeline.profile_error,
    }


@app.post("/api/profiles/select")
def select_profile(payload: ProfileSelect):
    """Cambia de modo de detección. La carga es en segundo plano: 202, y el progreso en /api/profiles."""
    try:
        started = pipeline.select_profile(payload.id)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Modo de detección desconocido: {payload.id}")
    except video.ProfileBusy as e:
        raise HTTPException(status_code=409, detail=f"Ya se está cargando el modo {e}")
    if not started:
        return {"status": "current", "id": payload.id}
    return JSONResponse(status_code=202, content={"status": "loading", "id": payload.id})


@app.get("/api/stats")
def get_stats(request: Request):
    base_url = str(request.base_url).rstrip("/")
    # Convierte rutas relativas a absolutas basadas en la petición actual (sobre copias)
    infractions = [
        {**item, "image_url": f"{base_url}{item['image_path']}"}
        for item in list(pipeline.infractions_log)[-12:]
    ]

    camera_info = pipeline.get_camera_info(probe=False)
    detector = pipeline.detector

    # Latencia de procesamiento del servidor: lectura del frame -> frame anotado y codificado (últimos 100 frames)
    window = sorted(pipeline.latency_window)
    processing_ms = round(sum(window) / len(window), 1) if window else 0.0
    processing_p95_ms = round(window[int(0.95 * (len(window) - 1))], 1) if window else 0.0

    return {
        "processing_ms": processing_ms,
        "processing_p95_ms": processing_p95_ms,
        "fps": round(pipeline.fps, 1),
        "device": detector.runtime.device.upper(),
        "weights_format": detector.weights_format,
        "profile": detector.summary(),
        "profile_loading": pipeline.profile_loading is not None,
        "items": pipeline.items,
        "neutral": pipeline.neutral,
        "violations": pipeline.violations,
        "violation_confirmed": pipeline.violation_confirmed,
        "safety_score": pipeline.safety_score,
        "conf_threshold": pipeline.conf_threshold,
        "infractions": infractions,
        "camera": camera_info["current"],
        "camera_error": camera_info["error"]
    }


async def stream_generator():
    """Generador asíncrono: no ocupa hilos del servidor mientras espera frames.

    Con un generador síncrono, Starlette ejecuta cada paso en su grupo de 40 hilos, que comparte con todos
    los endpoints; sin frames (cámara caída) cada espectador que se iba dejaba un hilo ocupado y, al
    acumularse, la API entera dejaba de responder. Al desconectarse el cliente, Starlette cancela el
    asyncio.sleep y el generador termina.
    """
    last_id = -1
    while True:
        pipeline.last_viewer_time = time.time()
        with pipeline.lock:
            frame_bytes = pipeline.latest_encoded_frame
            frame_id = pipeline.frame_id

        # Solo se envía si hay un frame nuevo (evita reenviar el mismo frame)
        if frame_bytes is not None and frame_id != last_id:
            last_id = frame_id
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        await asyncio.sleep(0.03)  # Limita el envío a ~30 FPS para no saturar el canal HTTP


@app.get("/api/stream")
async def video_feed():
    return StreamingResponse(stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn
    # GUARDWATCH_HOST=127.0.0.1 limita el acceso a este equipo (por defecto, toda la red local)
    # El flujo MJPEG no termina solo: sin este límite, Ctrl+C esperaría a que el navegador cierre la pestaña
    uvicorn.run(app, host=os.environ.get("GUARDWATCH_HOST", "0.0.0.0"), port=8000, timeout_graceful_shutdown=3)
