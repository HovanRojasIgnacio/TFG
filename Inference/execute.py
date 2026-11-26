import os
import venv_bootstrap
venv_bootstrap.setup(__name__ == "__main__")
# RTSP por TCP: FFmpeg usa UDP por defecto y, por Wi-Fi, los paquetes perdidos dan frames
# corruptos o grises. Se puede cambiar con la variable de entorno (p. ej. "rtsp_transport;udp")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import asyncio
import ipaddress
import logging
import socket
import time
from contextlib import asynccontextmanager
from pydantic import BaseModel, field_validator
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cameras
import config
import evidence
import mjpeg
import profiles
import runtime
import video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("guardwatch")

settings = config.Settings.from_env()  # Variables GUARDWATCH_*: si alguna no es válida, el servidor no arranca
pipeline = None  # VideoPipeline; se crea al arrancar el servidor (lifespan), no al importar el módulo


@asynccontextmanager
async def lifespan(_app):
    """Carga el modelo y arranca el hilo de vídeo al iniciar el servidor, y lo para al cerrarlo.

    Importar este módulo no carga el modelo ni abre la cámara, así que las pruebas pueden instalar antes su
    propio pipeline (con un modelo simulado y sin hilo de vídeo).
    """
    global pipeline
    if pipeline is None:
        recorder = evidence.EvidenceRecorder(evidence.INFRACTIONS_DIR, evidence.LOG_FILE, settings.snapshot_days)
        pipeline = video.VideoPipeline(video.load_detector(settings.profile), recorder)
    yield
    pipeline.stop()


app = FastAPI(title="GuardWatch AI Engine", lifespan=lifespan)

# Solo el panel puede llamar a la API desde un navegador. Con "*", cualquier página web abierta por el
# operador podía cambiar la cámara o el umbral y leer las respuestas. GUARDWATCH_ORIGINS añade otros
# orígenes separados por comas (por ejemplo, el panel servido desde otro equipo de la red).
DASHBOARD_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173",   # npm run dev
                     "http://localhost:4173", "http://127.0.0.1:4173",   # npm run preview
                     "https://hovanrojasignacio.github.io"]               # versión publicada en GitHub Pages

app.add_middleware(
    CORSMiddleware,
    allow_origins=DASHBOARD_ORIGINS + list(settings.origins),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Nombres con los que se puede llamar al backend. CORS no basta: con «DNS rebinding», una página web maliciosa hace
# que su propio dominio apunte a 127.0.0.1 y entonces sus peticiones al backend son del mismo origen, así que el
# navegador le deja leer el vídeo y las infracciones. Esas peticiones llevan el dominio de la página en la
# cabecera Host; las legítimas llevan localhost, el nombre de este equipo o una dirección IP (una IP no se puede
# redirigir). GUARDWATCH_HOSTS añade otros nombres separados por comas (por ejemplo, el del equipo en el DNS local).
ALLOWED_HOSTNAMES = {"localhost", socket.gethostname().lower()} | settings.hosts


def host_allowed(host: str) -> bool:
    """True si la cabecera Host (con o sin puerto) es una dirección IP o un nombre de ALLOWED_HOSTNAMES."""
    host = host.strip().lower()
    if host.startswith("["):  # IPv6: [::1]:8000
        name = host[1:].split("]", 1)[0]
    else:
        name = host.rsplit(":", 1)[0] if ":" in host else host
    if name in ALLOWED_HOSTNAMES:
        return True
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


class HostCheckMiddleware:
    """Rechaza con 400 las peticiones cuya cabecera Host no es este servidor (véase ALLOWED_HOSTNAMES)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            host = dict(scope["headers"]).get(b"host", b"").decode("latin-1")
            if not host_allowed(host):
                response = JSONResponse(status_code=400, content={
                    "detail": "Cabecera Host no admitida (véase GUARDWATCH_HOSTS en el README)"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


app.add_middleware(HostCheckMiddleware)

# Las capturas de las infracciones. La carpeta la crea el EvidenceRecorder al arrancar (no está en git): no se
# comprueba al importar
SNAPSHOTS_URL = "/static/infractions"
app.mount(SNAPSHOTS_URL, StaticFiles(directory=evidence.INFRACTIONS_DIR, check_dir=False), name="snapshots")


@app.exception_handler(RequestValidationError)
async def invalid_request(_request, exc: RequestValidationError):
    """Una petición no válida es un 400 con un solo mensaje, el que muestra el panel.

    Los mensajes para el operador los escriben los modelos (los validadores de SettingsUpdate y de las cámaras de
    cameras.py); un error de forma (falta un campo, un número que no lo es) lleva el de Pydantic tras el campo.
    """
    error = exc.errors()[0]
    if error["type"] == "value_error":
        message = str(error["ctx"]["error"])
    else:
        field = ".".join(str(part) for part in error["loc"] if part != "body")
        message = f"{field}: {error['msg']}" if field else error["msg"]
    return JSONResponse(status_code=400, content={"detail": message})


class SettingsUpdate(BaseModel):
    conf: float

    @field_validator("conf")
    @classmethod
    def _in_range(cls, conf: float) -> float:
        if not profiles.CONF_MIN <= conf <= profiles.CONF_MAX:
            raise ValueError(f"El umbral debe estar entre {profiles.CONF_MIN:.2f} y {profiles.CONF_MAX:.2f}")
        return conf


class ProfileSelect(BaseModel):
    id: str


@app.post("/api/settings")
def update_settings(payload: SettingsUpdate):
    pipeline.user_conf = payload.conf  # Desde ahora, cambiar de modo ya no cambia el umbral
    return {"status": "success", "conf_threshold": pipeline.conf_threshold}


@app.get("/api/cameras")
def get_cameras(refresh: bool = False):
    """Devuelve las cámaras disponibles y la seleccionada."""
    return pipeline.camera.info(force_refresh=refresh)


@app.post("/api/cameras/select")
def select_camera(payload: cameras.CameraSource):
    """Cambia a otra fuente de cámara (el tipo, el índice y la URL ya los ha validado el modelo)."""
    if pipeline.camera.select(payload):
        return {"status": "success", "camera": pipeline.camera.source.summary()}
    raise HTTPException(status_code=400, detail="No se pudo conectar a la cámara seleccionada")


@app.get("/api/profiles")
def get_profiles():
    """Modos de detección instalados, el actual y el estado de la carga en curso."""
    return {
        "profiles": [p.summary() for p in profiles.load_all().values()],
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


@app.get("/api/infractions")
def get_infractions(request: Request):
    """El registro completo de infracciones (hasta 100), de la más antigua a la más reciente.

    El panel lo pide cuando cambia last_infraction_id en /api/stats, no en cada consulta.
    """
    base_url = str(request.base_url).rstrip("/")
    # Dirección completa de cada imagen, a partir de la de esta petición (sobre copias)
    return {"infractions": [{**item._asdict(), "image_url": f"{base_url}{SNAPSHOTS_URL}/{item.image}"}
                            for item in list(pipeline.evidence.log)]}


@app.get("/api/stats")
def get_stats():
    camera_info = pipeline.camera.info(probe=False)
    detected = pipeline.detected  # Una sola lectura: los contadores y el modo son los del mismo frame
    counts = detected.counts
    log = list(pipeline.evidence.log)

    # Latencia de procesamiento del servidor: lectura del frame -> frame anotado y codificado (últimos 100 frames)
    window = sorted(pipeline.latency_window)
    processing_ms = round(sum(window) / len(window), 1) if window else 0.0
    processing_p95_ms = round(window[int(0.95 * (len(window) - 1))], 1) if window else 0.0

    return {
        "processing_ms": processing_ms,
        "processing_p95_ms": processing_p95_ms,
        "fps": round(pipeline.fps, 1),
        "device": runtime.DEVICE.upper(),
        "weights_format": pipeline.detector.model.weights_format,
        "profile": detected.profile.summary(),
        "profile_loading": pipeline.profile_loading,  # Id del modo que se está cargando, o None
        "profile_error": pipeline.profile_error,
        "items": counts.items,
        "neutral": counts.neutral,
        "violations": counts.violations,
        "violation_confirmed": detected.violation_confirmed,
        "safety_score": counts.safety_score,
        "conf_threshold": pipeline.conf_threshold,
        "conf_range": [profiles.CONF_MIN, profiles.CONF_MAX],  # Los límites del deslizador del panel
        "infractions_total": len(log),
        "last_infraction_id": log[-1].id if log else None,
        "camera": camera_info["current"],
        "camera_error": pipeline.camera_error
    }


async def stream_generator():
    """Generador asíncrono: no ocupa hilos del servidor mientras espera frames.

    Con un generador síncrono, Starlette ejecuta cada paso en su grupo de 40 hilos, que comparte con todos
    los endpoints; sin frames (cámara caída) cada espectador que se iba dejaba un hilo ocupado y, al
    acumularse, la API entera dejaba de responder. Al desconectarse el cliente, Starlette cancela el
    asyncio.sleep y el generador termina. El formato de cada parte está en mjpeg.py.
    """
    yield mjpeg.PART_HEADER
    last_id = -1
    while True:
        pipeline.last_viewer_time = time.time()
        with pipeline.lock:
            frame_bytes = pipeline.latest_encoded_frame
            frame_id = pipeline.frame_id

        # Solo se envía si hay un frame nuevo (evita reenviar el mismo frame)
        if frame_bytes is not None and frame_id != last_id:
            last_id = frame_id
            yield mjpeg.part(frame_bytes)
        await asyncio.sleep(0.03)  # Limita el envío a ~30 FPS para no saturar el canal HTTP


@app.get("/api/stream")
async def video_feed():
    return StreamingResponse(stream_generator(), media_type=mjpeg.MEDIA_TYPE)


if __name__ == "__main__":
    import uvicorn
    # El flujo MJPEG no termina solo: sin este límite, Ctrl+C esperaría a que el navegador cierre la pestaña
    uvicorn.run(app, host=settings.listen, port=8000, timeout_graceful_shutdown=3)
