import os
# Evita el error de Windows "OpenMP runtime already initialized" cuando torch, numpy y
# OpenCV cargan cada uno su propia copia de libiomp (workaround habitual). Debe fijarse
# antes de importar cualquiera de ellos.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import venv_bootstrap
if __name__ == "__main__":
    venv_bootstrap.ensure_venv()  # Usa Inference/.venv (PyTorch con CUDA) aunque se lance con otro Python
# Ultralytics no debe instalar paquetes por su cuenta con pip (puede modificar el Python global)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
# RTSP por TCP: FFmpeg usa UDP por defecto y, por Wi-Fi, los paquetes perdidos dan frames
# corruptos o grises. Se puede cambiar con la variable de entorno (p. ej. "rtsp_transport;udp")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import asyncio
import logging
import cv2
import time
import threading
from collections import deque
from typing import Dict, List, Optional, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import counting
from runtime import load_model, precision_kwargs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("guardwatch")

app = FastAPI(title="GuardWatch AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
infractions_dir = os.path.join(static_dir, "infractions")
os.makedirs(infractions_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Modelo en la GPU (solo GPU): motor TensorRT, generado automáticamente si hace falta, o best.pt.
# load_model() ya hace una inferencia de prueba, que sirve de warm-up (véase runtime.py)
model, (weights_path, device, use_half) = load_model(current_dir)
logger.info("Modelo: %s | dispositivo: %s | FP16: %s", os.path.basename(weights_path), device, use_half)


BOX_STYLES = {  # Color BGR y texto de la etiqueta
    "no_helmet": ((0, 0, 255), "PELIGRO: Sin Casco"),
    "helmet": ((0, 255, 0), "OK: Casco"),
    "face": ((0, 165, 255), "Rostro"),  # naranja
}

# Camera detection cache
_camera_cache = {"cameras": [], "last_update": 0}
_camera_cache_lock = threading.Lock()
CAMERA_CACHE_TTL = 10.0  # Cache for 10 seconds
ALLOWED_IP_SCHEMES = ("http://", "https://", "rtsp://")  # Esquemas admitidos para cámaras IP
VIEWER_TIMEOUT = 2.0  # Segundos sin pedir frames tras los que se deja de codificar el stream
RECONNECT_MIN_DELAY = 2.0 # Espera inicial entre reintentos de conexión (s)
RECONNECT_MAX_DELAY = 15.0  # Espera máxima entre reintentos (s)
IP_OPEN_TIMEOUT_MS = 5000  # Tiempo máximo para conectar con una cámara IP
IP_READ_TIMEOUT_MS = 5000  # Tiempo máximo esperando un frame de una cámara IP
CAMERA_TEST_TIMEOUT = (IP_OPEN_TIMEOUT_MS + IP_READ_TIMEOUT_MS) / 1000 + 1  # Prueba al cambiar de cámara (s)
FRAME_FAIL_TIMEOUT = 2.0  # Segundos sin frames válidos tras los que se da la cámara por desconectada
# Días que se conservan las capturas de infracciones. La Ley Orgánica 3/2018 (art. 22) fija un máximo de un
# mes para las imágenes de videovigilancia salvo que se necesiten para acreditar un hecho.
SNAPSHOT_RETENTION_DAYS = float(os.environ.get("GUARDWATCH_SNAPSHOT_DAYS", "30"))
PURGE_INTERVAL = 3600.0  # Cada cuánto se revisan las capturas antiguas mientras el servidor está en marcha (s)


def purge_old_snapshots() -> set:
    """Borra las capturas con más de SNAPSHOT_RETENTION_DAYS días y devuelve sus nombres.

    El registro de infracciones está en memoria y empieza vacío en cada arranque, así que sin esta limpieza
    las capturas de ejecuciones anteriores se quedarían en el disco para siempre.
    """
    limit = time.time() - SNAPSHOT_RETENTION_DAYS * 86400
    removed = set()
    for entry in os.scandir(infractions_dir):
        if entry.is_file() and entry.name.lower().endswith(".jpg") and entry.stat().st_mtime < limit:
            try:
                os.remove(entry.path)
                removed.add(entry.name)
            except OSError as e:
                logger.warning("No se pudo borrar la captura %s: %s", entry.name, e)
    if removed:
        logger.info("Borradas %d capturas con más de %g días", len(removed), SNAPSHOT_RETENTION_DAYS)
    return removed


SECRET_QUERY_KEYS = {"password", "passwd", "pass", "pwd", "token", "key", "auth"}


def mask_url(url: Optional[str]) -> Optional[str]:
    """Oculta usuario, contraseña y tokens de la URL de una cámara antes de mostrarla o registrarla.

    La API responde a cualquier origen, así que las credenciales nunca salen del servidor.
    """
    if not url:
        return url
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            netloc = "***@" + netloc.rsplit("@", 1)[1]
        query = urlencode([(k, "***" if k.lower() in SECRET_QUERY_KEYS else v)
                           for k, v in parse_qsl(parts.query, keep_blank_values=True)], safe="*")
        return urlunsplit(parts._replace(netloc=netloc, query=query))
    except ValueError:
        return "URL no válida"


def detect_local_cameras(max_cameras: int = 5, force_refresh: bool = False,
                         skip_index: Optional[int] = None) -> List[Dict]:
    """Detect available local cameras with caching.

    skip_index: índice de la cámara en uso por el pipeline; no se reabre para no interrumpirla.
    """
    global _camera_cache

    with _camera_cache_lock:
        now = time.time()
        # La caché también vale cuando no hay cámaras (lista vacía), si ya se sondeó antes
        if not force_refresh and _camera_cache["last_update"] and (now - _camera_cache["last_update"] < CAMERA_CACHE_TTL):
            return _camera_cache["cameras"]

        cameras = []
        consecutive_failures = 0
        for i in range(max_cameras):
            if i == skip_index:
                cameras.append({
                    "index": i,
                    "name": f"Cámara Local {i}",
                    "type": "local",
                    "width": 640,
                    "height": 480
                })
                consecutive_failures = 0
                continue
            cap = None
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cameras.append({
                        "index": i,
                        "name": f"Cámara Local {i}",
                        "type": "local",
                        "width": width if width > 0 else 640,
                        "height": height if height > 0 else 480
                    })
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except Exception as e:
                logger.warning("Error detecting camera %s: %s", i, e)
                consecutive_failures += 1
            finally:
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
            
            if consecutive_failures >= 2:
                break
        
        _camera_cache = {"cameras": cameras, "last_update": now}
        return cameras


class CameraSource(BaseModel):
    type: str  # "local" or "ip"
    index: Optional[int] = None
    url: Optional[str] = None
    name: Optional[str] = None


def _source_label(source: CameraSource) -> str:
    """Nombre de la cámara para mensajes y registros (sin credenciales)."""
    return source.name or mask_url(source.url) or f"Cámara Local {source.index}"


def _open_capture(source: CameraSource):
    """Abre la cámara indicada. Devuelve el VideoCapture (abierto o no) o None si falla.

    Las cámaras IP se abren con FFmpeg y tiempos máximos explícitos: sin ellos, una URL que no
    responde bloquea unos 30 s en FFmpeg y después OpenCV prueba otros backends (hasta ~2 min).
    """
    try:
        if source.type == "local":
            cap = cv2.VideoCapture(source.index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        elif source.type == "ip":
            cap = cv2.VideoCapture(source.url, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, IP_OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, IP_READ_TIMEOUT_MS,
            ])
        else:
            return None
        return cap
    except Exception as e:
        logger.error("Failed to open camera: %s", e)
        return None


class LatestFrameReader:
    """Lee una cámara IP en su propio hilo y conserva solo el último frame.

    FFmpeg ignora CAP_PROP_BUFFERSIZE: si la inferencia va más lenta que la cámara, los frames se
    acumulan en su búfer y el vídeo se retrasa cada vez más. Leyendo sin pausa en otro hilo, el
    bucle de inferencia siempre procesa el frame más reciente y descarta los intermedios.
    Ofrece la misma interfaz que cv2.VideoCapture que usa el bucle (isOpened, read, release).
    """

    def __init__(self, cap):
        self._cap = cap
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0        # Frames leídos de la cámara
        self._consumed = 0   # Último frame entregado al bucle
        self._stopped = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            while not self._stopped:
                ok, frame = self._cap.read()
                if not ok:
                    # El bucle da la cámara por desconectada si no llegan frames en FRAME_FAIL_TIMEOUT
                    time.sleep(0.01)
                    continue
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
        except Exception as e:
            logger.error("IP camera reader failed: %s", e)
        finally:
            # Se libera en este hilo para no cerrar la cámara a mitad de una lectura
            self._cap.release()

    def isOpened(self) -> bool:
        return not self._stopped and self._cap.isOpened()

    def read(self, timeout: float = 1.0):
        """Devuelve el frame más reciente que aún no se ha entregado, esperando hasta `timeout` s."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._consumed or self._stopped, timeout)
            if self._stopped or self._seq == self._consumed:
                return False, None
            self._consumed = self._seq
            return True, self._frame

    def release(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()


def _connect_camera(source: CameraSource, timeout: float = CAMERA_TEST_TIMEOUT):
    """Abre la cámara y lee un frame para confirmar que funciona, con tiempo máximo.

    Devuelve la cámara ya abierta (para que el pipeline la use sin volver a conectar: algunas
    cámaras IP solo admiten una sesión y conectar dos veces duplica la espera) o None si falla.
    """
    result = {}
    result_lock = threading.Lock()

    def connect():
        cap = _open_capture(source)
        ok = False
        try:
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
        except Exception as e:
            logger.warning("Camera test failed: %s", e)
        with result_lock:
            if ok and not result.get("abandoned"):
                result["cap"] = cap
                return
        # Fallo, o la prueba ya se dio por perdida: nadie más va a usar esta cámara
        if cap is not None:
            cap.release()

    thread = threading.Thread(target=connect, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    with result_lock:
        if "cap" in result:
            return result["cap"]
        result["abandoned"] = True
    if thread.is_alive():
        logger.warning("Camera connection test timed out after %ss", timeout)
    return None


class VideoPipeline:
    def __init__(self):
        self.conf_threshold: float = 0.325
        self.fps: float = 0.0
        self.active_detections: Dict[str, int] = {"helmet": 0, "no_helmet": 0, "face": 0}
        self.safety_score: int = 100
        self.infractions_log = deque(maxlen=100)
        self.last_infraction_save: float = 0.0
        self.save_cooldown: float = 4.0
        
        self.latest_encoded_frame = None
        self.frame_id: int = 0  # Se incrementa con cada frame nuevo codificado
        self.last_viewer_time: float = 0.0  # Última vez que un cliente pidió un frame del stream
        self.latency_window = deque(maxlen=100)  # ms desde que se lee un frame hasta que queda publicado
        self.lock = threading.Lock()
        self.is_running = True
        self.camera_source: Optional[CameraSource] = CameraSource(type="local", index=0, name="Cámara Local 0")
        self.camera_error: Optional[str] = None
        self._cap = None
        self._camera_lock = threading.Lock()
        self._handover = None  # (fuente, cámara ya abierta) que set_camera deja para el bucle
        purge_old_snapshots()
        self._last_purge = time.time()
        
        self.thread = threading.Thread(target=self._process_stream, daemon=True)
        self.thread.start()

    @staticmethod
    def _same_source(a: Optional[CameraSource], b: Optional[CameraSource]) -> bool:
        if a is None or b is None:
            return False
        return a.type == b.type and a.index == b.index and a.url == b.url

    def set_camera(self, source: CameraSource) -> bool:
        """Change the camera source. Returns True if successful."""
        if self._same_source(source, self.camera_source):
            return True

        # La prueba (con tiempo máximo) se hace fuera del lock para no bloquear el bucle de captura
        cap = _connect_camera(source)
        if cap is None:
            return False

        with self._camera_lock:
            # El hilo de proceso detecta el cambio, libera la cámara anterior y usa la ya abierta
            replaced = self._handover
            self._handover = (source, cap)
            self.camera_source = source
            self.camera_error = None
        if replaced is not None:
            replaced[1].release()  # Otro cambio anterior que el bucle aún no había recogido
        return True

    def get_camera_info(self, force_refresh: bool = False, probe: bool = True) -> Dict:
        """Get current camera info and available cameras.

        probe=False devuelve la lista en caché sin abrir ninguna cámara (para el sondeo frecuente).
        """
        try:
            if probe:
                active = self.camera_source
                skip = active.index if active and active.type == "local" else None
                available = detect_local_cameras(force_refresh=force_refresh, skip_index=skip)
            else:
                available = _camera_cache["cameras"]
        except Exception as e:
            logger.error("Camera detection failed: %s", e)
            available = []
        return {
            "current": {
                "type": self.camera_source.type,
                "index": self.camera_source.index,
                "url": mask_url(self.camera_source.url),
                "name": self.camera_source.name
            } if self.camera_source else None,
            "available_local": available,
            "error": self.camera_error
        }

    def _purge_log(self):
        """Borra las capturas caducadas y quita del registro las entradas cuya imagen ya no existe."""
        self._last_purge = time.time()
        removed = purge_old_snapshots()
        if removed:
            self.infractions_log = deque(
                (item for item in self.infractions_log if os.path.basename(item["image_path"]) not in removed),
                maxlen=self.infractions_log.maxlen)

    def _clear_detections(self):
        """Sin frames no hay detecciones: si se mantuvieran las del último frame, una infracción vista justo
        antes de perder la cámara dejaría la alarma sonando y el borde rojo mientras no hay vídeo."""
        self.active_detections = {"helmet": 0, "no_helmet": 0, "face": 0}
        self.safety_score = 100
        self.fps = 0.0

    def _release_cap(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _process_stream(self):
        prev_time = time.time()
        current_source = self.camera_source
        last_frame_time = time.time()
        next_retry = 0.0
        retry_delay = RECONNECT_MIN_DELAY

        while self.is_running:
            if time.time() - self._last_purge > PURGE_INTERVAL:
                self._purge_log()
            # Cambio de cámara pedido desde la API. La nueva se abre más abajo, fuera del lock:
            # abrir una cámara IP puede tardar segundos y el lock no debe bloquear a set_camera
            with self._camera_lock:
                requested = self.camera_source
                handover, self._handover = self._handover, None
            if requested != current_source:
                current_source = requested
                self._release_cap()
                next_retry = 0.0
                retry_delay = RECONNECT_MIN_DELAY
            if handover is not None:
                source, cap = handover
                if source is current_source:
                    self._cap = LatestFrameReader(cap) if source.type == "ip" else cap
                    self.camera_error = None
                    logger.info("Camera connected: %s", _source_label(source))
                    last_frame_time = time.time()
                    prev_time = time.time()
                else:
                    cap.release()

            if not self._cap or not self._cap.isOpened():
                # Sin cámara abierta (arranque, cambio, fallo o desconexión): reintento con espera creciente
                self._clear_detections()
                if time.time() >= next_retry:
                    cap = _open_capture(current_source)
                    if self.camera_source != current_source:
                        # Se cambió de cámara mientras se abría esta: se descarta
                        if cap:
                            cap.release()
                        continue
                    if cap and cap.isOpened():
                        self._cap = LatestFrameReader(cap) if current_source.type == "ip" else cap
                        self.camera_error = None
                        logger.info("Camera connected: %s", _source_label(current_source))
                        retry_delay = RECONNECT_MIN_DELAY
                        last_frame_time = time.time()
                        prev_time = time.time()
                        continue
                    if cap:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    self.camera_error = f"No se pudo conectar a la cámara: {_source_label(current_source)}"
                    next_retry = time.time() + retry_delay
                    logger.warning("Camera unavailable, retrying in %.0fs", retry_delay)
                    retry_delay = min(retry_delay * 2, RECONNECT_MAX_DELAY)
                time.sleep(0.5)
                continue

            try:
                success, frame = self._cap.read()
            except Exception as e:
                logger.error("Camera read failed: %s", e)
                success = False
                frame = None
            
            if not success or frame is None:
                # Por tiempo y no por número de fallos: en una cámara IP cada lectura fallida
                # puede tardar hasta IP_READ_TIMEOUT_MS
                if time.time() - last_frame_time > FRAME_FAIL_TIMEOUT:
                    logger.warning("No frames for %.0fs, releasing camera", FRAME_FAIL_TIMEOUT)
                    self._release_cap()
                    self.camera_error = "Error de lectura de cámara - cámara desconectada"
                    retry_delay = RECONNECT_MIN_DELAY
                    next_retry = time.time() + retry_delay  # Se reintenta automáticamente
                else:
                    time.sleep(0.01)
                continue

            last_frame_time = time.time()
            frame_start = time.perf_counter()

            # For local cameras, flip horizontally (mirror effect)
            if current_source.type == "local":
                frame = cv2.flip(frame, 1)

            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:
                self.fps = 1.0 / time_diff
            prev_time = current_time

            # Inferencia YOLO
            results = model.predict(
                frame,
                device=device,
                conf=self.conf_threshold,
                verbose=False,
                **precision_kwargs(use_half)
            )

            boxes = results[0].boxes
            # Una sola copia GPU -> CPU por frame en lugar de tres por caja
            coords, confs = boxes.xyxy.int().tolist(), boxes.conf.tolist()
            names = [model.names[c] for c in boxes.cls.int().tolist()]

            # Cada operario cuenta una vez aunque el modelo lo marque como persona, cabeza y casco
            current_counts, highest_infraction_conf, counted_flags = counting.count(names, confs)

            # Solo se dibujan los recuadros que cuentan, para que la imagen coincida con los contadores
            for (x1, y1, x2, y2), conf, name, counted in zip(coords, confs, names, counted_flags):
                if not counted:
                    continue
                color, text = BOX_STYLES[counting.class_category(name)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{text} ({conf:.2f})", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.active_detections = current_counts
            total = current_counts["helmet"] + current_counts["no_helmet"]
            self.safety_score = int((current_counts["helmet"] / total) * 100) if total > 0 else 100

            # Registro de infracciones
            if current_counts["no_helmet"] > 0 and (current_time - self.last_infraction_save) > self.save_cooldown:
                filename = f"inf_{int(current_time)}.jpg"
                filepath = os.path.join(infractions_dir, filename)
                cv2.imwrite(filepath, frame)
                self.last_infraction_save = current_time

                # El registro guarda las últimas 100; la imagen de la que sale se borra del disco
                if len(self.infractions_log) == self.infractions_log.maxlen:
                    evicted = os.path.basename(self.infractions_log[0]["image_path"])
                    try:
                        os.remove(os.path.join(infractions_dir, evicted))
                    except OSError as e:
                        logger.warning("No se pudo borrar la captura %s: %s", evicted, e)

                self.infractions_log.append({
                    "id": int(current_time * 1000),
                    "timestamp": time.strftime("%H:%M:%S", time.localtime(current_time)),
                    "confidence": round(highest_infraction_conf, 2),
                    "image_path": f"/static/infractions/{filename}",
                    "details": f"Infracción detectada: {current_counts['no_helmet']} operario(s) sin protección."
                })

            # El JPEG solo se codifica si alguien está viendo el stream (ahorra CPU)
            if time.time() - self.last_viewer_time < VIEWER_TIMEOUT:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    with self.lock:
                        self.latest_encoded_frame = buffer.tobytes()
                        self.frame_id += 1
            elif self.latest_encoded_frame is not None:
                with self.lock:
                    self.latest_encoded_frame = None  # Evita mostrar un frame antiguo al reconectar

            self.latency_window.append((time.perf_counter() - frame_start) * 1000)

        if self._cap:
            self._cap.release()


pipeline = VideoPipeline()


class SettingsUpdate(BaseModel):
    conf: float


class CameraSelect(BaseModel):
    type: str  # "local" or "ip"
    index: Optional[int] = None
    url: Optional[str] = None
    name: Optional[str] = None


@app.post("/api/settings")
def update_settings(payload: SettingsUpdate):
    if 0.1 <= payload.conf <= 0.95:
        pipeline.conf_threshold = payload.conf
        return {"status": "success", "conf_threshold": pipeline.conf_threshold}
    raise HTTPException(status_code=400, detail="El umbral debe estar entre 0.10 y 0.95")


@app.get("/api/cameras")
def get_cameras(refresh: bool = False):
    """Get available cameras and current selection."""
    return pipeline.get_camera_info(force_refresh=refresh)


@app.post("/api/cameras/refresh")
def refresh_cameras():
    """Force refresh of available cameras."""
    return pipeline.get_camera_info(force_refresh=True)


@app.post("/api/cameras/select")
def select_camera(payload: CameraSelect):
    """Switch to a different camera source."""
    if payload.type == "ip":
        # Solo protocolos de red: evita que OpenCV abra rutas de archivo locales u otros esquemas
        if not payload.url or not payload.url.strip().lower().startswith(ALLOWED_IP_SCHEMES):
            raise HTTPException(status_code=400, detail="La URL de la cámara IP debe empezar por http://, https:// o rtsp://")
    elif payload.type == "local":
        if payload.index is None or payload.index < 0:
            raise HTTPException(status_code=400, detail="Índice de cámara local no válido")
    else:
        raise HTTPException(status_code=400, detail="Tipo de cámara no válido (use 'local' o 'ip')")

    source = CameraSource(**payload.model_dump())
    if pipeline.set_camera(source):
        return {"status": "success", "camera": pipeline.get_camera_info(probe=False)["current"]}
    raise HTTPException(status_code=400, detail="No se pudo conectar a la cámara seleccionada")


@app.get("/api/stats")
def get_stats(request: Request):
    base_url = str(request.base_url).rstrip("/")
    # Convierte rutas relativas a absolutas basadas en la petición actual (sobre copias)
    infractions = [
        {**item, "image_url": f"{base_url}{item['image_path']}"}
        for item in list(pipeline.infractions_log)[-12:]
    ]

    camera_info = pipeline.get_camera_info(probe=False)

    # Latencia de procesamiento del servidor: lectura del frame -> frame anotado y codificado (últimos 100 frames)
    window = sorted(pipeline.latency_window)
    processing_ms = round(sum(window) / len(window), 1) if window else 0.0
    processing_p95_ms = round(window[int(0.95 * (len(window) - 1))], 1) if window else 0.0

    return {
        "processing_ms": processing_ms,
        "processing_p95_ms": processing_p95_ms,
        "fps": round(pipeline.fps, 1),
        "device": device.upper(),
        "weights_format": os.path.basename(weights_path).split('.')[-1].upper(),
        "active_detections": pipeline.active_detections,
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
    uvicorn.run(app, host=os.environ.get("GUARDWATCH_HOST", "0.0.0.0"), port=8000)