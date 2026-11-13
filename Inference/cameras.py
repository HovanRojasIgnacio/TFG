"""Cámaras: detección de las locales, apertura, lectura del último frame y enmascarado de las URL.

No conoce el pipeline de vídeo ni la API: solo abre cámaras y entrega frames.
"""
import logging
import re
import threading
import time
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit
import cv2
from pydantic import BaseModel

logger = logging.getLogger("guardwatch")

# Caché de detección de cámaras
camera_cache = {"cameras": [], "last_update": 0}
_camera_cache_lock = threading.Lock()
CAMERA_CACHE_TTL = 10.0  # La caché dura 10 segundos
ALLOWED_IP_SCHEMES = ("http://", "https://", "rtsp://")  # Esquemas admitidos para cámaras IP
IP_OPEN_TIMEOUT_MS = 5000  # Tiempo máximo para conectar con una cámara IP
IP_READ_TIMEOUT_MS = 5000  # Tiempo máximo esperando un frame de una cámara IP
CAMERA_TEST_TIMEOUT = (IP_OPEN_TIMEOUT_MS + IP_READ_TIMEOUT_MS) / 1000 + 1  # Prueba al cambiar de cámara (s)


# Una clave que contiene alguna de estas partes se considera secreta (password, loginPas, access_token, api_key...)
SECRET_KEY_PARTS = ("pass", "pwd", "token", "key", "auth", "secret", "user", "usr", "login", "sign")
_KEY_VALUE = re.compile(r"([^=&;/?#]+)=([^&;/?#]*)")


def _mask_pairs(text: str) -> str:
    """Sustituye por *** el valor de cada par clave=valor cuya clave parece secreta."""
    return _KEY_VALUE.sub(lambda m: f"{m.group(1)}=***"
                          if any(part in m.group(1).lower() for part in SECRET_KEY_PARTS) else m.group(0), text)


def mask_url(url: Optional[str]) -> Optional[str]:
    """Oculta usuario, contraseña y tokens de la URL de una cámara antes de mostrarla o registrarla.

    Cualquier equipo de la red puede consultar la API, así que las credenciales nunca salen del servidor. Se
    enmascaran el usuario y la contraseña de la dirección y los pares clave=valor secretos tanto en la consulta
    como en la ruta: muchas cámaras y grabadores los ponen en la ruta, como en
    rtsp://host:554/user=admin&password=...&channel=1.
    """
    if not url:
        return url
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            netloc = "***@" + netloc.rsplit("@", 1)[1]
        return urlunsplit(parts._replace(netloc=netloc, path=_mask_pairs(parts.path),
                                         query=_mask_pairs(parts.query), fragment=_mask_pairs(parts.fragment)))
    except ValueError:
        return "URL no válida"


def _camera_entry(index: int, width: int, height: int) -> Dict:
    """Ficha de una cámara local para la lista que ve el panel."""
    return {"index": index, "name": f"Cámara Local {index}", "type": "local", "width": width, "height": height}


def _probe_local_camera(index: int) -> Optional[Dict]:
    """Abre la cámara local `index` y devuelve su ficha, o None si no está disponible."""
    cap = None
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return _camera_entry(index, width if width > 0 else 640, height if height > 0 else 480)
    except Exception as e:
        logger.warning("Error al detectar la cámara %s: %s", index, e)
        return None
    finally:
        if cap:
            try:
                cap.release()
            except Exception:
                pass


def detect_local_cameras(max_cameras: int = 5, force_refresh: bool = False,
                         skip_index: Optional[int] = None) -> List[Dict]:
    """Detecta las cámaras locales disponibles, con caché.

    skip_index: índice de la cámara en uso por el pipeline; no se reabre para no interrumpirla.
    """
    global camera_cache

    with _camera_cache_lock:
        now = time.time()
        # La caché también vale cuando no hay cámaras (lista vacía), si ya se sondeó antes
        if not force_refresh and camera_cache["last_update"] and (now - camera_cache["last_update"] < CAMERA_CACHE_TTL):
            return camera_cache["cameras"]

        cameras = []
        consecutive_failures = 0
        for i in range(max_cameras):
            camera = _camera_entry(i, 640, 480) if i == skip_index else _probe_local_camera(i)
            if camera:
                cameras.append(camera)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            if consecutive_failures >= 2:
                break

        camera_cache = {"cameras": cameras, "last_update": now}
        return cameras


class CameraSource(BaseModel):
    type: str  # "local" o "ip"
    index: Optional[int] = None
    url: Optional[str] = None
    name: Optional[str] = None


def source_label(source: CameraSource) -> str:
    """Nombre de la cámara para mensajes y registros (sin credenciales)."""
    return source.name or mask_url(source.url) or f"Cámara Local {source.index}"


def open_capture(source: CameraSource):
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
        logger.error("No se pudo abrir la cámara: %s", e)
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
            logger.error("Falló la lectura de la cámara IP: %s", e)
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


def connect_camera(source: CameraSource, timeout: float = CAMERA_TEST_TIMEOUT):
    """Abre la cámara y lee un frame para confirmar que funciona, con tiempo máximo.

    Devuelve la cámara ya abierta (para que el pipeline la use sin volver a conectar: algunas
    cámaras IP solo admiten una sesión y conectar dos veces duplica la espera) o None si falla.
    """
    result = {}
    result_lock = threading.Lock()

    def connect():
        cap = open_capture(source)
        ok = False
        try:
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
        except Exception as e:
            logger.warning("Falló la prueba de la cámara: %s", e)
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
        logger.warning("La prueba de conexión de la cámara superó el tiempo máximo (%ss)", timeout)
    return None
