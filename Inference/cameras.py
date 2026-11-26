"""Cámaras: detección de las locales, apertura, lectura del último frame, enmascarado de las URL y la cámara que
lee el bucle de vídeo (CameraManager: cambio de fuente, reconexión y detección de una cámara perdida).

No conoce la detección ni la API: solo abre cámaras y entrega frames.
"""
import logging
import re
import threading
import time
from typing import Annotated, Callable, ClassVar, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlsplit, urlunsplit
import cv2
from pydantic import BaseModel, Discriminator, Tag, field_validator

logger = logging.getLogger("guardwatch")

CAMERA_LIST_TTL = 10.0  # Segundos durante los que se reutiliza la lista de cámaras locales detectadas
ALLOWED_IP_SCHEMES = ("http://", "https://", "rtsp://")  # Esquemas admitidos para cámaras IP
IP_OPEN_TIMEOUT_MS = 5000  # Tiempo máximo para conectar con una cámara IP
IP_READ_TIMEOUT_MS = 5000  # Tiempo máximo esperando un frame de una cámara IP
CAMERA_TEST_TIMEOUT = (IP_OPEN_TIMEOUT_MS + IP_READ_TIMEOUT_MS) / 1000 + 1  # Prueba al cambiar de cámara (s)
RECONNECT_MIN_DELAY = 2.0   # Espera inicial entre reintentos de conexión (s)
RECONNECT_MAX_DELAY = 15.0  # Espera máxima entre reintentos (s)
FRAME_FAIL_TIMEOUT = 2.0    # Segundos sin frames válidos tras los que se da una cámara local por desconectada
# Las cámaras IP pueden enviar pocos frames (modos de bajo consumo, cámaras de instantáneas, cortes breves de
# Wi-Fi): se espera al menos el tiempo de lectura de FFmpeg antes de darlas por desconectadas
IP_FRAME_FAIL_TIMEOUT = IP_READ_TIMEOUT_MS / 1000 + 1


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


def release(cap) -> None:
    """Cierra una cámara (VideoCapture o LatestFrameReader); un fallo del driver al cerrarla no se propaga."""
    if cap is None:
        return
    try:
        cap.release()
    except Exception as e:
        logger.debug("Falló el cierre de la cámara: %s", e)


def local_name(index: Optional[int]) -> str:
    """Nombre por defecto de una cámara local (el único sitio donde se construye)."""
    return f"Cámara Local {index}"


def _camera_entry(index: int, width: int, height: int) -> Dict:
    """Ficha de una cámara local para la lista que ve el panel."""
    return {"index": index, "name": local_name(index), "type": "local", "width": width, "height": height}


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
        release(cap)


def detect_local_cameras(max_cameras: int = 5, skip_index: Optional[int] = None) -> List[Dict]:
    """Detecta las cámaras locales disponibles (abre cada índice, así que tarda: CameraManager guarda la lista).

    skip_index: índice de la cámara que el pipeline tiene abierta; no se reabre para no interrumpirla y
    aparece siempre en la lista, aunque falten las de los índices anteriores.
    """
    cameras = []
    consecutive_failures = 0
    last_known = -1 if skip_index is None else skip_index
    for i in range(max(max_cameras, last_known + 1)):
        camera = _camera_entry(i, 640, 480) if i == skip_index else _probe_local_camera(i)
        if camera:
            cameras.append(camera)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        # Dos índices seguidos sin cámara: no hay más, salvo la que está en uso en un índice posterior
        if consecutive_failures >= 2 and i >= last_known:
            break
    return cameras


class LocalCamera(BaseModel):
    """Una cámara conectada al equipo (integrada o USB), por su índice."""
    type: Literal["local"] = "local"
    index: int
    name: Optional[str] = None
    mirror: ClassVar[bool] = True  # Se voltea horizontalmente: el operador que se mira se ve como en un espejo

    @field_validator("index")
    @classmethod
    def _valid_index(cls, index: int) -> int:
        if index < 0:
            raise ValueError("Índice de cámara local no válido")
        return index

    @property
    def label(self) -> str:
        """Nombre para el panel, los mensajes y los registros."""
        return self.name or local_name(self.index)

    @property
    def key(self) -> Tuple:
        """Qué cámara es (el nombre no cuenta)."""
        return self.type, self.index

    @property
    def fail_timeout(self) -> float:
        """Segundos sin frames válidos tras los que se da por desconectada."""
        return FRAME_FAIL_TIMEOUT

    def summary(self) -> Dict:
        return {"type": self.type, "index": self.index, "url": None, "name": self.label}

    def open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def reader(self, cap):
        """Lo que lee el bucle de vídeo: la propia cámara."""
        return cap


class IpCamera(BaseModel):
    """Una cámara de red (o un móvil con una aplicación de cámara IP), por su URL."""
    type: Literal["ip"] = "ip"
    url: str
    name: Optional[str] = None
    mirror: ClassVar[bool] = False

    @field_validator("url")
    @classmethod
    def _network_url(cls, url: str) -> str:
        # Solo protocolos de red: evita que OpenCV abra rutas de archivo locales u otros esquemas
        url = url.strip()
        if not url.lower().startswith(ALLOWED_IP_SCHEMES):
            raise ValueError("La URL de la cámara IP debe empezar por http://, https:// o rtsp://")
        return url

    @property
    def label(self) -> str:
        """Nombre para el panel, los mensajes y los registros (sin credenciales)."""
        return self.name or mask_url(self.url)

    @property
    def key(self) -> Tuple:
        return self.type, self.url

    @property
    def fail_timeout(self) -> float:
        return IP_FRAME_FAIL_TIMEOUT

    def summary(self) -> Dict:
        return {"type": self.type, "index": None, "url": mask_url(self.url), "name": self.label}

    def open(self):
        """Con FFmpeg y tiempos máximos explícitos: sin ellos, una URL que no responde bloquea unos 30 s en
        FFmpeg y después OpenCV prueba otros backends (hasta ~2 min)."""
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, IP_OPEN_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, IP_READ_TIMEOUT_MS,
        ])

    def reader(self, cap):
        """Lo que lee el bucle de vídeo: siempre el último frame (véase LatestFrameReader)."""
        return LatestFrameReader(cap)


def _camera_type(value) -> Optional[str]:
    """El tipo de una cámara, venga como JSON de la API o ya construida."""
    return value.get("type") if isinstance(value, dict) else getattr(value, "type", None)


# Una cámara de cualquiera de los dos tipos; es el cuerpo de POST /api/cameras/select
CameraSource = Annotated[
    Union[Annotated[LocalCamera, Tag("local")], Annotated[IpCamera, Tag("ip")]],
    Discriminator(_camera_type, custom_error_type="camera_type",
                  custom_error_message="Tipo de cámara no válido (use 'local' o 'ip')")]


def open_capture(source: CameraSource):
    """Abre la cámara indicada. Devuelve el VideoCapture (abierto o no) o None si falla."""
    try:
        return source.open()
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
            release(self._cap)

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
        release(cap)

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


class Backoff:
    """Cuándo se vuelve a intentar conectar: tras cada fallo la espera se duplica, de RECONNECT_MIN_DELAY a
    RECONNECT_MAX_DELAY."""

    def __init__(self):
        self.restart()

    def restart(self, wait: bool = False) -> None:
        """Vuelve a la espera mínima. Se intenta ya, o tras esa espera si wait=True."""
        self.delay = RECONNECT_MIN_DELAY
        self.next_try = time.time() + self.delay if wait else 0.0

    def failed(self) -> float:
        """Un intento ha fallado: el siguiente espera `delay`, que se duplica. Devuelve la espera."""
        wait, self.delay = self.delay, min(self.delay * 2, RECONNECT_MAX_DELAY)
        self.next_try = time.time() + wait
        return wait

    @property
    def due(self) -> bool:
        return time.time() >= self.next_try


class CameraManager:
    """La cámara que lee el bucle de vídeo: cambio de fuente, reconexión con espera creciente y cámara perdida.

    select() e info() se llaman desde la API; read() y close(), solo desde el hilo de vídeo. `on_switch` se llama
    (en el hilo de vídeo) cada vez que se empieza a usar otra cámara o se reconecta: lo que se había detectado
    con la anterior ya no vale.
    """

    def __init__(self, source: CameraSource, on_switch: Callable[[], None] = lambda: None):
        self.source = source                 # La cámara elegida (la que muestra el panel)
        self.error: Optional[str] = None
        self._on_switch = on_switch
        self._lock = threading.Lock()
        # Cámaras locales detectadas la última vez y cuándo (0: aún no se ha sondeado); se reutilizan durante
        # CAMERA_LIST_TTL, también si no había ninguna
        self._local_cameras: List[Dict] = []
        self._local_listed_at = 0.0
        self._list_lock = threading.Lock()
        self._handover = None                # (fuente, cámara ya abierta) que select deja para read
        self._retry_requested = False        # select pide reconectar la cámara actual sin esperar
        # Estado del hilo de vídeo
        self._cap = None
        self._current: Optional[CameraSource] = None  # La que read está usando
        self._backoff = Backoff()
        self._last_frame_time = 0.0

    @property
    def current(self) -> Optional[CameraSource]:
        return self._current

    @property
    def connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def select(self, source: CameraSource) -> bool:
        """Cambia de cámara. Devuelve True si la nueva funciona (se prueba antes de cambiar)."""
        if source.key == self.source.key:
            if self.error:
                # La cámara actual está fallando: elegirla de nuevo pide reconectar ya, sin esperar al
                # siguiente reintento (que puede tardar hasta RECONNECT_MAX_DELAY)
                with self._lock:
                    self._retry_requested = True
            return True

        # La prueba (con tiempo máximo) se hace fuera del lock para no bloquear el bucle de captura
        cap = connect_camera(source)
        if cap is None:
            return False

        with self._lock:
            # El hilo de vídeo detecta el cambio, libera la cámara anterior y usa la ya abierta
            replaced = self._handover
            self._handover = (source, cap)
            self.source = source
            self.error = None
        if replaced is not None:
            release(replaced[1])  # Otro cambio anterior que el bucle aún no había recogido
        return True

    def info(self, force_refresh: bool = False, probe: bool = True) -> Dict:
        """La cámara elegida, las cámaras locales disponibles y el error actual.

        probe=False devuelve la lista en caché sin abrir ninguna cámara (para el sondeo frecuente).
        """
        source = self.source
        try:
            if probe:
                # La cámara local que el bucle tiene abierta no se vuelve a abrir (se interrumpiría). Si no está
                # abierta (aún no conectó o falló) se sondea como las demás: darla por buena mostraba en la
                # lista una cámara que no existe
                in_use = isinstance(source, LocalCamera) and self._cap is not None and self.error is None
                available = self._list_local(force_refresh, source.index if in_use else None)
            else:
                available = self._local_cameras
        except Exception as e:
            logger.error("Falló la detección de cámaras: %s", e)
            available = []
        return {
            "current": source.summary(),
            "available_local": available,
            "error": self.error,
        }

    def _list_local(self, force_refresh: bool, skip_index: Optional[int]) -> List[Dict]:
        """La lista de cámaras locales: la guardada si es reciente; si no, se vuelve a detectar."""
        with self._list_lock:
            now = time.time()
            if force_refresh or not self._local_listed_at or now - self._local_listed_at >= CAMERA_LIST_TTL:
                self._local_cameras = detect_local_cameras(skip_index=skip_index)
                self._local_listed_at = now
            return self._local_cameras

    def read(self):
        """El siguiente frame, ya volteado si la cámara es un espejo, o None si no lo hay (sin cámara,
        reconectando o un fallo de lectura)."""
        self._take_request()
        if not self.connected:
            self._reconnect()
            return None
        frame = self._read_frame()
        return cv2.flip(frame, 1) if frame is not None and self._current.mirror else frame

    def close(self):
        self._release()

    def _release(self):
        release(self._cap)
        self._cap = None

    def _take_request(self):
        """Recoge el cambio de cámara o la petición de reconectar que haya dejado select."""
        # La cámara nueva se abre fuera del lock: abrir una cámara IP puede tardar segundos y el lock no
        # debe bloquear a select
        with self._lock:
            requested = self.source
            handover, self._handover = self._handover, None
            retry_now, self._retry_requested = self._retry_requested, False
        changed = requested is not self._current  # select crea siempre una fuente nueva
        if retry_now or changed:
            self._backoff.restart()
        if changed:
            # Hasta que _use conecte la nueva no hay frames (y el bucle borra lo detectado con la anterior)
            self._current = requested
            self._release()
        if handover is not None:
            source, cap = handover
            if source is self._current:
                self._use(source, cap)
            else:
                release(cap)

    def _use(self, source: CameraSource, cap):
        """Empieza a leer de la cámara ya abierta (la única entrada a una cámara nueva o reconectada)."""
        self._cap = source.reader(cap)
        self.error = None
        self._backoff.restart()
        logger.info("Cámara conectada: %s", source.label)
        self._last_frame_time = time.time()
        self._on_switch()  # Nada de la cámara anterior debe confirmar una infracción de esta

    def _reconnect(self):
        """Sin cámara abierta (arranque, cambio, fallo o desconexión): reintento con espera creciente."""
        if not self._backoff.due:
            time.sleep(0.5)
            return
        source = self._current
        cap = open_capture(source)
        if self.source is not source:
            release(cap)  # Se cambió de cámara mientras se abría esta: se descarta
            return
        if cap is not None and cap.isOpened():
            self._use(source, cap)
            return
        release(cap)
        self.error = f"No se pudo conectar a la cámara: {source.label}"
        logger.warning("Cámara no disponible, se reintenta en %.0fs", self._backoff.failed())
        time.sleep(0.5)

    def _read_frame(self):
        """Lee un frame. Devuelve None si falla (y libera la cámara si lleva un tiempo sin dar frames)."""
        try:
            success, frame = self._cap.read()
        except Exception as e:
            logger.error("Falló la lectura de la cámara: %s", e)
            success, frame = False, None
        if success and frame is not None:
            self._last_frame_time = time.time()
            return frame
        # Por tiempo y no por número de fallos: en una cámara IP cada lectura fallida puede tardar hasta
        # IP_READ_TIMEOUT_MS
        fail_timeout = self._current.fail_timeout
        if time.time() - self._last_frame_time > fail_timeout:
            logger.warning("Sin frames durante %.0fs, se libera la cámara", fail_timeout)
            self._release()
            self.error = "Error de lectura de cámara - cámara desconectada"
            self._backoff.restart(wait=True)  # Se reintenta automáticamente
        else:
            time.sleep(0.01)
        return None
