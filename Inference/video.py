"""Modelo y bucle de vídeo: carga del detector, lectura de la cámara, detección, conteo y registro."""
import gc
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Dict, Optional
import cv2
import torch
import cameras
import counting
import evidence
import profiles
from runtime import Runtime, engine_needed, load_model, precision_kwargs

logger = logging.getLogger("guardwatch")

current_dir = os.path.dirname(os.path.abspath(__file__))

VIEWER_TIMEOUT = 2.0  # Segundos sin pedir frames tras los que se deja de codificar el stream
RECONNECT_MIN_DELAY = 2.0 # Espera inicial entre reintentos de conexión (s)
RECONNECT_MAX_DELAY = 15.0  # Espera máxima entre reintentos (s)
FRAME_FAIL_TIMEOUT = 2.0  # Segundos sin frames válidos tras los que se da una cámara local por desconectada
# Las cámaras IP pueden enviar pocos frames (modos de bajo consumo, cámaras de instantáneas, cortes breves de
# Wi-Fi): se espera al menos el tiempo de lectura de FFmpeg antes de darlas por desconectadas
IP_FRAME_FAIL_TIMEOUT = cameras.IP_READ_TIMEOUT_MS / 1000 + 1
PROCESSING_ERROR = "Error al procesar el vídeo (detalles en el registro del servidor)"
PROCESSING_ERROR_AFTER = 2.0  # Segundos fallando seguidos tras los que el error se muestra en el panel
PURGE_INTERVAL = 3600.0  # Cada cuánto se revisan las capturas antiguas mientras el servidor está en marcha (s)
# Una infracción se confirma cuando aparece en VIOLATION_MIN_FRAMES de los últimos VIOLATION_WINDOW frames: un
# falso positivo de un solo frame no guarda una captura ni hace sonar la alarma, y un parpadeo no la anula
VIOLATION_WINDOW = 8
VIOLATION_MIN_FRAMES = 5
FPS_SMOOTHING = 0.1  # Peso del último frame en la media móvil de los FPS

STATE_COLORS = {"ok": (0, 255, 0), "missing": (0, 0, 255), "neutral": (0, 165, 255)}  # BGR: verde, rojo, naranja


class Detector:
    """Modelo cargado en la GPU junto con el perfil que da significado a sus clases.

    El bucle de vídeo lee pipeline.detector una vez por frame, así que cambiar de modo es sustituir la
    referencia: el frame en curso termina con el detector anterior y el siguiente ya usa el nuevo.
    """

    def __init__(self, profile: profiles.Profile, model, runtime: Runtime):
        profiles.check_model(profile, model.names)
        self.profile, self.model, self.runtime = profile, model, runtime
        self.styles = {}  # Color BGR y texto de la etiqueta de cada clase que se dibuja
        for name, rule in profile.classes.items():
            if rule.state == "ignore":
                continue
            if rule.state == "neutral":
                text = rule.label or name
            else:
                label = rule.label or profile.items[rule.item]
                text = f"OK: {label}" if rule.state == "ok" else f"PELIGRO: Sin {label}"
            self.styles[name] = (STATE_COLORS[rule.state], text)

    @property
    def weights_format(self) -> str:
        return os.path.basename(self.runtime.weights_path).split(".")[-1].upper()

    def summary(self) -> Dict:
        return {"id": self.profile.id, "name": self.profile.name, "items": self.profile.items}


def _check_weights(profile: profiles.Profile) -> None:
    """Comprueba que profile.json describe las clases de best.pt antes de generar el motor (varios minutos)."""
    pt = os.path.join(profile.base_dir, "best.pt")
    if os.path.exists(pt):
        from ultralytics import YOLO
        profiles.check_model(profile, YOLO(pt, task="detect").names)  # Solo lee los pesos en la CPU


def _build_engine_apart(base_dir: str) -> None:
    """Genera el motor TensorRT en otro proceso.

    Con el servidor en marcha, la exportación (ONNX, ModelOpt) dentro de este proceso compite por el GIL con el
    hilo de vídeo y lo congela varios segundos. Si falla, load_model(build=False) usa best.pt.
    """
    code = "import sys, runtime; sys.exit(0 if runtime.build_engine(sys.argv[1]) else 1)"
    result = subprocess.run([sys.executable, "-c", code, base_dir], cwd=current_dir)
    if result.returncode != 0:
        logger.warning("No se pudo generar el motor de %s en otro proceso (código %s)", base_dir, result.returncode)


def load_detector(profile_id: str, build_apart: bool = False) -> Detector:
    """Carga el perfil y su modelo en la GPU: motor TensorRT (se genera la primera vez, ~1 min) o best.pt.

    build_apart: generar el motor, si hace falta, en otro proceso (al cambiar de modo con el vídeo en marcha).
    load_model() ya hace una inferencia de prueba, que sirve de warm-up (véase runtime.py).
    """
    profile = profiles.load(profile_id)
    _check_weights(profile)
    if build_apart and engine_needed(profile.base_dir):
        logger.info("Generando el motor TensorRT de %s en otro proceso (tarda aproximadamente un minuto)...",
                    profile.name)
        _build_engine_apart(profile.base_dir)
    model, runtime = load_model(profile.base_dir, build=not build_apart)
    detector = Detector(profile, model, runtime)
    logger.info("Perfil: %s | modelo: %s | dispositivo: %s | FP16: %s", profile.name,
                os.path.basename(runtime.weights_path), runtime.device, runtime.half)
    return detector


class ProfileBusy(Exception):
    """Ya se está cargando otro modo de detección."""


class VideoPipeline:
    def __init__(self, detector: Detector, start: bool = True):
        """start=False no arranca el hilo de vídeo (pruebas que llaman a _process_frame directamente)."""
        self.detector = detector
        self.conf_threshold: float = detector.profile.conf or 0.325
        self.conf_set_by_user = False  # El operador movió el umbral: cambiar de modo ya no lo sustituye
        self.fps: float = 0.0
        self.items: Dict[str, Dict[str, int]] = counting.empty_items(detector.profile)
        self.neutral: int = 0
        self.violations: int = 0
        self.safety_score: int = 100
        self.violation_confirmed: bool = False  # Infracción presente en varios frames seguidos (alarma)
        self._recent_violations = deque(maxlen=VIOLATION_WINDOW)  # ¿Hubo infracción en cada frame reciente?
        self._profile_lock = threading.Lock()
        self.profile_loading: Optional[str] = None  # Modo que se está cargando en segundo plano
        self.profile_error: Optional[str] = None    # Error de la última carga fallida
        self.infractions_log = deque(maxlen=evidence.LOG_SIZE)
        self.last_infraction_save: float = 0.0
        self.save_cooldown: float = 4.0
        
        self.latest_encoded_frame = None
        self.frame_id: int = 0  # Se incrementa con cada frame nuevo codificado
        self.last_viewer_time: float = 0.0  # Última vez que un cliente pidió un frame del stream
        self.latency_window = deque(maxlen=100)  # ms desde que se lee un frame hasta que queda publicado
        self.lock = threading.Lock()
        self.is_running = True
        self.camera_source: Optional[cameras.CameraSource] = cameras.CameraSource(
            type="local", index=0, name="Cámara Local 0")
        self.camera_error: Optional[str] = None
        self._cap = None
        self._camera_lock = threading.Lock()
        self._handover = None  # (fuente, cámara ya abierta) que set_camera deja para el bucle
        self._retry_requested = False  # set_camera pide reconectar la cámara actual sin esperar
        evidence.purge_old_snapshots()
        # Las infracciones de ejecuciones anteriores siguen en el panel
        self.infractions_log.extend(evidence.load_log())
        self._last_purge = time.time()
        self._last_error_log = 0.0
        self._failing_since = None  # Inicio de una racha de fallos al procesar frames
        # Estado del bucle de vídeo: _process_stream les da su valor real al arrancar el hilo
        self._current_source: Optional[cameras.CameraSource] = None
        self._next_retry = 0.0
        self._retry_delay = RECONNECT_MIN_DELAY
        self._last_frame_time = 0.0
        self._prev_time = 0.0

        self.thread = threading.Thread(target=self._process_stream, daemon=True)
        if start:
            self.thread.start()

    def stop(self):
        """Termina el hilo de vídeo y libera la cámara (al cerrar el servidor)."""
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join(timeout=cameras.CAMERA_TEST_TIMEOUT)

    @staticmethod
    def _same_source(a: Optional[cameras.CameraSource], b: Optional[cameras.CameraSource]) -> bool:
        if a is None or b is None:
            return False
        return a.type == b.type and a.index == b.index and a.url == b.url

    def set_camera(self, source: cameras.CameraSource) -> bool:
        """Cambia la fuente de la cámara. Devuelve True si lo consigue."""
        if self._same_source(source, self.camera_source):
            if self.camera_error:
                # La cámara actual está fallando: elegirla de nuevo pide reconectar ya, sin esperar al
                # siguiente reintento (que puede tardar hasta RECONNECT_MAX_DELAY)
                with self._camera_lock:
                    self._retry_requested = True
            return True

        # La prueba (con tiempo máximo) se hace fuera del lock para no bloquear el bucle de captura
        cap = cameras.connect_camera(source)
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
        """Devuelve la información de la cámara actual y las cámaras disponibles.

        probe=False devuelve la lista en caché sin abrir ninguna cámara (para el sondeo frecuente).
        """
        try:
            if probe:
                active = self.camera_source
                skip = active.index if active and active.type == "local" else None
                available = cameras.detect_local_cameras(force_refresh=force_refresh, skip_index=skip)
            else:
                available = cameras.camera_cache["cameras"]
        except Exception as e:
            logger.error("Falló la detección de cámaras: %s", e)
            available = []
        return {
            "current": {
                "type": self.camera_source.type,
                "index": self.camera_source.index,
                "url": cameras.mask_url(self.camera_source.url),
                "name": self.camera_source.name
            } if self.camera_source else None,
            "available_local": available,
            "error": self.camera_error
        }

    def _purge_log(self):
        """Borra las capturas caducadas y quita del registro las entradas cuya imagen ya no existe."""
        self._last_purge = time.time()
        try:
            removed = evidence.purge_old_snapshots()
        except OSError as e:
            # Se ejecuta en el único hilo de proceso: un fallo de disco no debe terminarlo
            logger.error("No se pudieron revisar las capturas antiguas: %s", e)
            return
        if removed:
            self.infractions_log = deque(
                (item for item in self.infractions_log if os.path.basename(item["image_path"]) not in removed),
                maxlen=self.infractions_log.maxlen)
            evidence.save_log(self.infractions_log)

    def _on_processing_error(self, frame):
        """Tras un fallo al procesar un frame, el panel no debe seguir mostrando el último resultado bueno.

        Se borran los contadores (si no, una infracción anterior mantendría la alarma), se sigue enviando el
        vídeo sin anotar y, si el fallo dura más de PROCESSING_ERROR_AFTER segundos, se muestra como error.
        """
        self._clear_detections()
        now = time.time()
        if self._failing_since is None:
            self._failing_since = now
        elif now - self._failing_since > PROCESSING_ERROR_AFTER and not self.camera_error:
            self.camera_error = PROCESSING_ERROR
        if now - self.last_viewer_time < VIEWER_TIMEOUT:
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    with self.lock:
                        self.latest_encoded_frame = buffer.tobytes()
                        self.frame_id += 1
            except Exception:
                pass

    def _clear_detections(self):
        """Sin frames no hay detecciones: si se mantuvieran las del último frame, una infracción vista justo
        antes de perder la cámara dejaría la alarma sonando y el borde rojo mientras no hay vídeo."""
        self.items = counting.empty_items(self.detector.profile)
        self.neutral = 0
        self.violations = 0
        self.safety_score = 100
        self.violation_confirmed = False
        self._recent_violations.clear()
        self.fps = 0.0

    def select_profile(self, profile_id: str) -> bool:
        """Empieza a cargar otro modo de detección en segundo plano. Devuelve False si ya es el actual.

        El modelo actual sigue procesando vídeo mientras tanto; generar el motor TensorRT la primera vez tarda
        alrededor de un minuto y reduce los FPS durante ese tiempo. Lanza KeyError si el modo no existe y
        ProfileBusy si ya se está cargando otro.
        """
        if profile_id not in profiles.discover():  # Solo carpetas instaladas: el id nunca se usa como ruta
            raise KeyError(profile_id)
        with self._profile_lock:
            if self.profile_loading is not None:
                raise ProfileBusy(self.profile_loading)
            if profile_id == self.detector.profile.id:
                self.profile_error = None
                return False
            self.profile_loading = profile_id
            self.profile_error = None
        threading.Thread(target=self._load_profile, args=(profile_id,), daemon=True).start()
        return True

    def _load_profile(self, profile_id: str):
        try:
            new = load_detector(profile_id, build_apart=True)
        except Exception as e:
            # Se conserva el modelo actual: el panel muestra el error y el vídeo no se interrumpe
            logger.exception("No se pudo cargar el modo %s", profile_id)
            with self._profile_lock:
                self.profile_error = f"No se pudo cargar el modo {profile_id}: {e}"
                self.profile_loading = None
            return
        old, self.detector = self.detector, new
        if new.profile.conf is not None and not self.conf_set_by_user:
            self.conf_threshold = new.profile.conf  # Umbral recomendado para este modelo
        self._clear_detections()
        logger.info("Modo de detección cambiado: %s -> %s", old.profile.name, new.profile.name)
        with self._profile_lock:
            self.profile_loading = None
        # Se libera el modelo anterior cuando termina el frame que aún pudiera estar usándolo
        del old
        time.sleep(1.0)
        gc.collect()
        torch.cuda.empty_cache()

    def _release_cap(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _process_frame(self, frame, source: cameras.CameraSource, current_time: float, frame_start: float):
        """Detección, dibujo, registro de infracciones y codificación de un frame."""
        # En las cámaras locales se voltea horizontalmente (efecto espejo)
        if source.type == "local":
            frame = cv2.flip(frame, 1)

        # Una sola lectura por frame: si se cambia de modo a mitad, este frame termina con el detector anterior
        detector = self.detector
        results = detector.model.predict(
            frame,
            device=detector.runtime.device,
            conf=self.conf_threshold,
            verbose=False,
            **precision_kwargs(detector.runtime.half)
        )

        boxes = results[0].boxes
        # Una sola copia GPU -> CPU por frame en lugar de tres por caja
        coords, confs = boxes.xyxy.int().tolist(), boxes.conf.tolist()
        names = [detector.model.names[c] for c in boxes.cls.int().tolist()]

        # Cada operario cuenta una vez aunque el modelo lo marque como persona, cabeza y casco
        counts = counting.count(detector.profile, names, confs, coords)

        # Solo se dibujan los recuadros que cuentan, para que la imagen coincida con los contadores
        self._draw_boxes(frame, detector, coords, confs, names, counts.flags)

        # Si se cambió de modo durante este frame, sus contadores ya no corresponden al modo actual
        current_mode = detector is self.detector
        if current_mode:
            self.items, self.neutral = counts.items, counts.neutral
            self.violations, self.safety_score = counts.violations, counts.safety_score
            self._recent_violations.append(counts.violations > 0)
            self.violation_confirmed = sum(self._recent_violations) >= VIOLATION_MIN_FRAMES

        # Registro de infracciones: solo las confirmadas, y con un frame en el que la infracción se ve
        if current_mode and self.violation_confirmed and counts.violations > 0 and \
                (current_time - self.last_infraction_save) > self.save_cooldown:
            self._record_infraction(frame, detector, counts, current_time)

        self._publish_frame(frame)
        self.latency_window.append((time.perf_counter() - frame_start) * 1000)

    @staticmethod
    def _draw_boxes(frame, detector: Detector, coords, confs, names, flags):
        """Dibuja sobre el frame solo los recuadros que cuentan, para que la imagen coincida con los contadores."""
        for (x1, y1, x2, y2), conf, name, counted in zip(coords, confs, names, flags):
            if not counted:
                continue
            color, text = detector.styles[name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{text} ({conf:.2f})", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _record_infraction(self, frame, detector: Detector, counts, current_time: float):
        """Guarda la captura de la infracción y la añade al registro (si está lleno, borra la más antigua)."""
        # /static no tiene autenticación: el sufijo aleatorio impide adivinar la dirección de una captura
        filename = f"inf_{int(current_time)}_{secrets.token_hex(8)}.jpg"
        filepath = os.path.join(evidence.infractions_dir, filename)
        self.last_infraction_save = current_time
        if not cv2.imwrite(filepath, frame):
            # Disco lleno o carpeta sin permisos: sin imagen no se registra una entrada con la evidencia rota
            logger.error("No se pudo guardar la captura de la infracción en %s", filepath)
            return
        # El registro guarda las últimas 100; la imagen de la que sale se borra del disco
        if len(self.infractions_log) == self.infractions_log.maxlen:
            evicted = os.path.basename(self.infractions_log[0]["image_path"])
            try:
                os.remove(os.path.join(evidence.infractions_dir, evicted))
            except OSError as e:
                logger.warning("No se pudo borrar la captura %s: %s", evicted, e)

        self.infractions_log.append({
            "id": int(current_time * 1000),
            "date": time.strftime("%Y-%m-%d", time.localtime(current_time)),
            "timestamp": time.strftime("%H:%M:%S", time.localtime(current_time)),
            "confidence": round(counts.highest_violation, 2),
            "image_path": f"/static/infractions/{filename}",
            "profile": detector.profile.name,
            "details": evidence.missing_details(detector.profile, counts.items),
        })
        evidence.save_log(self.infractions_log)

    def _publish_frame(self, frame):
        """Codifica el JPEG para el stream, solo si alguien lo está viendo (ahorra CPU)."""
        if time.time() - self.last_viewer_time < VIEWER_TIMEOUT:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self.lock:
                    self.latest_encoded_frame = buffer.tobytes()
                    self.frame_id += 1
        elif self.latest_encoded_frame is not None:
            with self.lock:
                self.latest_encoded_frame = None  # Evita mostrar un frame antiguo al reconectar

    def _take_camera_request(self):
        """Recoge el cambio de cámara o la petición de reconectar que haya dejado set_camera."""
        # La cámara nueva se abre fuera del lock: abrir una cámara IP puede tardar segundos y el lock no
        # debe bloquear a set_camera
        with self._camera_lock:
            requested = self.camera_source
            handover, self._handover = self._handover, None
            retry_now, self._retry_requested = self._retry_requested, False
        if retry_now:
            self._next_retry = 0.0
            self._retry_delay = RECONNECT_MIN_DELAY
        if requested != self._current_source:
            self._current_source = requested
            self._release_cap()
            self._clear_detections()  # Los contadores y la ventana de infracciones eran de la cámara anterior
            self._next_retry = 0.0
            self._retry_delay = RECONNECT_MIN_DELAY
        if handover is not None:
            source, cap = handover
            if source is self._current_source:
                self._clear_detections()  # Nada de la cámara anterior debe confirmar una infracción de esta
                self._use_camera(source, cap)
            else:
                cap.release()

    def _use_camera(self, source: cameras.CameraSource, cap):
        """Empieza a leer de la cámara ya abierta: la guarda, borra el error y reinicia los tiempos del bucle."""
        self._cap = cameras.LatestFrameReader(cap) if source.type == "ip" else cap
        self.camera_error = None
        logger.info("Cámara conectada: %s", cameras.source_label(source))
        self._last_frame_time = time.time()
        self._prev_time = time.time()

    def _reconnect(self):
        """Sin cámara abierta (arranque, cambio, fallo o desconexión): reintento con espera creciente."""
        source = self._current_source
        self._clear_detections()
        if time.time() >= self._next_retry:
            cap = cameras.open_capture(source)
            if self.camera_source != source:
                # Se cambió de cámara mientras se abría esta: se descarta
                if cap:
                    cap.release()
                return
            if cap and cap.isOpened():
                self._retry_delay = RECONNECT_MIN_DELAY
                self._use_camera(source, cap)
                return
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self.camera_error = f"No se pudo conectar a la cámara: {cameras.source_label(source)}"
            self._next_retry = time.time() + self._retry_delay
            logger.warning("Cámara no disponible, se reintenta en %.0fs", self._retry_delay)
            self._retry_delay = min(self._retry_delay * 2, RECONNECT_MAX_DELAY)
        time.sleep(0.5)

    def _read_frame(self):
        """Lee un frame de la cámara. Devuelve None si falla (y la libera si lleva un tiempo sin dar frames)."""
        try:
            success, frame = self._cap.read()
        except Exception as e:
            logger.error("Falló la lectura de la cámara: %s", e)
            success = False
            frame = None

        if not success or frame is None:
            # Por tiempo y no por número de fallos: en una cámara IP cada lectura fallida
            # puede tardar hasta cameras.IP_READ_TIMEOUT_MS
            fail_timeout = IP_FRAME_FAIL_TIMEOUT if self._current_source.type == "ip" else FRAME_FAIL_TIMEOUT
            if time.time() - self._last_frame_time > fail_timeout:
                logger.warning("Sin frames durante %.0fs, se libera la cámara", fail_timeout)
                self._release_cap()
                self.camera_error = "Error de lectura de cámara - cámara desconectada"
                self._retry_delay = RECONNECT_MIN_DELAY
                self._next_retry = time.time() + self._retry_delay  # Se reintenta automáticamente
            else:
                time.sleep(0.01)
            return None
        return frame

    def _handle_frame(self, frame):
        """Procesa un frame leído: mide el tiempo, lo analiza y, si falla, deja el error para el panel."""
        source = self._current_source
        self._last_frame_time = time.time()
        frame_start = time.perf_counter()

        current_time = time.time()
        time_diff = current_time - self._prev_time
        self._prev_time = current_time

        try:
            self._process_frame(frame, source, current_time, frame_start)
        except Exception:
            # Un error inesperado (CUDA, TensorRT, OpenCV...) no debe terminar el único hilo de proceso
            if time.time() - self._last_error_log > 10:
                logger.exception("Error al procesar un frame; se continúa con el siguiente")
                self._last_error_log = time.time()
            self._on_processing_error(cv2.flip(frame, 1) if source.type == "local" else frame)
            time.sleep(0.1)
        else:
            self._after_good_frame(time_diff)

    def _after_good_frame(self, time_diff: float):
        """Actualiza los FPS y da por recuperado un error de procesamiento anterior."""
        if time_diff > 0:
            # Media móvil: con 1 / Δt de un solo frame la cifra del panel saltaba de un valor a otro
            instant = 1.0 / time_diff
            self.fps = instant if self.fps == 0 else FPS_SMOOTHING * instant + (1 - FPS_SMOOTHING) * self.fps
        if self._failing_since is not None:  # El procesamiento se ha recuperado
            self._failing_since = None
            if self.camera_error == PROCESSING_ERROR:
                self.camera_error = None

    def _process_stream(self):
        """Bucle del hilo de vídeo: atiende los cambios de cámara, reconecta si hace falta y procesa frames."""
        self._prev_time = time.time()
        self._current_source = self.camera_source
        self._last_frame_time = time.time()
        self._next_retry = 0.0
        self._retry_delay = RECONNECT_MIN_DELAY

        while self.is_running:
            if time.time() - self._last_purge > PURGE_INTERVAL:
                self._purge_log()
            self._take_camera_request()
            if not self._cap or not self._cap.isOpened():
                self._reconnect()
                continue
            frame = self._read_frame()
            if frame is not None:
                self._handle_frame(frame)

        if self._cap:
            self._cap.release()
