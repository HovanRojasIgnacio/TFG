"""Modelo y bucle de vídeo: carga del detector, detección, conteo y publicación del frame anotado.

La cámara la gestiona cameras.CameraManager y las capturas de las infracciones, evidence.EvidenceRecorder.
"""
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Dict, Optional
import cv2
import numpy as np
import cameras
import counting
import evidence
import profiles
from runtime import (DEVICE, Builder, Model, build_engine, ensure_weights, free_gpu_memory, load_model, pt_path,
                     tensorrt_available)

logger = logging.getLogger("guardwatch")

current_dir = os.path.dirname(os.path.abspath(__file__))

VIEWER_TIMEOUT = 2.0  # Segundos sin pedir frames tras los que se deja de codificar el stream
PROCESSING_ERROR = "Error al procesar el vídeo (detalles en el registro del servidor)"
PROCESSING_ERROR_AFTER = 2.0  # Segundos fallando seguidos tras los que el error se muestra en el panel
# Una infracción se confirma cuando aparece en VIOLATION_MIN_FRAMES de los últimos VIOLATION_WINDOW frames: un
# falso positivo de un solo frame no guarda una captura ni hace sonar la alarma, y un parpadeo no la anula
VIOLATION_WINDOW = 8
VIOLATION_MIN_FRAMES = 5
FPS_SMOOTHING = 0.1  # Peso del último frame en la media móvil de los FPS

STATE_COLORS = {"ok": (0, 255, 0), "missing": (0, 0, 255), "neutral": (0, 165, 255)}  # BGR: verde, rojo, naranja


def _no_signal_jpeg() -> bytes:
    """Imagen que sustituye al vídeo mientras no hay cámara."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    text, font, scale, thickness = "SIN IMAGEN", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(image, text, ((640 - width) // 2, (480 + height) // 2), font, scale, (200, 200, 200), thickness)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen que sustituye al vídeo")
    return buffer.tobytes()


NO_SIGNAL_JPEG = _no_signal_jpeg()


class Detector:
    """Modelo cargado en la GPU junto con el perfil que da significado a sus clases.

    El bucle de vídeo lee pipeline.detector una vez por frame, así que cambiar de modo es sustituir la
    referencia: el frame en curso termina con el detector anterior y el siguiente ya usa el nuevo.
    """

    def __init__(self, profile: profiles.Profile, model: Model):
        profiles.check_model(profile, model.names)
        self.profile, self.model = profile, model
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

    def summary(self) -> Dict:
        return {"id": self.profile.id, "name": self.profile.name, "items": self.profile.items}


def _check_weights(profile: profiles.Profile) -> None:
    """Descarga best.pt si falta y comprueba que profile.json describe sus clases antes de generar el motor."""
    ensure_weights(profile.base_dir, profile.weights_url, profile.weights_sha256)
    pt = pt_path(profile.base_dir)
    if os.path.exists(pt):
        from ultralytics import YOLO
        profiles.check_model(profile, YOLO(pt, task="detect").names)  # Solo lee los pesos en la CPU


def _build_engine_apart(base_dir: str) -> bool:
    """runtime.build_engine en otro proceso (un runtime.Builder). Devuelve True si genera el motor.

    Con el servidor en marcha, la exportación (ONNX, ModelOpt) dentro de este proceso compite por el GIL con el
    hilo de vídeo y lo congela varios segundos. Si falla, load_model usa best.pt.
    """
    if not (os.path.exists(pt_path(base_dir)) and tensorrt_available()):
        return False  # Fallaría enseguida: no merece la pena arrancar otro proceso con PyTorch y Ultralytics
    logger.info("Generando el motor TensorRT de %s en otro proceso (tarda aproximadamente un minuto)...",
                os.path.basename(base_dir))
    code = "import sys, runtime; sys.exit(0 if runtime.build_engine(sys.argv[1]) else 1)"
    result = subprocess.run([sys.executable, "-c", code, base_dir], cwd=current_dir)
    if result.returncode != 0:
        logger.warning("No se pudo generar el motor de %s en otro proceso (código %s)", base_dir, result.returncode)
    return result.returncode == 0


def load_detector(profile_id: str, build: Builder = build_engine) -> Detector:
    """Carga el perfil y su modelo en la GPU: motor TensorRT (se genera la primera vez, ~1 min) o best.pt.

    build genera el motor si hace falta: en este proceso al arrancar el servidor, y con _build_engine_apart al
    cambiar de modo con el vídeo en marcha. load_model() ya hace una inferencia de prueba, que sirve de warm-up.
    """
    profile = profiles.load(profile_id)
    _check_weights(profile)
    detector = Detector(profile, load_model(profile.base_dir, build))
    logger.info("Perfil: %s | modelo: %s | dispositivo: %s | FP16: %s", profile.name,
                os.path.basename(detector.model.weights_path), DEVICE, detector.model.half)
    return detector


class ProfileBusy(Exception):
    """Ya se está cargando otro modo de detección."""


class VideoPipeline:
    """Bucle de vídeo: lee la cámara (cameras.CameraManager), detecta y cuenta, guarda las evidencias
    (evidence.EvidenceRecorder) y publica el frame anotado para el stream."""

    def __init__(self, detector: Detector, start: bool = True):
        """start=False no arranca el hilo de vídeo (pruebas que llaman a _process_frame directamente)."""
        self.detector = detector
        self.conf_threshold: float = detector.profile.threshold
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

        self.latest_encoded_frame = None
        self.frame_id: int = 0  # Se incrementa con cada frame nuevo codificado
        self.last_viewer_time: float = 0.0  # Última vez que un cliente pidió un frame del stream
        self.latency_window = deque(maxlen=100)  # ms desde que se lee un frame hasta que queda publicado
        self.lock = threading.Lock()
        self.is_running = True
        self._last_error_log = 0.0
        self._failing_since = None  # Inicio de una racha de fallos al procesar frames
        self._processing_failed = False  # La racha dura más de PROCESSING_ERROR_AFTER: se muestra en el panel
        self._prev_time: Optional[float] = None  # Hora del frame anterior (None: el primero de esta cámara)

        self.evidence = evidence.EvidenceRecorder()
        self.camera = cameras.CameraManager(
            cameras.CameraSource(type="local", index=0, name=cameras.local_name(0)), on_switch=self._on_camera_switch)

        self.thread = threading.Thread(target=self._process_stream, daemon=True)
        if start:
            self.thread.start()

    @property
    def camera_error(self) -> Optional[str]:
        """El error de la cámara o, si la cámara va bien pero el procesamiento lleva un rato fallando, ese."""
        return self.camera.error or (PROCESSING_ERROR if self._processing_failed else None)

    def stop(self):
        """Termina el hilo de vídeo y libera la cámara (al cerrar el servidor)."""
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join(timeout=cameras.CAMERA_TEST_TIMEOUT)

    def _clear_detections(self):
        """Borra lo detectado. Sin frames, o con otra cámara u otro modo, una infracción vista antes dejaría la
        alarma sonando y el borde rojo, o confirmaría una infracción que ya no está."""
        self.items = counting.empty_items(self.detector.profile)
        self.neutral = 0
        self.violations = 0
        self.safety_score = 100
        self.violation_confirmed = False
        self._recent_violations.clear()
        self.evidence.reset_episode()
        self.fps = 0.0

    def _on_camera_switch(self):
        """Otra cámara, o la misma reconectada: lo detectado con la anterior ya no vale y los FPS empiezan de cero."""
        self._clear_detections()
        self._prev_time = None

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
            new = load_detector(profile_id, build=_build_engine_apart)
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
        free_gpu_memory()

    def _process_frame(self, frame, source: cameras.CameraSource, current_time: float, frame_start: float):
        """Detección, dibujo, registro de infracciones y codificación de un frame."""
        # En las cámaras locales se voltea horizontalmente (efecto espejo)
        if source.type == "local":
            frame = cv2.flip(frame, 1)

        # Una sola lectura por frame: si se cambia de modo a mitad, este frame termina con el detector anterior
        detector = self.detector
        names, confs, coords = detector.model.detect(frame, self.conf_threshold)

        # Cada operario cuenta una vez aunque el modelo lo marque como persona, cabeza y casco
        counts = counting.count(detector.profile, names, confs, coords)

        # Solo se dibujan los recuadros que cuentan, para que la imagen coincida con los contadores
        self._draw_boxes(frame, detector, coords, confs, names, counts.flags)

        # Si se cambió de modo durante este frame, sus contadores ya no corresponden al modo actual
        if detector is self.detector:
            self.items, self.neutral = counts.items, counts.neutral
            self.violations, self.safety_score = counts.violations, counts.safety_score
            self._recent_violations.append(counts.violations > 0)
            self.violation_confirmed = sum(self._recent_violations) >= VIOLATION_MIN_FRAMES
            self.evidence.update(frame, detector.profile, counts, self.violation_confirmed, current_time)

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

    def _publish_no_signal(self):
        """Sin cámara, el stream muestra «SIN IMAGEN» en lugar del último frame.

        El navegador deja en pantalla el último frame recibido: sin esto, quien miraba el vídeo (o quien se
        conectara después) seguía viendo a las personas y los recuadros de antes de perder la cámara.
        """
        with self.lock:
            if self.latest_encoded_frame is not NO_SIGNAL_JPEG:
                self.latest_encoded_frame = NO_SIGNAL_JPEG
                self.frame_id += 1

    def _handle_frame(self, frame, source: cameras.CameraSource):
        """Procesa un frame leído: lo analiza, actualiza los FPS y, si falla, deja el error para el panel."""
        frame_start = time.perf_counter()
        current_time = time.time()
        time_diff = current_time - self._prev_time if self._prev_time is not None else 0.0
        self._prev_time = current_time

        try:
            self._process_frame(frame, source, current_time, frame_start)
        except Exception:
            self._on_processing_error(cv2.flip(frame, 1) if source.type == "local" else frame)
            return
        if time_diff > 0:
            # Media móvil: con 1 / Δt de un solo frame la cifra del panel saltaba de un valor a otro
            instant = 1.0 / time_diff
            self.fps = instant if self.fps == 0 else FPS_SMOOTHING * instant + (1 - FPS_SMOOTHING) * self.fps
        self._failing_since = None  # El procesamiento funciona (o se ha recuperado)
        self._processing_failed = False

    def _on_processing_error(self, frame):
        """Tras un fallo al procesar un frame (CUDA, TensorRT, OpenCV...), el hilo sigue con el siguiente.

        El panel no debe seguir mostrando el último resultado bueno: se borran los contadores (si no, una
        infracción anterior mantendría la alarma), se sigue enviando el vídeo sin anotar y, si el fallo dura más
        de PROCESSING_ERROR_AFTER segundos, se muestra como error.
        """
        now = time.time()
        if now - self._last_error_log > 10:
            logger.exception("Error al procesar un frame; se continúa con el siguiente")
            self._last_error_log = now
        self._clear_detections()
        if self._failing_since is None:
            self._failing_since = now
        elif now - self._failing_since > PROCESSING_ERROR_AFTER:
            self._processing_failed = True
        try:
            self._publish_frame(frame)
        except Exception:
            pass
        time.sleep(0.1)

    def _process_stream(self):
        """Bucle del hilo de vídeo: limpieza periódica de evidencias, lectura de la cámara y proceso del frame."""
        while self.is_running:
            self.evidence.purge_if_due()
            frame = self.camera.read()
            if frame is not None:
                self._handle_frame(frame, self.camera.current)
            elif not self.camera.connected:
                self._clear_detections()
                self._publish_no_signal()
        self.camera.close()
