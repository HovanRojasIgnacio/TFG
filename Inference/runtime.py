"""El modelo en la GPU: descarga de los pesos, motor TensorRT, carga e inferencia. Solo GPU: no hay modo CPU.

Lo usan el servidor (video.py) y las herramientas (benchmark.py, validate_model.py, export.py y las de
Documentation/tools), así que todas cargan e infieren igual. Orden de preferencia (siempre en la GPU NVIDIA):
  1. best.engine (TensorRT). Si falta, se generó a partir de otros pesos, en otra GPU o con otro TensorRT, o no
     se puede cargar, se genera de nuevo a partir de best.pt (tarda aproximadamente un minuto).
  2. best.pt (PyTorch en FP16), si TensorRT no está instalado o la generación falla.
"""
import gc
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import tempfile
import urllib.request
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
from counting import Detections

logger = logging.getLogger("guardwatch")

IMGSZ = 640
DEVICE = "cuda"
FP16 = {"quantize": 16}  # La versión de Ultralytics de requirements.txt usa `quantize` (la opción `half` ya no existe)

# Genera best.engine en la carpeta de un modo y devuelve True si lo logra: build_engine, o la función de video.py
# que lo genera en otro proceso con el servidor en marcha
Builder = Callable[[str], bool]


def pt_path(base_dir: str) -> str:
    return os.path.join(base_dir, "best.pt")


def engine_path(base_dir: str) -> str:
    return os.path.join(base_dir, "best.engine")


def require_cuda() -> None:
    import torch  # Aquí y no al principio: así las pruebas de la API se ejecutan sin PyTorch (CI)
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GuardWatch AI necesita una GPU NVIDIA con CUDA, pero este Python (torch {torch.__version__}) "
            "no la ve. Instala PyTorch con CUDA (véase el README) y arranca el servidor con el entorno "
            "virtual de Inference/.venv."
        )


def free_gpu_memory() -> None:
    """Devuelve a CUDA la memoria de los modelos que ya no se usan (al cambiar de modo de detección)."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class Model(NamedTuple):
    """Un modelo YOLO con sus pesos y su precisión: la inferencia es la misma en el servidor y en las herramientas."""
    yolo: Any            # ultralytics.YOLO
    weights_path: str
    half: bool           # FP16 (best.pt en el servidor). Con False, FP32; un motor TensorRT lleva su precisión dentro

    @property
    def names(self):
        return self.yolo.names

    @property
    def weights_format(self) -> str:
        """ENGINE o PT, para el panel."""
        return os.path.basename(self.weights_path).split(".")[-1].upper()

    def predict(self, image, conf: float):
        return self.yolo.predict(image, device=DEVICE, conf=conf, verbose=False, **(FP16 if self.half else {}))

    def detect(self, image, conf: float) -> Detections:
        """Los recuadros de una imagen, con una sola copia GPU -> CPU por imagen en lugar de tres por recuadro."""
        boxes = self.predict(image, conf)[0].boxes
        return Detections([self.names[c] for c in boxes.cls.int().tolist()], boxes.conf.tolist(),
                          boxes.xyxy.int().tolist())


def tensorrt_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


def _weights_digest(path: str) -> str:
    """SHA-256 de un archivo de pesos (unos 50 ms para los 20 MB de best.pt)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weights(base_dir: str, url: Optional[str], sha256: Optional[str]) -> None:
    """Descarga best.pt en base_dir si falta y el perfil dice de dónde (los pesos no se guardan en git).

    El archivo se descarga aparte y solo se coloca si su SHA-256 es el que indica profile.json: una descarga
    cortada o un archivo cambiado en el servidor no deja unos pesos rotos o distintos en la carpeta.
    """
    pt = pt_path(base_dir)
    if os.path.exists(pt) or not url:
        return
    logger.info("Descargando los pesos del modo desde %s (unos 20 MB)...", url)
    staged = pt + ".download"
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(staged, "wb") as f:
            shutil.copyfileobj(response, f)
        got = _weights_digest(staged)
        if got != sha256:
            raise ValueError(f"el SHA-256 del archivo descargado ({got}) no es el de profile.json ({sha256})")
        os.replace(staged, pt)
    except Exception as e:
        raise RuntimeError(f"No se pudieron descargar los pesos de {base_dir}: {e}") from e
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    logger.info("Pesos descargados: %s", pt)


def _source_path(engine: str) -> str:
    """Archivo junto al motor que dice a partir de qué pesos, en qué GPU y con qué TensorRT se generó."""
    return engine + ".source"


def _build_environment() -> dict:
    """GPU y versión de TensorRT de este equipo: un motor solo es válido con las mismas con las que se generó."""
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        gpu = None
    try:
        import tensorrt
        trt = tensorrt.__version__
    except Exception:
        trt = None
    return {"gpu": gpu, "tensorrt": trt}


def _engine_signature(weights: str) -> dict:
    """Lo que se anota al generar un motor y se compara al cargarlo: pesos, GPU y versión de TensorRT."""
    return {"weights_sha256": _weights_digest(weights), **_build_environment()}


def build_engine(base_dir: str) -> bool:
    """Genera best.engine (TensorRT, FP16, formas dinámicas) a partir de best.pt. Devuelve True si lo logra.

    Se exporta en una carpeta temporal para no pisar el best.onnx que pueda haber junto a los pesos.
    Las formas dinámicas son necesarias: un motor estático 640x640 cambia el relleno de la imagen y
    pierde detecciones respecto a best.pt.
    """
    pt, engine = pt_path(base_dir), engine_path(base_dir)
    if not os.path.exists(pt):
        logger.warning("No se puede generar el motor TensorRT: falta %s", pt)
        return False
    if not tensorrt_available():
        logger.warning("No se puede generar el motor TensorRT: TensorRT no está instalado "
                       "(pip install -r requirements-tensorrt.txt)")
        return False

    from ultralytics import YOLO
    logger.info("Generando el motor TensorRT para esta GPU (tarda aproximadamente un minuto)...")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_pt = os.path.join(tmp, "best.pt")
            shutil.copy2(pt, tmp_pt)
            built = YOLO(tmp_pt).export(format="engine", device=0, dynamic=True, batch=1, imgsz=IMGSZ, **FP16)
            staged = engine + ".tmp"
            shutil.copy2(built, staged)
            os.replace(staged, engine)
            # Se anota de qué pesos sale el motor (los copiados, no best.pt, por si cambió durante la exportación),
            # en qué GPU y con qué TensorRT
            with open(_source_path(engine), "w", encoding="utf-8") as f:
                json.dump(_engine_signature(tmp_pt), f)
    except Exception as e:
        logger.warning("No se pudo generar el motor TensorRT: %s", e)
        return False
    logger.info("Motor TensorRT generado: %s", engine)
    return True


def engine_is_stale(base_dir: str) -> bool:
    """True si best.engine no se generó a partir del best.pt actual, en esta GPU y con este TensorRT.

    Se comparan el SHA-256 de best.pt, el nombre de la GPU y la versión de TensorRT con los que build_engine
    anotó al generar el motor; las fechas de los archivos no sirven, porque copiar unos pesos antiguos conserva
    su fecha. Sin anotación, o con la de una versión anterior (solo el SHA-256), el motor también se considera
    antiguo. Sin esta comprobación, al sustituir best.pt el servidor seguiría usando el motor del modelo
    anterior, y un motor copiado de un equipo con otra GPU se cargaría aunque TensorRT avisa de que puede ir
    más lento, dar errores o bloquearse.
    """
    pt, engine = pt_path(base_dir), engine_path(base_dir)
    if not (os.path.exists(pt) and os.path.exists(engine)):
        return False
    try:
        with open(_source_path(engine), encoding="utf-8") as f:
            built_with = json.load(f)
    except (OSError, ValueError):
        return True
    return built_with != _engine_signature(pt)


def ensure_engine(base_dir: str, build: Builder) -> bool:
    """True si hay un best.engine actual (véase engine_is_stale); si falta o es antiguo, lo genera con build.

    Un motor sin best.pt se da por actual: no hay pesos con los que compararlo.
    """
    if not os.path.exists(engine_path(base_dir)):
        return build(base_dir)
    if not engine_is_stale(base_dir):
        return True
    logger.warning("best.engine no se generó a partir del best.pt actual")
    if build(base_dir):
        return True
    logger.warning("No se regeneró el motor; se usa best.pt para no ejecutar el modelo anterior")
    return False


def _load(path: str) -> Model:
    """Carga los pesos (best.pt en FP16) y hace una inferencia de prueba, que detecta motores incompatibles y
    sirve de warm-up."""
    from ultralytics import YOLO
    model = Model(YOLO(path, task="detect"), path, half=path.endswith(".pt"))
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), conf=0.325)
    return model


def _try_load_engine(base_dir: str) -> Optional[Model]:
    try:
        return _load(engine_path(base_dir))
    except Exception as e:
        logger.warning("El motor TensorRT no se pudo cargar (¿otra GPU o versión de TensorRT?): %s", e)
        return None


def load_model(base_dir: str, build: Builder = build_engine) -> Model:
    """El modelo de un modo en el mejor formato disponible para la GPU: best.engine o, si no, best.pt en FP16.

    build genera el motor cuando falta, es antiguo o no carga (en este caso, una sola vez). Por defecto se genera
    en este proceso; con el servidor en marcha, video.py lo genera en otro para no congelar el vídeo.
    """
    require_cuda()
    if ensure_engine(base_dir, build):
        model = _try_load_engine(base_dir)
        if model is None and build(base_dir):
            model = _try_load_engine(base_dir)
        if model is not None:
            return model

    pt = pt_path(base_dir)
    if not os.path.exists(pt):
        raise FileNotFoundError(
            f"No hay pesos utilizables en {base_dir}: se necesita best.pt (o best.engine) y una GPU con CUDA.")
    logger.info("Usando best.pt en la GPU (FP16).")
    return _load(pt)
