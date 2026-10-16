"""Carga del modelo en la GPU, compartida por execute.py y benchmark.py. Solo GPU: no hay modo CPU.

Orden de preferencia (siempre en la GPU NVIDIA):
  1. best.engine (TensorRT). Si no existe, o no se puede cargar (otra GPU, otra versión de
     TensorRT...), se genera automáticamente a partir de best.pt (tarda aproximadamente un minuto).
  2. best.pt (PyTorch en FP16), si TensorRT no está instalado o la generación falla.
  3. best.onnx, solo si ONNX Runtime tiene el proveedor CUDA.
"""
import hashlib
import importlib.util
import logging
import os
import shutil
import tempfile
from typing import NamedTuple

import numpy as np
import torch

logger = logging.getLogger("guardwatch")

IMGSZ = 640


class Runtime(NamedTuple):
    weights_path: str
    device: str      # siempre "cuda"
    half: bool       # FP16 (solo con pesos .pt; un motor TensorRT ya lleva la precisión dentro)


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GuardWatch AI necesita una GPU NVIDIA con CUDA, pero este Python (torch {torch.__version__}) "
            "no la ve. Instala PyTorch con CUDA (véase el README) y arranca el servidor con el entorno "
            "virtual de Inference/.venv."
        )


def precision_kwargs(half: bool) -> dict:
    """Argumentos de precisión para predict()/export(), válidos en versiones nuevas y antiguas de Ultralytics.

    Las versiones recientes sustituyen `half` por `quantize` y avisan en cada llamada si se usa `half`.
    Con FP32 o con un motor TensorRT no hace falta pasar nada.
    """
    if not half:
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:
        pass
    return {"half": True}


def _tensorrt_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


def _onnx_has_cuda() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _weights_digest(path: str) -> str:
    """SHA-256 de un archivo de pesos (unos 50 ms para los 20 MB de best.pt)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(engine: str) -> str:
    """Archivo junto al motor con el SHA-256 de los pesos a partir de los que se generó."""
    return engine + ".source"


def build_engine(base_dir: str) -> bool:
    """Genera best.engine (TensorRT, FP16, formas dinámicas) a partir de best.pt. Devuelve True si lo logra.

    Se exporta en una carpeta temporal para no pisar el best.onnx que pueda haber junto a los pesos.
    Las formas dinámicas son necesarias: un motor estático 640x640 cambia el relleno de la imagen y
    pierde detecciones respecto a best.pt.
    """
    pt = os.path.join(base_dir, "best.pt")
    engine = os.path.join(base_dir, "best.engine")
    if not os.path.exists(pt):
        logger.warning("No se puede generar el motor TensorRT: falta %s", pt)
        return False
    if not _tensorrt_available():
        logger.warning("No se puede generar el motor TensorRT: TensorRT no está instalado "
                       "(pip install -r requirements-tensorrt.txt)")
        return False

    from ultralytics import YOLO
    logger.info("Generando el motor TensorRT para esta GPU (tarda aproximadamente un minuto)...")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_pt = os.path.join(tmp, "best.pt")
            shutil.copy2(pt, tmp_pt)
            built = YOLO(tmp_pt).export(format="engine", device=0, dynamic=True, batch=1,
                                        imgsz=IMGSZ, **precision_kwargs(True))
            staged = engine + ".tmp"
            shutil.copy2(built, staged)
            os.replace(staged, engine)
            # Se anota de qué pesos sale el motor (los copiados, no best.pt, por si cambió durante la exportación)
            with open(_source_path(engine), "w", encoding="ascii") as f:
                f.write(_weights_digest(tmp_pt))
    except Exception as e:
        logger.warning("No se pudo generar el motor TensorRT: %s", e)
        return False
    logger.info("Motor TensorRT generado: %s", engine)
    return True


def engine_is_stale(base_dir: str) -> bool:
    """True si best.engine no se generó a partir del best.pt actual.

    Se compara el SHA-256 de best.pt con el que build_engine anotó al generar el motor; las fechas de los
    archivos no sirven, porque copiar unos pesos antiguos conserva su fecha. Sin anotación (motor generado
    antes de existir esta comprobación o copiado de otro equipo) el motor también se considera antiguo.
    Sin esta comprobación, al sustituir best.pt el servidor seguiría usando el motor del modelo anterior e
    informaría de ENGINE como si fuera el actual.
    """
    pt, engine = os.path.join(base_dir, "best.pt"), os.path.join(base_dir, "best.engine")
    if not (os.path.exists(pt) and os.path.exists(engine)):
        return False
    try:
        with open(_source_path(engine), encoding="ascii") as f:
            built_from = f.read().strip()
    except OSError:
        return True
    return built_from != _weights_digest(pt)


def _load(path: str):
    """Carga los pesos y hace una inferencia de prueba para detectar motores incompatibles."""
    from ultralytics import YOLO
    model = YOLO(path, task="detect")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(blank, device="cuda", conf=0.325, verbose=False,
                  **precision_kwargs(path.endswith(".pt")))
    return model


def load_model(base_dir: str):
    """Devuelve (modelo, Runtime) con el mejor formato disponible para la GPU."""
    require_cuda()
    engine, pt, onnx = (os.path.join(base_dir, f"best.{ext}") for ext in ("engine", "pt", "onnx"))

    use_engine = True
    if not os.path.exists(engine):
        build_engine(base_dir)
    elif engine_is_stale(base_dir):
        logger.warning("best.engine no se generó a partir del best.pt actual: se regenera el motor")
        if not build_engine(base_dir):
            logger.warning("No se pudo regenerar el motor; se usa best.pt para no ejecutar el modelo anterior")
            use_engine = False

    if use_engine and os.path.exists(engine):
        try:
            return _load(engine), Runtime(engine, "cuda", False)
        except Exception as e:
            logger.warning("El motor TensorRT no se pudo cargar (¿otra GPU o versión de TensorRT?): %s", e)
            if build_engine(base_dir):
                try:
                    return _load(engine), Runtime(engine, "cuda", False)
                except Exception as e2:
                    logger.warning("El motor regenerado tampoco se pudo cargar: %s", e2)

    if os.path.exists(pt):
        logger.info("Usando best.pt en la GPU (FP16).")
        return _load(pt), Runtime(pt, "cuda", True)

    if os.path.exists(onnx) and _onnx_has_cuda():
        logger.info("Usando best.onnx en la GPU (ONNX Runtime CUDA).")
        return _load(onnx), Runtime(onnx, "cuda", False)

    raise FileNotFoundError(
        f"No hay pesos utilizables en {base_dir}: se necesita best.pt (o best.engine) y una GPU con CUDA."
    )
