"""Selección de dispositivo y de pesos del modelo, compartida por execute.py y benchmark.py.

Prioridad: usar la GPU siempre que sea posible.
  - Con CUDA:  best.engine (TensorRT) > best.pt (PyTorch en FP16) > best.onnx (solo si ONNX Runtime tiene CUDA)
  - Sin CUDA:  best.onnx > best.pt   (best.engine necesita GPU)
"""
import logging
import os
from typing import NamedTuple

import torch

logger = logging.getLogger("guardwatch")


class Runtime(NamedTuple):
    weights_path: str
    device: str      # dispositivo que se pasa a model.predict(): "cuda" o "cpu"
    half: bool       # FP16 (solo con pesos .pt en GPU)


def precision_kwargs(half: bool) -> dict:
    """Argumentos de precisión para model.predict(), válidos en versiones nuevas y antiguas de Ultralytics.

    Las versiones recientes sustituyen `half` por `quantize` y avisan en cada llamada si se usa `half`.
    Con FP32 o con un motor TensorRT (la precisión ya va dentro del motor) no hace falta pasar nada.
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


def _onnx_has_cuda() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def select_runtime(base_dir: str) -> Runtime:
    engine, onnx, pt = (os.path.join(base_dir, f"best.{ext}") for ext in ("engine", "onnx", "pt"))
    cuda = torch.cuda.is_available()

    if cuda:
        candidates = [engine, pt] + ([onnx] if _onnx_has_cuda() else [])
    else:
        candidates = [onnx, pt]
        logger.warning(
            "CUDA no disponible: la inferencia se ejecutará en la CPU. "
            "Instala PyTorch con CUDA (véase el README) para usar la GPU."
        )

    for path in candidates:
        if os.path.exists(path):
            return Runtime(path, "cuda" if cuda else "cpu", cuda and path.endswith(".pt"))

    # Último recurso: un .onnx sin soporte CUDA en ONNX Runtime solo puede ir en la CPU
    if os.path.exists(onnx):
        logger.warning("best.onnx se ejecutará en la CPU (ONNX Runtime sin CUDA). Usa best.pt o best.engine para la GPU.")
        return Runtime(onnx, "cpu", False)

    raise FileNotFoundError(f"No se encontró best.engine, best.onnx ni best.pt en {base_dir}")
