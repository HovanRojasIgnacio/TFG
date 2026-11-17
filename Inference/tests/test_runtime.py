"""Pruebas de la comprobación del motor TensorRT (runtime.engine_is_stale), sin GPU ni TensorRT.

Necesitan NumPy (lo importa runtime.py); sin él se omiten.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import runtime
except ImportError as e:
    raise unittest.SkipTest(f"faltan las dependencias de runtime.py: {e}")

THIS_PC = {"gpu": "NVIDIA GeForce RTX 5070 Ti", "tensorrt": "11.4.0.106"}


class EngineIsStaleTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.base = self.dir.name
        self.pt = os.path.join(self.base, "best.pt")
        self.engine = os.path.join(self.base, "best.engine")
        with open(self.pt, "wb") as f:
            f.write(b"pesos")
        with open(self.engine, "wb") as f:
            f.write(b"motor")
        patcher = mock.patch.object(runtime, "_build_environment", return_value=dict(THIS_PC))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_source(self, content):
        with open(self.engine + ".source", "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))

    def signature(self, **changes):
        return {"weights_sha256": runtime._weights_digest(self.pt), **THIS_PC, **changes}

    def test_same_weights_gpu_and_tensorrt_is_current(self):
        self.write_source(self.signature())
        self.assertFalse(runtime.engine_is_stale(self.base))

    def test_engine_from_another_gpu_is_stale(self):
        # El caso real: el proyecto se copió de un equipo con una RTX 5060 junto con su best.engine.source
        self.write_source(self.signature(gpu="NVIDIA GeForce RTX 5060"))
        self.assertTrue(runtime.engine_is_stale(self.base))

    def test_engine_from_another_tensorrt_is_stale(self):
        self.write_source(self.signature(tensorrt="10.13.0"))
        self.assertTrue(runtime.engine_is_stale(self.base))

    def test_engine_from_other_weights_is_stale(self):
        self.write_source(self.signature(weights_sha256="0" * 64))
        self.assertTrue(runtime.engine_is_stale(self.base))

    def test_old_record_with_only_the_hash_is_stale(self):
        # Formato anterior: solo el SHA-256 de los pesos, sin GPU ni versión de TensorRT
        self.write_source(runtime._weights_digest(self.pt))
        self.assertTrue(runtime.engine_is_stale(self.base))

    def test_missing_record_is_stale(self):
        self.assertTrue(runtime.engine_is_stale(self.base))

    def test_no_engine_is_not_stale(self):
        os.remove(self.engine)
        self.assertFalse(runtime.engine_is_stale(self.base))


class EnsureWeightsTest(unittest.TestCase):
    """Descarga de best.pt cuando falta (con una dirección file:// en lugar de la release de GitHub)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.source = os.path.join(self.dir.name, "publicado.pt")
        with open(self.source, "wb") as f:
            f.write(b"pesos publicados")
        self.url = "file:///" + self.source.replace(os.sep, "/").lstrip("/")
        self.sha = runtime._weights_digest(self.source)
        self.base = os.path.join(self.dir.name, "modo")
        os.mkdir(self.base)
        self.pt = os.path.join(self.base, "best.pt")

    def test_missing_weights_are_downloaded_and_checked(self):
        with self.assertLogs("guardwatch", level="INFO"):
            runtime.ensure_weights(self.base, self.url, self.sha)
        with open(self.pt, "rb") as f:
            self.assertEqual(f.read(), b"pesos publicados")

    def test_a_file_with_another_hash_is_not_kept(self):
        with self.assertLogs("guardwatch", level="INFO"), self.assertRaises(RuntimeError):
            runtime.ensure_weights(self.base, self.url, "0" * 64)
        self.assertEqual(os.listdir(self.base), [])  # Ni best.pt ni la descarga a medias

    def test_existing_weights_are_not_downloaded_again(self):
        with open(self.pt, "wb") as f:
            f.write(b"locales")
        runtime.ensure_weights(self.base, "https://no.se.usa/best.pt", self.sha)
        with open(self.pt, "rb") as f:
            self.assertEqual(f.read(), b"locales")


if __name__ == "__main__":
    unittest.main()
