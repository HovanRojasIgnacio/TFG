"""Pruebas de los pesos y del motor TensorRT (runtime.py), sin GPU, PyTorch ni TensorRT.

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


class ModeFolderTestCase(unittest.TestCase):
    """La carpeta de un modo con su best.pt, en un equipo con la GPU y el TensorRT de THIS_PC."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.base = self.dir.name
        self.pt = os.path.join(self.base, "best.pt")
        self.engine = os.path.join(self.base, "best.engine")
        with open(self.pt, "wb") as f:
            f.write(b"pesos")
        self.patch(runtime, "_build_environment", return_value=dict(THIS_PC))

    def patch(self, target, name, **kwargs):
        patcher = mock.patch.object(target, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def write_source(self, content):
        with open(self.engine + ".source", "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))

    def signature(self, **changes):
        return {"weights_sha256": runtime._weights_digest(self.pt), **THIS_PC, **changes}

    def write_engine(self, **changes):
        """Lo que deja build_engine: best.engine y su anotación (changes: de otros pesos, otra GPU u otro TensorRT)."""
        with open(self.engine, "wb") as f:
            f.write(b"motor")
        self.write_source(self.signature(**changes))


class EngineIsStaleTest(ModeFolderTestCase):
    def setUp(self):
        super().setUp()
        with open(self.engine, "wb") as f:
            f.write(b"motor")

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


class LoadModelTest(ModeFolderTestCase):
    """Qué pesos elige load_model según el estado de best.engine, con la carga y la generación simuladas."""

    def setUp(self):
        super().setUp()
        self.patch(runtime, "require_cuda")
        self.patch(runtime, "_load", side_effect=self.fake_load)
        self.logger = self.patch(runtime, "logger")
        self.can_build = True      # False: como sin TensorRT, el motor no se puede generar
        self.builds = 0
        self.engine_failures = 0   # Veces que best.engine no carga (otra GPU u otro TensorRT)

    def fake_load(self, path):
        if path == self.engine and self.engine_failures:
            self.engine_failures -= 1
            raise RuntimeError("motor incompatible")
        return runtime.Model(None, path, half=path.endswith(".pt"))

    def build(self, base_dir):
        self.assertEqual(base_dir, self.base)
        self.builds += 1
        if self.can_build:
            self.write_engine()
        return self.can_build

    def load(self):
        return runtime.load_model(self.base, self.build)

    def assertLoaded(self, path, builds):
        model = self.load()
        self.assertEqual((model.weights_path, model.half, self.builds), (path, path == self.pt, builds))

    def test_a_current_engine_is_loaded_without_building(self):
        self.write_engine()
        self.assertLoaded(self.engine, builds=0)

    def test_a_missing_engine_is_built_and_loaded(self):
        self.assertLoaded(self.engine, builds=1)

    def test_without_an_engine_best_pt_runs_in_fp16(self):
        self.can_build = False
        self.assertLoaded(self.pt, builds=1)

    def test_a_stale_engine_is_rebuilt(self):
        self.write_engine(gpu="NVIDIA GeForce RTX 5060")
        self.assertLoaded(self.engine, builds=1)

    def test_a_stale_engine_that_cannot_be_rebuilt_is_not_used(self):
        # Si no, el servidor ejecutaría el modelo anterior a best.pt
        self.write_engine(weights_sha256="0" * 64)
        self.can_build = False
        self.assertLoaded(self.pt, builds=1)
        warnings = " ".join(str(c.args) for c in self.logger.warning.call_args_list)
        self.assertIn("se usa best.pt para no ejecutar el modelo anterior", warnings)

    def test_an_engine_that_does_not_load_is_rebuilt_once(self):
        self.write_engine()
        self.engine_failures = 1
        self.assertLoaded(self.engine, builds=1)

    def test_an_engine_that_still_does_not_load_gives_best_pt(self):
        self.write_engine()
        self.engine_failures = 2
        self.assertLoaded(self.pt, builds=1)

    def test_an_engine_without_best_pt_is_used(self):
        self.write_engine()
        os.remove(self.pt)
        self.assertLoaded(self.engine, builds=0)

    def test_no_weights_is_an_error(self):
        os.remove(self.pt)
        self.can_build = False
        with self.assertRaises(FileNotFoundError):
            self.load()


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
