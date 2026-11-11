"""Pruebas de la API y del pipeline de vídeo con un modelo simulado, sin GPU ni cámara.

Necesitan los paquetes de requirements.txt (FastAPI, OpenCV, PyTorch...) y httpx; sin ellos se omiten, de modo
que `python -m unittest discover -s Inference/tests` sigue funcionando solo con la biblioteca estándar.
"""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TESTS_DIR))

try:
    import numpy as np
    from fastapi.testclient import TestClient

    import execute
    import profiles
    from runtime import Runtime
except ImportError as e:  # Entorno sin las dependencias del servidor (por ejemplo, la integración continua)
    raise unittest.SkipTest(f"faltan las dependencias del servidor: {e}")

NAMES = {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"}
NO_HELMET = [(1, [100, 100, 200, 200], 0.9)]
HELMET = [(0, [100, 100, 200, 200], 0.9)]


class _Values(list):
    """Lo que el pipeline usa de un tensor de Ultralytics: .int() y .tolist()."""

    def int(self):
        return self

    def tolist(self):
        return list(self)


class FakeModel:
    """Modelo que devuelve las detecciones que se le indiquen: (clase, recuadro, confianza)."""
    names = NAMES

    def __init__(self):
        self.detections = []

    def predict(self, frame, **kwargs):
        boxes = SimpleNamespace(xyxy=_Values(d[1] for d in self.detections),
                                conf=_Values(d[2] for d in self.detections),
                                cls=_Values(d[0] for d in self.detections))
        return [SimpleNamespace(boxes=boxes)]


class PipelineTestCase(unittest.TestCase):
    """Crea un pipeline sin hilo de vídeo que guarda las capturas y el registro en una carpeta temporal."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_file = os.path.join(self.tmp.name, "infractions.json")
        for name, value in (("infractions_dir", self.tmp.name), ("LOG_FILE", self.log_file)):
            patcher = mock.patch.object(execute, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.model = FakeModel()
        self.pipeline = self.new_pipeline()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.now = 1_000_000.0

    def new_pipeline(self):
        detector = execute.Detector(profiles.load("casco_chaleco"), self.model, Runtime("best.pt", "cuda", False))
        return execute.VideoPipeline(detector, start=False)

    def process(self, detections, seconds=0.1):
        self.model.detections = detections
        self.now += seconds
        self.pipeline._process_frame(self.frame.copy(), self.pipeline.camera_source, self.now, 0.0)

    def snapshots(self):
        return [f for f in os.listdir(self.tmp.name) if f.endswith(".jpg")]


class ViolationConfirmationTest(PipelineTestCase):
    def test_one_frame_is_counted_but_not_recorded(self):
        self.process(NO_HELMET)
        self.assertEqual(self.pipeline.violations, 1)
        self.assertFalse(self.pipeline.violation_confirmed)
        self.assertEqual(self.snapshots(), [])
        self.assertEqual(len(self.pipeline.infractions_log), 0)

    def test_several_frames_confirm_and_record_once(self):
        for _ in range(execute.VIOLATION_MIN_FRAMES + 2):
            self.process(NO_HELMET)
        self.assertTrue(self.pipeline.violation_confirmed)
        self.assertEqual(len(self.snapshots()), 1)  # Las siguientes esperan los 4 s entre capturas
        entry = self.pipeline.infractions_log[0]
        self.assertEqual(entry["details"], "Falta: casco ×1")
        self.assertRegex(entry["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(entry["image_path"], r"^/static/infractions/inf_\d+_[0-9a-f]{16}\.jpg$")

    def test_a_missed_frame_does_not_cancel_the_violation(self):
        for detections in (NO_HELMET, NO_HELMET, HELMET, NO_HELMET, NO_HELMET, NO_HELMET):
            self.process(detections)
        self.assertTrue(self.pipeline.violation_confirmed)

    def test_confirmation_ends_when_the_violation_disappears(self):
        for _ in range(execute.VIOLATION_WINDOW):
            self.process(NO_HELMET)
        for _ in range(execute.VIOLATION_WINDOW):
            self.process(HELMET)
        self.assertFalse(self.pipeline.violation_confirmed)
        self.assertEqual(self.pipeline.violations, 0)


class InfractionLogTest(PipelineTestCase):
    def record_one(self):
        for _ in range(execute.VIOLATION_MIN_FRAMES):
            self.process(NO_HELMET)
        self.assertEqual(len(self.pipeline.infractions_log), 1)

    def test_the_log_survives_a_restart(self):
        self.record_one()
        restarted = self.new_pipeline()
        self.assertEqual(list(restarted.infractions_log), list(self.pipeline.infractions_log))

    def test_entries_without_their_snapshot_are_dropped(self):
        self.record_one()
        os.remove(os.path.join(self.tmp.name, self.snapshots()[0]))
        self.assertEqual(len(self.new_pipeline().infractions_log), 0)

    def test_a_corrupt_log_is_ignored(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("{no es json")
        self.assertEqual(len(self.new_pipeline().infractions_log), 0)


class ApiTest(PipelineTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(execute, "pipeline", self.pipeline)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(execute.app)

    def test_stats(self):
        for _ in range(execute.VIOLATION_MIN_FRAMES):
            self.process(NO_HELMET)
        data = self.client.get("/api/stats").json()
        self.assertEqual(data["items"], {"helmet": {"ok": 0, "missing": 1}, "vest": {"ok": 0, "missing": 0}})
        self.assertEqual((data["violations"], data["violation_confirmed"], data["safety_score"]), (1, True, 0))
        self.assertEqual(data["profile"]["id"], "casco_chaleco")
        self.assertEqual(len(data["infractions"]), 1)
        self.assertTrue(data["infractions"][0]["image_url"].startswith("http://testserver/static/infractions/"))

    def test_settings(self):
        self.assertEqual(self.client.post("/api/settings", json={"conf": 0.5}).json()["conf_threshold"], 0.5)
        self.assertTrue(self.pipeline.conf_set_by_user)
        for conf in (0.05, 0.99):
            self.assertEqual(self.client.post("/api/settings", json={"conf": conf}).status_code, 400)
        self.assertEqual(self.client.post("/api/settings", json={"conf": "alto"}).status_code, 422)
        self.assertEqual(self.pipeline.conf_threshold, 0.5)

    def test_invalid_cameras_are_refused_without_opening_anything(self):
        for body in ({"type": "ip", "url": "file:///C:/Windows/win.ini"}, {"type": "ip"}, {"type": "usb"},
                     {"type": "local"}, {"type": "local", "index": -1}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/api/cameras/select", json=body).status_code, 400)

    def test_profiles(self):
        data = self.client.get("/api/profiles").json()
        self.assertEqual(data["current"], "casco_chaleco")
        self.assertIn("casco", [p["id"] for p in data["profiles"]])
        self.assertEqual(self.client.post("/api/profiles/select", json={"id": "../casco"}).status_code, 400)
        self.assertEqual(self.client.post("/api/profiles/select", json={"id": "casco_chaleco"}).json()["status"],
                         "current")


class MaskUrlTest(unittest.TestCase):
    def test_credentials_never_leave_the_server(self):
        cases = {
            "http://admin:s3cr3t@127.0.0.1:8090/video?pwd=hunter2": "http://***@127.0.0.1:8090/video?pwd=***",
            "rtsp://host:554/user=admin&password=x&channel=1": "rtsp://host:554/user=***&password=***&channel=1",
            "http://192.168.1.100:8080/video": "http://192.168.1.100:8080/video",
        }
        for url, masked in cases.items():
            with self.subTest(url=url):
                self.assertEqual(execute.mask_url(url), masked)


class ImportTest(unittest.TestCase):
    def test_importing_the_module_starts_nothing(self):
        self.assertIsNone(execute.pipeline)


if __name__ == "__main__":
    unittest.main()
