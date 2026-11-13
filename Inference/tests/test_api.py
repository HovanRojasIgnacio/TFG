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

    import execute  # Primero: fija variables de entorno que deben preceder a torch y OpenCV
    import cameras
    import evidence
    import profiles
    import video
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
            patcher = mock.patch.object(evidence, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.model = FakeModel()
        self.pipeline = self.new_pipeline()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.now = 1_000_000.0

    def new_pipeline(self):
        detector = video.Detector(profiles.load("casco_chaleco"), self.model, Runtime("best.pt", "cuda", False))
        return video.VideoPipeline(detector, start=False)

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
        for _ in range(video.VIOLATION_MIN_FRAMES + 2):
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
        for _ in range(video.VIOLATION_WINDOW):
            self.process(NO_HELMET)
        for _ in range(video.VIOLATION_WINDOW):
            self.process(HELMET)
        self.assertFalse(self.pipeline.violation_confirmed)
        self.assertEqual(self.pipeline.violations, 0)


class InfractionLogTest(PipelineTestCase):
    def record_one(self):
        for _ in range(video.VIOLATION_MIN_FRAMES):
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

    def test_a_log_that_is_valid_json_but_not_a_list_is_ignored(self):
        # JSON válido pero que no es una lista de infracciones: el servidor debe arrancar igualmente
        for content in ("null", "5", "{}", '"x"', "[1, null, {}]", "[" * 5000):
            with self.subTest(content=content[:20]):
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write(content)
                self.assertEqual(len(self.new_pipeline().infractions_log), 0)


class ApiTest(PipelineTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(execute, "pipeline", self.pipeline)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(execute.app)

    def test_stats(self):
        for _ in range(video.VIOLATION_MIN_FRAMES):
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


CASCO_NAMES = {0: "face", 1: "head", 2: "head_with_helmet", 3: "helmet", 4: "person", 5: "person_no_helmet",
               6: "person_with_helmet"}


class ProfileChangeTest(PipelineTestCase):
    """Cambio de modo con load_detector simulado (el real carga el modelo en la GPU)."""

    def casco_detector(self, *_args, **_kwargs):
        model = FakeModel()
        model.names = CASCO_NAMES
        return video.Detector(profiles.load("casco"), model, Runtime("best.pt", "cuda", False))

    def load(self, profile_id, loader):
        with mock.patch.object(video, "load_detector", loader), mock.patch.object(video.time, "sleep"):
            self.pipeline.profile_loading = profile_id  # Lo que deja select_profile antes de lanzar el hilo
            self.pipeline._load_profile(profile_id)

    def test_new_mode_replaces_the_detector_and_clears_the_counters(self):
        for _ in range(video.VIOLATION_MIN_FRAMES):
            self.process(NO_HELMET)
        self.load("casco", self.casco_detector)
        self.assertEqual(self.pipeline.detector.profile.id, "casco")
        self.assertEqual(self.pipeline.items, {"helmet": {"ok": 0, "missing": 0}})
        self.assertEqual((self.pipeline.violations, self.pipeline.violation_confirmed), (0, False))
        self.assertIsNone(self.pipeline.profile_loading)
        self.assertEqual(self.pipeline.conf_threshold, 0.325)  # El recomendado del modo casco

    def test_the_operators_threshold_is_kept(self):
        self.pipeline.conf_threshold, self.pipeline.conf_set_by_user = 0.6, True
        self.load("casco", self.casco_detector)
        self.assertEqual(self.pipeline.conf_threshold, 0.6)

    def test_a_failed_load_keeps_the_current_mode(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("pesos dañados")
        with self.assertLogs("guardwatch", level="ERROR"):
            self.load("casco", broken)
        self.assertEqual(self.pipeline.detector.profile.id, "casco_chaleco")
        self.assertIn("pesos dañados", self.pipeline.profile_error)
        self.assertIsNone(self.pipeline.profile_loading)

    def test_select_profile_refuses_unknown_and_concurrent_changes(self):
        with self.assertRaises(KeyError):
            self.pipeline.select_profile("../casco")
        self.assertFalse(self.pipeline.select_profile("casco_chaleco"))  # Ya es el actual
        self.pipeline.profile_loading = "casco"
        with self.assertRaises(video.ProfileBusy):
            self.pipeline.select_profile("casco")


class FakeCapture:
    """Cámara simulada: entrega `frames` fotogramas y después falla, como una cámara desconectada."""

    def __init__(self, frames=1000, opened=True):
        self.frames, self.opened, self.released = frames, opened, False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if self.frames <= 0:
            return False, None
        self.frames -= 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class CameraTest(PipelineTestCase):
    LOCAL_1 = dict(type="local", index=1, name="Otra")

    def test_selecting_the_camera_in_use_connects_nothing(self):
        with mock.patch.object(cameras, "open_capture") as open_capture:
            self.assertTrue(self.pipeline.set_camera(cameras.CameraSource(type="local", index=0, name="x")))
        open_capture.assert_not_called()

    def test_a_working_camera_is_handed_to_the_video_loop_already_open(self):
        cap = FakeCapture()
        with mock.patch.object(cameras, "open_capture", return_value=cap):
            self.assertTrue(self.pipeline.set_camera(cameras.CameraSource(**self.LOCAL_1)))
        self.assertEqual(self.pipeline.camera_source.index, 1)
        self.assertIs(self.pipeline._handover[1], cap)
        self.assertFalse(cap.released)

    def test_a_camera_without_frames_is_refused_and_released(self):
        for cap in (FakeCapture(frames=0), FakeCapture(opened=False)):
            with self.subTest(opened=cap.opened), mock.patch.object(cameras, "open_capture", return_value=cap):
                self.assertFalse(self.pipeline.set_camera(cameras.CameraSource(**self.LOCAL_1)))
                self.assertEqual(self.pipeline.camera_source.index, 0)  # Sigue la cámara anterior
                self.assertTrue(cap.released)

    def test_a_camera_that_does_not_answer_in_time_is_refused(self):
        import threading
        release = threading.Event()
        cap = FakeCapture()

        def slow_open(_source):
            release.wait(5)
            return cap
        with mock.patch.object(cameras, "open_capture", slow_open), self.assertLogs("guardwatch", level="WARNING"):
            self.assertIsNone(cameras.connect_camera(cameras.CameraSource(**self.LOCAL_1), timeout=0.05))
        release.set()
        for _ in range(100):  # La prueba abandonada libera la cámara cuando por fin se abre
            if cap.released:
                break
            video.time.sleep(0.01)
        self.assertTrue(cap.released)

    def test_ip_reader_keeps_only_the_newest_frame_and_releases_the_camera(self):
        cap = FakeCapture(frames=50)
        reader = cameras.LatestFrameReader(cap)
        ok, frame = reader.read(timeout=1.0)
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (48, 64, 3))
        for _ in range(100):  # Espera a que el hilo lector agote los 50 fotogramas
            if cap.frames == 0:
                break
            video.time.sleep(0.01)
        reader.read(timeout=0.2)                       # Entrega el último pendiente, si lo hay
        self.assertEqual(reader.read(timeout=0.05), (False, None))  # No repite un fotograma ya entregado
        reader.release()
        for _ in range(100):
            if cap.released:
                break
            video.time.sleep(0.01)
        self.assertTrue(cap.released)

    def test_video_loop_reports_a_lost_camera_and_reconnects(self):
        caps = [FakeCapture(frames=3), FakeCapture(frames=10_000)]
        opened = []

        def open_capture(_source):
            opened.append(caps[min(len(opened), 1)])
            return opened[-1]
        fast = dict(FRAME_FAIL_TIMEOUT=0.1, RECONNECT_MIN_DELAY=0.1)
        patches = [mock.patch.object(video, name, value) for name, value in fast.items()]
        patches.append(mock.patch.object(cameras, "open_capture", open_capture))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.model.detections = HELMET
        self.pipeline.thread.start()
        self.addCleanup(self.pipeline.stop)
        with self.assertLogs("guardwatch", level="WARNING"):
            for _ in range(300):  # Hasta 6 s: la primera cámara falla, el bucle la suelta y abre la segunda
                if len(opened) >= 2 and self.pipeline.items["helmet"]["ok"] == 1:
                    break
                video.time.sleep(0.02)
        self.assertGreaterEqual(len(opened), 2)
        self.assertTrue(caps[0].released)
        self.assertIsNone(self.pipeline.camera_error)
        self.assertEqual(self.pipeline.items["helmet"]["ok"], 1)
        self.pipeline.stop()
        self.assertTrue(caps[1].released)


    def test_a_camera_change_clears_the_previous_cameras_violations(self):
        for _ in range(video.VIOLATION_MIN_FRAMES):
            self.process(NO_HELMET)
        self.assertTrue(self.pipeline.violation_confirmed)
        # Cambio de cámara con el bucle real: la ventana de infracciones de la anterior no debe sobrevivir
        # La cámara nueva da un solo fotograma tras la prueba de conexión (y luego falla durante
        # FRAME_FAIL_TIMEOUT), para que la ventana que se comprueba sea la del primer fotograma
        with mock.patch.object(cameras, "open_capture", return_value=FakeCapture(frames=2)):
            self.assertTrue(self.pipeline.set_camera(cameras.CameraSource(**self.LOCAL_1)))
        self.model.detections = HELMET
        self.pipeline.thread.start()
        self.addCleanup(self.pipeline.stop)
        for _ in range(150):  # Hasta 3 s a que el bucle procese un fotograma de la cámara nueva
            if self.pipeline.items["helmet"]["ok"] == 1:
                break
            video.time.sleep(0.02)
        self.assertEqual(self.pipeline.items["helmet"]["ok"], 1)
        self.assertFalse(self.pipeline.violation_confirmed)
        self.assertEqual(sum(self.pipeline._recent_violations), 0)


class MaskUrlTest(unittest.TestCase):
    def test_credentials_never_leave_the_server(self):
        cases = {
            "http://admin:s3cr3t@127.0.0.1:8090/video?pwd=hunter2": "http://***@127.0.0.1:8090/video?pwd=***",
            "rtsp://host:554/user=admin&password=x&channel=1": "rtsp://host:554/user=***&password=***&channel=1",
            "http://192.168.1.100:8080/video": "http://192.168.1.100:8080/video",
        }
        for url, masked in cases.items():
            with self.subTest(url=url):
                self.assertEqual(cameras.mask_url(url), masked)


class ImportTest(unittest.TestCase):
    def test_importing_the_module_starts_nothing(self):
        self.assertIsNone(execute.pipeline)


if __name__ == "__main__":
    unittest.main()
