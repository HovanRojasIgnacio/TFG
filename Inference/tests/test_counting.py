"""Pruebas de los perfiles y del recuento: python -m unittest discover Inference/tests"""
import json
import os
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TESTS_DIR))

import counting
import profiles

VEST_PROFILE = {
    "name": "Casco y chaleco",
    "items": {"helmet": "Casco", "vest": "Chaleco"},
    "classes": {
        "Helmet": {"item": "helmet", "state": "ok"},
        "NoHelmet": {"item": "helmet", "state": "missing"},
        "Vest": {"item": "vest", "state": "ok"},
        "NoVest": {"item": "vest", "state": "missing"},
    },
}


class CascoRegressionTest(unittest.TestCase):
    """El perfil casco cuenta exactamente igual que counting.py antes de los perfiles (fixture grabado)."""

    @classmethod
    def setUpClass(cls):
        cls.profile = profiles.load("casco")
        with open(os.path.join(TESTS_DIR, "fixtures", "casco_detections.json"), encoding="utf-8") as f:
            cls.fixture = json.load(f)

    def test_profile_describes_every_model_class(self):
        profiles.check_model(self.profile, dict(enumerate(self.fixture["model_names"])))

    def _check(self, frame, result, counts_key, flags_key):
        old = frame[counts_key]
        self.assertEqual(result.items, {"helmet": {"ok": old["helmet"], "missing": old["no_helmet"]}})
        self.assertEqual(result.neutral, old["face"])
        self.assertEqual(result.flags, frame[flags_key])

    def test_counts_with_boxes(self):
        for i, frame in enumerate(self.fixture["frames"]):
            with self.subTest(frame=i):
                result = counting.count(self.profile, frame["names"], frame["confs"], frame["boxes"])
                self._check(frame, result, "counts", "flags")
                self.assertEqual(result.highest_violation, frame["highest"])

    def test_counts_without_boxes(self):
        for i, frame in enumerate(self.fixture["frames"]):
            with self.subTest(frame=i):
                result = counting.count(self.profile, frame["names"], frame["confs"])
                self._check(frame, result, "counts_no_boxes", "flags_no_boxes")


class VestProfileTest(unittest.TestCase):
    def setUp(self):
        self.profile = profiles.parse("casco_chaleco", VEST_PROFILE)

    def test_worker_without_vest(self):
        result = counting.count(self.profile, ["Helmet", "NoVest"], [0.9, 0.7],
                                [[10, 10, 50, 50], [5, 50, 60, 150]])
        self.assertEqual(result.items, {"helmet": {"ok": 1, "missing": 0}, "vest": {"ok": 0, "missing": 1}})
        self.assertEqual(result.violations, 1)
        self.assertEqual(result.safety_score, 50)
        self.assertEqual(result.highest_violation, 0.7)
        self.assertEqual(result.flags, [True, True])

    def test_every_box_counts_without_hierarchy(self):
        names = ["Helmet", "Helmet", "Vest", "NoHelmet", "NoVest"]
        result = counting.count(self.profile, names, [0.5, 0.6, 0.7, 0.4, 0.8])
        self.assertEqual(result.items, {"helmet": {"ok": 2, "missing": 1}, "vest": {"ok": 1, "missing": 1}})
        self.assertEqual(result.safety_score, 60)
        self.assertEqual(result.highest_violation, 0.8)

    def test_empty_frame(self):
        result = counting.count(self.profile, [], [], [])
        self.assertEqual(result.items, {"helmet": {"ok": 0, "missing": 0}, "vest": {"ok": 0, "missing": 0}})
        self.assertEqual((result.violations, result.safety_score, result.highest_violation), (0, 100, 0.0))


class NeutralOnlyTest(unittest.TestCase):
    def test_neutral_and_ignored_classes(self):
        casco = profiles.load("casco")
        result = counting.count(casco, ["face", "person", "face"], [0.9, 0.9, 0.8])
        self.assertEqual(result.neutral, 2)
        self.assertEqual(result.flags, [True, False, True])
        self.assertEqual((result.violations, result.safety_score), (0, 100))


class ProfileValidationTest(unittest.TestCase):
    def test_model_with_undescribed_class_is_rejected(self):
        vest = profiles.parse("casco_chaleco", VEST_PROFILE)
        with self.assertRaises(ValueError):
            profiles.check_model(vest, {0: "Helmet", 1: "NoHelmet", 2: "face"})

    def test_invalid_profiles(self):
        bad_state = {**VEST_PROFILE, "classes": {"Helmet": {"item": "helmet", "state": "worn"}}}
        unknown_item = {**VEST_PROFILE, "classes": {"Gloves": {"item": "gloves", "state": "ok"}}}
        bad_level = {**VEST_PROFILE, "classes": {"Helmet": {"item": "helmet", "state": "ok", "level": "torso"}}}
        for data in (bad_state, unknown_item, bad_level, {**VEST_PROFILE, "items": {}}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                profiles.parse("x", data)

    def test_only_installed_folders_are_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "uno"))
            with open(os.path.join(tmp, "uno", "profile.json"), "w", encoding="utf-8") as f:
                json.dump(VEST_PROFILE, f)
            os.makedirs(os.path.join(tmp, "sin_json"))
            self.assertEqual(list(profiles.discover(tmp)), ["uno"])
            self.assertEqual(profiles.load("uno", tmp).name, "Casco y chaleco")
            for bad_id in ("sin_json", "..", "../uno", tmp):
                with self.subTest(id=bad_id), self.assertRaises(KeyError):
                    profiles.load(bad_id, tmp)


if __name__ == "__main__":
    unittest.main()
