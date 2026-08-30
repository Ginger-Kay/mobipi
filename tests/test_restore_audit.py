import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_mobipi_restores.py"
SPEC = importlib.util.spec_from_file_location("audit_mobipi_restores", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load restore audit script")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class RestoreAuditToleranceTest(unittest.TestCase):
    def assert_within_default_tolerance(self, reference, current, expected):
        comparison = AUDIT.compare_observations(reference, current)
        self.assertEqual(
            AUDIT.observation_within_tolerance(
                comparison,
                image_max_abs_error=AUDIT.DEFAULT_IMAGE_MAX_ABS_ERROR,
                image_max_changed_fraction=AUDIT.DEFAULT_IMAGE_MAX_CHANGED_FRACTION,
            ),
            expected,
        )

    def test_accepts_sparse_one_gray_level_image_jitter(self):
        reference = {"camera_image": np.zeros((10, 3, 128, 128), dtype=np.float32)}
        current = {key: value.copy() for key, value in reference.items()}
        current["camera_image"].reshape(-1)[:15] = np.float32(1.0 / 255.0)
        self.assert_within_default_tolerance(reference, current, True)

    def test_rejects_non_image_drift(self):
        reference = {"robot_state": np.zeros(10, dtype=np.float32)}
        current = {"robot_state": reference["robot_state"].copy()}
        current["robot_state"][0] = np.float32(1e-7)
        self.assert_within_default_tolerance(reference, current, False)

    def test_rejects_large_or_dense_image_drift(self):
        reference = {"camera_image": np.zeros(1000, dtype=np.float32)}
        large = {"camera_image": reference["camera_image"].copy()}
        large["camera_image"][0] = np.float32(2.0 / 255.0)
        self.assert_within_default_tolerance(reference, large, False)

        dense = {"camera_image": reference["camera_image"].copy()}
        dense["camera_image"][:2] = np.float32(1.0 / 255.0)
        self.assert_within_default_tolerance(reference, dense, False)

    def test_state_comparison_reports_largest_changed_indices(self):
        reference = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        current = np.array([0.0, 1.25, 1.5], dtype=np.float64)
        comparison = AUDIT.compare_state_vectors(reference, current)
        self.assertFalse(comparison["exact_equal"])
        self.assertEqual(comparison["changed_elements"], 2)
        self.assertEqual(comparison["max_abs_error"], 0.5)
        self.assertEqual(
            [item["flat_index"] for item in comparison["largest_differences"]],
            [2, 1],
        )

    def test_state_metadata_comparison_ignores_only_declared_volatile_metadata(self):
        reference = {
            "model": "<mujoco name='expected'/>",
            "ep_meta": '{"lang":"close","object_cfgs":[{"path":"old"}]}',
        }
        current = {
            "model": "<mujoco name='observed'/>",
            "ep_meta": '{"object_cfgs":[{"path":"new"}],"lang":"close"}',
        }
        comparison = AUDIT.compare_state_metadata(reference, current)
        self.assertFalse(comparison["model_equal"])
        self.assertTrue(comparison["stable_ep_meta_equal"])
        self.assertEqual(comparison["differing_stable_ep_meta_keys"], [])
        self.assertIsInstance(comparison["first_model_difference"], int)


if __name__ == "__main__":
    unittest.main()
