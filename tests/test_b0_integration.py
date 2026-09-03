import unittest

from mobiwam.b0_integration import (
    validate_a0_identity,
    validate_native_external_camera,
    validate_persistent_trace,
    validate_scene_template,
    validate_realized_total_travel,
)


class B0IntegrationTest(unittest.TestCase):
    def test_persistent_query_cadence_and_minimum_chunks(self):
        validate_persistent_trace({"assist_chunk_count": 3, "assist_query_count": 3})
        with self.assertRaises(ValueError):
            validate_persistent_trace({"assist_chunk_count": 2, "assist_query_count": 2})

    def test_a0_identity_is_full_trajectory(self):
        validate_a0_identity({"actions": [1], "states": [2], "observations": [3], "terminal": [4]}, {"actions": [1], "states": [2], "observations": [3], "terminal": [4]})
        with self.assertRaises(ValueError):
            validate_a0_identity({"actions": [1], "states": [2], "observations": [3], "terminal": [4]}, {"actions": [1], "states": [9], "observations": [3], "terminal": [4]})

    def test_native_camera_rejects_low_res_and_upscale(self):
        validate_native_external_camera({"camera_type": "external_world_frame_fixed", "width": 1920, "height": 1080, "upscale_ratio": 1.0})
        with self.assertRaises(ValueError):
            validate_native_external_camera({"camera_type": "policy_view", "width": 256, "height": 256})

    def test_scene_fixture_constraints(self):
        validate_scene_template({"task": "CloseDrawer", "fixture_type": "full_size_kitchen_drawer", "free_swept_corridor_m": 0.5})
        with self.assertRaises(ValueError):
            validate_scene_template({"task": "CloseSingleDoor", "fixture_type": "microwave", "free_swept_corridor_m": 1.0})

    def test_realized_total_travel_cap(self):
        validate_realized_total_travel(0.45, 0.45)
        with self.assertRaises(ValueError):
            validate_realized_total_travel(0.451, 0.45)


if __name__ == "__main__":
    unittest.main()
