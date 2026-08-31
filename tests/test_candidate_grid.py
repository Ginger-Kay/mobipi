import json
import unittest
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


class PilotCandidateGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = json.loads((CONFIG_ROOT / "pilot_v1.json").read_text())
        cls.v2 = json.loads((CONFIG_ROOT / "pilot_v2.json").read_text())

    def test_redesign_preserves_schedule_execute_and_assist(self):
        for key in (
            "source_count",
            "tasks",
            "task_families",
            "layouts",
            "seeds_per_candidate",
            "schedule_seed",
            "environment_seed_start",
            "policy_seed_start",
            "route_seed_start",
            "save_video",
        ):
            self.assertEqual(self.v2[key], self.v1[key])
        self.assertEqual(self.v2["execute_candidates"], self.v1["execute_candidates"])
        self.assertEqual(self.v2["assist_candidates"], self.v1["assist_candidates"])

    def test_redesign_freezes_five_unique_direct_standoff_candidates(self):
        candidates = self.v2["dock_candidates"]
        self.assertEqual([item["candidate_id"] for item in candidates], ["d0", "d1", "d2", "d3", "d4"])
        expected_offsets = [
            [-0.08, 0.0],
            [-0.08, 0.04],
            [-0.08, -0.04],
            [-0.12, 0.0],
            [-0.16, 0.0],
        ]
        self.assertEqual(
            [item["candidate_params"]["target_offset_local_xy_m"] for item in candidates],
            expected_offsets,
        )
        self.assertEqual(
            len({item["candidate_params"]["approach_variant"] for item in candidates}),
            5,
        )
        for item in candidates:
            params = item["candidate_params"]
            self.assertTrue(params["approach_variant"].endswith("_direct"))
            self.assertFalse(any(key.startswith("waypoint_") for key in params))
            self.assertEqual(params["dock_max_steps"], 240)
            self.assertEqual(params["position_tolerance_m"], 0.005)
            self.assertEqual(params["yaw_tolerance_rad"], 0.017453292519943295)
            self.assertEqual(params["command_gain"], 1.0)

    def test_redesign_metadata_is_pilot_only_and_same_semantics(self):
        redesign = self.v2["redesign"]
        self.assertEqual(redesign["round"], 2)
        self.assertEqual(redesign["redesigned_route"], "D")
        self.assertTrue(redesign["same_option_semantics"])
        self.assertLess(redesign["trigger_dock_hard_valid_rate"], redesign["trigger_threshold"])


if __name__ == "__main__":
    unittest.main()
