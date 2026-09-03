import unittest

import numpy as np

from mobiwam.persistent_assist import (
    compile_persistent_assist,
    compile_source_without_outcome,
    realized_motion_metrics,
)


def pose(x=0.0, y=0.0):
    p = np.eye(4)
    p[:2, 3] = [x, y]
    return p


class PersistentAssistTest(unittest.TestCase):
    def test_actual_pose_is_used_at_each_boundary(self):
        plan = compile_persistent_assist(
            [pose(1, 0), pose(1, 0)], [pose(), pose(0.2, 0)], "a2"
        )
        self.assertAlmostEqual(plan.chunk_poses_world[0, 0, 3], 0.45)
        self.assertAlmostEqual(plan.chunk_poses_world[1, 0, 3], 0.65)
        self.assertEqual(plan.chunk_count, 2)

    def test_a0_is_not_formal_candidate(self):
        with self.assertRaises(ValueError):
            compile_persistent_assist([pose()], [pose()], "a0")

    def test_realized_motion_ignores_target_hold(self):
        result = realized_motion_metrics(
            [pose(), pose(), pose()], [[0, 0], [0, 0], [0, 0]], moving_threshold_mps=0.01
        )
        self.assertEqual(result["moving_steps"], 0)
        self.assertEqual(result["actual_net_translation_m"], 0.0)

    def test_source_compiler_rejects_outcome(self):
        with self.assertRaises(ValueError):
            compile_source_without_outcome({"target_visible": True, "success": True})


if __name__ == "__main__":
    unittest.main()
