import unittest

import numpy as np

from mobiwam.assist_trajectory import build_truncated_assist_trajectory


def pose(x=0.0, y=0.0, yaw=0.0):
    value = np.eye(4)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    value[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    value[:2, 3] = [x, y]
    return value


class AssistTrajectoryTest(unittest.TestCase):
    def test_zero_fraction_is_a_zero(self):
        start = pose(1.0, 2.0, 0.3)
        result = build_truncated_assist_trajectory(
            start, pose(2.0, 3.0, 1.0), fraction_toward_dock=0.0
        )
        for generated in result.poses_world:
            np.testing.assert_allclose(generated, start, atol=1e-9)

    def test_canonical_assist_is_truncated_to_five_cm_and_three_degrees(self):
        result = build_truncated_assist_trajectory(
            pose(), pose(1.0, 1.0, np.pi / 2), steps=10
        )
        self.assertEqual(result.poses_world.shape, (11, 4, 4))
        self.assertAlmostEqual(result.translation_m, 0.05)
        self.assertAlmostEqual(result.yaw_rad, np.deg2rad(3.0))
        np.testing.assert_allclose(result.poses_world[0], np.eye(4), atol=1e-9)
        self.assertAlmostEqual(
            np.linalg.norm(result.poses_world[-1, :2, 3]), 0.05
        )


if __name__ == "__main__":
    unittest.main()
