import unittest

import numpy as np

from mobiwam.mobipi_actions import (
    BASE,
    compensate_world_intent,
    lock_base,
    nominal_world_intent,
    split_action,
)


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    value = np.eye(4)
    value[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    value[:3, 3] = [x, y, z]
    return value


class MobiPiActionTest(unittest.TestCase):
    def test_layout_and_base_lock(self):
        action = np.arange(12, dtype=np.float64) / 12.0
        parts = split_action(action)
        np.testing.assert_allclose(parts.arm_position, action[:3])
        np.testing.assert_allclose(parts.arm_rotation, action[3:6])
        self.assertEqual(parts.torso, action[6])
        np.testing.assert_allclose(parts.base, action[7:10])
        self.assertEqual(parts.gripper, action[10])
        self.assertEqual(parts.control_mode, action[11])
        locked = lock_base(action)
        np.testing.assert_allclose(locked[BASE], 0.0)
        np.testing.assert_allclose(locked[:7], action[:7])
        np.testing.assert_allclose(locked[10:], action[10:])

    def test_nominal_world_intent_uses_controller_scale(self):
        action = np.zeros(12)
        action[0] = 0.2
        desired = nominal_world_intent(action, pose(), pose(0.5, 0.0, 1.0))
        np.testing.assert_allclose(desired[:3, 3], [0.51, 0.0, 1.0], atol=1e-9)

    def test_a_zero_is_exactly_nominal_arm_action(self):
        action = np.zeros(12)
        action[:6] = [0.2, -0.1, 0.05, 0.02, -0.03, 0.04]
        action[6:] = [0.1, 0.0, 0.0, 0.0, -1.0, -1.0]
        origin = pose(0.2, -0.3, 0.0, yaw=0.4)
        eef = origin @ pose(0.5, 0.1, 0.8, yaw=-0.2)
        result = compensate_world_intent(
            action,
            nominal_origin_pose_world=origin,
            nominal_eef_pose_world=eef,
            assist_origin_pose_world_current=origin,
            assist_origin_pose_world_next=origin,
            assist_eef_pose_world_current=eef,
        )
        np.testing.assert_allclose(result.action, action, atol=1e-9)
        self.assertFalse(result.saturated)
        self.assertLess(result.transform_closure_pos_error_m, 1e-12)
        self.assertLess(result.transform_closure_rot_error_rad, 1e-9)

    def test_base_motion_is_compensated_in_arm_command(self):
        action = np.zeros(12)
        origin = pose()
        eef = pose(0.5, 0.0, 1.0)
        result = compensate_world_intent(
            action,
            nominal_origin_pose_world=origin,
            nominal_eef_pose_world=eef,
            assist_origin_pose_world_current=origin,
            assist_origin_pose_world_next=pose(x=0.01),
            assist_eef_pose_world_current=eef,
        )
        self.assertAlmostEqual(result.action[0], -0.2, places=9)
        np.testing.assert_allclose(result.desired_eef_pose_world, eef, atol=1e-9)
        self.assertLess(result.transform_closure_pos_error_m, 1e-12)
        self.assertLess(result.transform_closure_rot_error_rad, 1e-9)

    def test_base_motion_can_supply_nominal_world_motion(self):
        action = np.zeros(12)
        action[0] = 0.2
        origin = pose()
        eef = pose(0.5, 0.0, 1.0)
        result = compensate_world_intent(
            action,
            nominal_origin_pose_world=origin,
            nominal_eef_pose_world=eef,
            assist_origin_pose_world_current=origin,
            assist_origin_pose_world_next=pose(x=0.01),
            assist_eef_pose_world_current=eef,
        )
        self.assertAlmostEqual(result.action[0], 0.0, places=9)

    def test_saturation_fails_transform_closure(self):
        action = np.zeros(12)
        origin = pose()
        eef = pose(0.5, 0.0, 1.0)
        result = compensate_world_intent(
            action,
            nominal_origin_pose_world=origin,
            nominal_eef_pose_world=eef,
            assist_origin_pose_world_current=origin,
            assist_origin_pose_world_next=pose(x=0.10),
            assist_eef_pose_world_current=eef,
        )
        self.assertTrue(result.saturated)
        self.assertGreater(result.transform_closure_pos_error_m, 0.0)


if __name__ == "__main__":
    unittest.main()
