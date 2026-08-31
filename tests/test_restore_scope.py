import unittest
from types import SimpleNamespace

import numpy as np

from mobiwam.adapters.mobipi import (
    _capture_controller_state,
    _contact_hash,
    _controller_state_hash,
    _restore_controller_state,
)


class FakePartController:
    def __init__(self):
        self.goal = np.array([1.0, 2.0])
        self.integrator = 0.25
        self.sim = object()


class FakeComposite:
    def __init__(self):
        self.part_controllers = {"right": FakePartController()}
        self.enabled = True


class FakeRobot:
    def __init__(self):
        self.composite_controller = FakeComposite()


class FakeRaw:
    def __init__(self):
        self.robots = [FakeRobot()]


class ControllerRestoreTest(unittest.TestCase):
    def test_numeric_controller_buffers_are_restored_and_hashed(self):
        raw = FakeRaw()
        state = _capture_controller_state(raw)
        expected_hash = _controller_state_hash(state)
        controller = raw.robots[0].composite_controller.part_controllers["right"]
        controller.goal[:] = 9.0
        controller.integrator = 8.0
        _restore_controller_state(raw, state)
        restored = _capture_controller_state(raw)
        self.assertEqual(_controller_state_hash(restored), expected_hash)
        np.testing.assert_array_equal(restored["right"]["goal"], [1.0, 2.0])

    def test_contact_hash_scopes_out_fixture_self_collisions(self):
        names = {
            0: "cabinet_bottom",
            1: "cabinet_door",
            2: "robot0_gripper",
        }

        def contact(geom1, geom2, distance):
            return SimpleNamespace(
                geom1=geom1,
                geom2=geom2,
                dist=np.array(distance),
                pos=np.zeros(3),
                frame=np.eye(3).reshape(-1),
                friction=np.ones(5),
            )

        fixture_contact = contact(0, 1, -0.01)
        robot_contact = contact(2, 1, -0.02)
        data = SimpleNamespace(contact=[fixture_contact, robot_contact], ncon=2)
        model = SimpleNamespace(geom_id2name=names.__getitem__)
        raw = SimpleNamespace(sim=SimpleNamespace(data=data, model=model))
        reference = _contact_hash(raw)

        fixture_contact.dist = np.array(-0.5)
        self.assertEqual(_contact_hash(raw), reference)
        robot_contact.dist = np.array(-0.5)
        self.assertNotEqual(_contact_hash(raw), reference)


if __name__ == "__main__":
    unittest.main()
