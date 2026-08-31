import unittest

import numpy as np

from mobiwam.adapters.mobipi import (
    _capture_controller_state,
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


if __name__ == "__main__":
    unittest.main()
