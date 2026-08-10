import copy
from types import SimpleNamespace

import numpy as np
import pytest

from mobipi.utils.controller_state import (
    CONTROLLER_STATE_SCHEMA_VERSION,
    UnsupportedControllerState,
    capture_controller_state,
    env_controller_adapters,
    restore_controller_state,
)


class OperationalSpaceController:
    def __init__(self):
        self.goal_pos = np.array([1.0, 2.0, 3.0])
        self.goal_ori = np.eye(3)
        self._goal_update_mode = "achieved"
        self.ori_ref = np.eye(3)
        self.relative_ori = np.array([0.1, 0.2, 0.3])
        self.initial_joint = np.arange(7, dtype=np.float64)
        self.origin_pos = np.array([0.4, 0.5, 0.6])
        self.origin_ori = np.eye(3)
        self.kp = np.ones(6)
        self.kd = np.ones(6) * 2
        self.new_update = True


class MobileBaseJointVelocityController:
    def __init__(self):
        self.goal_qvel = np.array([0.1, 0.2, 0.3])
        self.init_pos = np.array([1.0, 2.0, 0.0])
        self.init_ori = np.eye(3)
        self.new_update = True


class JointPositionController:
    def __init__(self):
        self.goal_qpos = np.array([0.2, 0.3])
        self.kp = np.array([10.0, 20.0])
        self.kd = np.array([1.0, 2.0])
        self.new_update = True


class SimpleGripController:
    def __init__(self):
        self.goal_qvel = np.array([0.7, 0.8])
        self.new_update = True


class HybridMobileBase:
    def __init__(self):
        self.part_controllers = {
            "right": OperationalSpaceController(),
            "right_gripper": SimpleGripController(),
            "base": MobileBaseJointVelocityController(),
            "torso": JointPositionController(),
        }


def test_controller_state_is_explicit_and_deep_copied():
    controller = HybridMobileBase()
    state = capture_controller_state(controller)

    controller.part_controllers["base"].goal_qvel[...] = 99
    controller.part_controllers["right"].goal_pos[...] = 99

    assert state["schema_version"] == CONTROLLER_STATE_SCHEMA_VERSION
    np.testing.assert_array_equal(
        state["parts"]["base"]["fields"]["goal_qvel"], [0.1, 0.2, 0.3]
    )
    np.testing.assert_array_equal(
        state["parts"]["right"]["fields"]["goal_pos"], [1.0, 2.0, 3.0]
    )


def test_controller_state_restores_all_supported_parts():
    source = HybridMobileBase()
    state = capture_controller_state(source)
    target = HybridMobileBase()
    target.part_controllers["base"].goal_qvel[...] = -1
    target.part_controllers["right"].goal_pos[...] = -1

    restore_controller_state(target, state)

    for part_name, part_state in state["parts"].items():
        for field, expected in part_state["fields"].items():
            actual = getattr(target.part_controllers[part_name], field)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected


def test_env_adapter_resolves_controller_after_reset_replacement():
    first = HybridMobileBase()
    second = HybridMobileBase()
    holder = SimpleNamespace(robots=[SimpleNamespace(composite_controller=first)])
    env = SimpleNamespace(unwrapped=SimpleNamespace(env=holder))
    get_state, set_state = env_controller_adapters(env)
    state = get_state()

    holder.robots[0].composite_controller = second
    second.part_controllers["base"].goal_qvel[...] = -7
    set_state(state)

    np.testing.assert_array_equal(
        second.part_controllers["base"].goal_qvel, [0.1, 0.2, 0.3]
    )
    np.testing.assert_array_equal(
        first.part_controllers["base"].goal_qvel, [0.1, 0.2, 0.3]
    )


def test_unsupported_controller_fails_closed():
    class UnknownController:
        pass

    composite = SimpleNamespace(part_controllers={"base": UnknownController()})
    with pytest.raises(UnsupportedControllerState, match="unsupported controller"):
        capture_controller_state(composite)


def test_controller_restore_does_not_alias_snapshot_arrays():
    controller = HybridMobileBase()
    state = capture_controller_state(controller)
    restored = HybridMobileBase()
    restore_controller_state(restored, copy.deepcopy(state))
    restored.part_controllers["right"].goal_ori[...] = 11
    np.testing.assert_array_equal(state["parts"]["right"]["fields"]["goal_ori"], np.eye(3))
