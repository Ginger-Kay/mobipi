import copy
import random
from collections import deque

import numpy as np
import pytest

from mobipi.utils.handoff_state import (
    HANDOFF_SNAPSHOT_SCHEMA_VERSION,
    IncompleteHandoffSnapshot,
    capture_handoff_snapshot,
    restore_handoff_snapshot,
)


class FakeBaseEnv:
    def __init__(self):
        self.state = {"states": np.array([1.0, 2.0, 3.0])}

    def get_state(self):
        return copy.deepcopy(self.state)

    def reset_to(self, state):
        self.state = copy.deepcopy(state)


class FakeFrameStack:
    def __init__(self):
        self.env = FakeBaseEnv()
        self.num_frames = 3
        self.timestep = 7
        self.obs_history = {
            "image": deque(
                [np.full((1, 2), value, dtype=np.float32) for value in (1, 2, 3)],
                maxlen=self.num_frames,
            ),
            "actions": deque(
                [np.full((2,), value, dtype=np.float32) for value in (4, 5, 6)],
                maxlen=self.num_frames,
            ),
        }

    def get_state(self):
        return self.env.get_state()

    def reset_to(self, state):
        self.env.reset_to(state)
        self.timestep = 0
        self.obs_history = {
            "image": deque(
                [np.zeros((1, 2), dtype=np.float32)] * self.num_frames,
                maxlen=self.num_frames,
            ),
            "actions": deque(
                [np.zeros((2,), dtype=np.float32)] * self.num_frames,
                maxlen=self.num_frames,
            ),
        }


def _controller_adapter(state):
    def get_state():
        return copy.deepcopy(state)

    def set_state(value):
        state.clear()
        state.update(copy.deepcopy(value))

    return get_state, set_state


def test_snapshot_copies_simulator_and_frame_stack_state():
    env = FakeFrameStack()
    controller = {"target": np.array([0.4, -0.2])}
    controller_get, controller_set = _controller_adapter(controller)
    named_rng = np.random.default_rng(31)

    snapshot = capture_handoff_snapshot(
        env,
        controller_get_state=controller_get,
        controller_set_state=controller_set,
        numpy_generators={"navigation": named_rng},
        require_torch=False,
        require_cuda=False,
    )

    env.env.state = {"states": np.array([-9.0, -9.0, -9.0])}
    env.timestep = 99
    env.obs_history["image"][0][...] = -10
    controller["target"][...] = 9

    assert snapshot.schema_version == HANDOFF_SNAPSHOT_SCHEMA_VERSION
    np.testing.assert_array_equal(snapshot.env_state["states"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        snapshot.frame_stacks[0].obs_history["image"][0], [[1.0, 1.0]]
    )
    np.testing.assert_array_equal(snapshot.controller_state["target"], [0.4, -0.2])


def test_restore_reinstates_simulator_frame_stack_controller_and_rng():
    env = FakeFrameStack()
    controller = {"target": np.array([0.4, -0.2])}
    controller_get, controller_set = _controller_adapter(controller)
    named_rng = np.random.default_rng(31)

    random.seed(123)
    np.random.seed(456)
    snapshot = capture_handoff_snapshot(
        env,
        controller_get_state=controller_get,
        controller_set_state=controller_set,
        numpy_generators={"navigation": named_rng},
        require_torch=False,
        require_cuda=False,
    )

    expected = (random.random(), np.random.random(), named_rng.random())
    env.env.state = {"states": np.array([-1.0, -1.0, -1.0])}
    env.timestep = 100
    env.obs_history["image"][0][...] = 100
    controller["target"][...] = 100
    random.seed(999)
    np.random.seed(999)
    named_rng = np.random.default_rng(999)

    restore_handoff_snapshot(
        env,
        snapshot,
        controller_set_state=controller_set,
        numpy_generators={"navigation": named_rng},
        require_torch=False,
        require_cuda=False,
    )
    actual = (random.random(), np.random.random(), named_rng.random())

    np.testing.assert_array_equal(env.env.state["states"], [1.0, 2.0, 3.0])
    assert env.timestep == 7
    np.testing.assert_array_equal(env.obs_history["image"][0], [[1.0, 1.0]])
    np.testing.assert_array_equal(controller["target"], [0.4, -0.2])
    np.testing.assert_allclose(actual, expected)


def test_repeated_restores_reproduce_the_same_handoff_probe():
    env = FakeFrameStack()
    controller = {"target": np.array([0.4, -0.2])}
    controller_get, controller_set = _controller_adapter(controller)
    named_rng = np.random.default_rng(31)
    random.seed(123)
    np.random.seed(456)
    snapshot = capture_handoff_snapshot(
        env,
        controller_get_state=controller_get,
        controller_set_state=controller_set,
        numpy_generators={"navigation": named_rng},
        require_torch=False,
        require_cuda=False,
    )

    probe_results = []
    for replicate in range(5):
        env.env.state = {"states": np.array([replicate] * 3, dtype=np.float64)}
        env.timestep = 100 + replicate
        env.obs_history["image"][0][...] = replicate
        controller["target"][...] = replicate
        random.seed(900 + replicate)
        np.random.seed(900 + replicate)
        named_rng = np.random.default_rng(900 + replicate)

        restore_handoff_snapshot(
            env,
            snapshot,
            controller_set_state=controller_set,
            numpy_generators={"navigation": named_rng},
            require_torch=False,
            require_cuda=False,
        )
        probe_results.append(
            (
                tuple(env.env.state["states"]),
                env.timestep,
                tuple(env.obs_history["image"][0].ravel()),
                controller["target"].copy(),
                random.random(),
                np.random.random(),
                named_rng.random(),
            )
        )

    for result in probe_results[1:]:
        assert result[:3] == probe_results[0][:3]
        np.testing.assert_array_equal(result[3], probe_results[0][3])
        np.testing.assert_allclose(result[4:], probe_results[0][4:])


def test_snapshot_fails_closed_without_frame_stack():
    class NoFrameStackEnv(FakeBaseEnv):
        pass

    controller = {"target": np.array([0.0])}
    controller_get, controller_set = _controller_adapter(controller)
    with pytest.raises(IncompleteHandoffSnapshot, match="frame-stack"):
        capture_handoff_snapshot(
            NoFrameStackEnv(),
            controller_get_state=controller_get,
            controller_set_state=controller_set,
            require_torch=False,
            require_cuda=False,
        )


def test_snapshot_fails_closed_without_controller_adapter():
    with pytest.raises(IncompleteHandoffSnapshot, match="controller state adapter"):
        capture_handoff_snapshot(
            FakeFrameStack(),
            require_torch=False,
            require_cuda=False,
        )


def test_snapshot_fails_closed_when_torch_state_is_required():
    controller = {"target": np.array([0.0])}
    controller_get, controller_set = _controller_adapter(controller)
    with pytest.raises(IncompleteHandoffSnapshot, match="torch_module"):
        capture_handoff_snapshot(
            FakeFrameStack(),
            controller_get_state=controller_get,
            controller_set_state=controller_set,
            require_torch=True,
            require_cuda=False,
        )
