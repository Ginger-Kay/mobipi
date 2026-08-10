import copy
from collections import deque

import numpy as np
import pytest

from types import SimpleNamespace

from mobipi.eval.eval_ec2_history_swap import _approach_definition, _max_abs
from mobipi.utils.handoff_state import capture_handoff_snapshot
from mobipi.utils.history_swap import (
    HistorySwapError,
    build_history_intervention,
    validate_history_intervention,
)

from test_handoff_state import FakeFrameStack, _controller_adapter


def _snapshot(values, state_value, timestep):
    env = FakeFrameStack()
    env.env.state = {"states": np.full(3, state_value, dtype=np.float64)}
    env.timestep = timestep
    env.obs_history = {
        "image": deque(
            [np.full((1, 2), value, dtype=np.float32) for value in values],
            maxlen=3,
        ),
        "actions": deque(
            [np.full((2,), value + 10, dtype=np.float32) for value in values],
            maxlen=3,
        ),
    }
    controller = {"target": np.array([state_value])}
    controller_get, controller_set = _controller_adapter(controller)
    return capture_handoff_snapshot(
        env,
        controller_get_state=controller_get,
        controller_set_state=controller_set,
        require_torch=False,
        require_cuda=False,
    )


def test_swap_replaces_only_past_frames_for_every_key():
    physical = _snapshot((1, 2, 3), state_value=11, timestep=7)
    history = _snapshot((4, 5, 6), state_value=22, timestep=9)

    hybrid = build_history_intervention(
        physical, history, physical_source="A", history_source="B"
    )
    validate_history_intervention(hybrid, physical, history)

    stack = hybrid.frame_stacks[0]
    np.testing.assert_array_equal(stack.obs_history["image"][0], [[4, 4]])
    np.testing.assert_array_equal(stack.obs_history["image"][1], [[5, 5]])
    np.testing.assert_array_equal(stack.obs_history["image"][2], [[3, 3]])
    np.testing.assert_array_equal(stack.obs_history["actions"][2], [13, 13])
    np.testing.assert_array_equal(hybrid.env_state["states"], [11, 11, 11])
    np.testing.assert_array_equal(hybrid.controller_state["target"], [11])
    assert stack.timestep == 7
    assert hybrid.metadata["history_intervention"]["current_source"] == "A"


def test_swap_deep_copies_both_sources():
    physical = _snapshot((1, 2, 3), state_value=11, timestep=7)
    history = _snapshot((4, 5, 6), state_value=22, timestep=9)
    hybrid = build_history_intervention(
        physical, history, physical_source="A", history_source="B"
    )

    hybrid.frame_stacks[0].obs_history["image"][0][...] = 99
    hybrid.frame_stacks[0].obs_history["image"][-1][...] = 98
    assert not np.any(history.frame_stacks[0].obs_history["image"][0] == 99)
    assert not np.any(physical.frame_stacks[0].obs_history["image"][-1] == 98)


def test_swap_fails_closed_on_key_mismatch():
    physical = _snapshot((1, 2, 3), state_value=11, timestep=7)
    history = _snapshot((4, 5, 6), state_value=22, timestep=9)
    del history.frame_stacks[0].obs_history["actions"]
    with pytest.raises(HistorySwapError, match="keys differ"):
        build_history_intervention(
            physical, history, physical_source="A", history_source="B"
        )


def test_swap_fails_closed_on_frame_shape_mismatch():
    physical = _snapshot((1, 2, 3), state_value=11, timestep=7)
    history = _snapshot((4, 5, 6), state_value=22, timestep=9)
    changed = list(history.frame_stacks[0].obs_history["image"])
    changed[1] = np.zeros((1, 3), dtype=np.float32)
    history.frame_stacks[0].obs_history["image"] = tuple(changed)
    with pytest.raises(HistorySwapError, match="frame schema differs"):
        build_history_intervention(
            physical, history, physical_source="A", history_source="B"
        )


def test_validator_rejects_current_frame_from_history_source():
    physical = _snapshot((1, 2, 3), state_value=11, timestep=7)
    history = _snapshot((4, 5, 6), state_value=22, timestep=9)
    hybrid = build_history_intervention(
        physical, history, physical_source="A", history_source="B"
    )
    hybrid.frame_stacks[0].obs_history["image"] = tuple(
        list(hybrid.frame_stacks[0].obs_history["image"][:-1])
        + [copy.deepcopy(history.frame_stacks[0].obs_history["image"][-1])]
    )
    with pytest.raises(HistorySwapError, match="current frame"):
        validate_history_intervention(hybrid, physical, history)


def test_numeric_comparison_handles_boolean_controller_fields():
    assert _max_abs(np.array([True, False]), np.array([True, False])) == 0.0
    assert _max_abs(np.array([True, False]), np.array([False, False])) == 1.0


def test_approach_definition_freezes_common_terminal_alignment():
    definition, _digest = _approach_definition(
        SimpleNamespace(
            target_x=0.1,
            target_y=0.0,
            detour_x=0.0,
            detour_y=0.08,
            settle_steps=5,
        )
    )
    assert definition["terminal_alignment"] == {
        "normalize_simulator_time_before_settle": True,
        "neutral_action_steps": 5,
    }
