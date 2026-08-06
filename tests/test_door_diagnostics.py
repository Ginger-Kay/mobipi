import json
from pathlib import Path

import numpy as np
import pytest

from mobipi.utils.door_diagnostics import (
    DEFAULT_STALL_CONFIG,
    aggregate_robot_door_contacts,
    crossing_event,
    door_open_fraction_from_raw,
    normalize_closing_progress,
    plane_from_points,
    raw_threshold_from_normalized,
    signed_distance_to_plane,
    validate_case_list,
    validate_diagnostic_trajectory,
)
from mobipi.utils.paired_rollout_utils import load_candidate_config, sha256_file


def test_door_progress_uses_actual_negative_raw_joint_direction():
    threshold = raw_threshold_from_normalized()
    values = np.array([-1.5, -0.75, threshold])
    progress = normalize_closing_progress(values[0], values, threshold)
    np.testing.assert_allclose(progress, [0.0, (0.75) / (threshold + 1.5), 1.0])
    np.testing.assert_allclose(door_open_fraction_from_raw(values), [1.5 / (np.pi / 2), 0.47746483, 0.05], atol=1e-7)


def test_plane_signed_distance_and_dynamic_axes():
    origin, u, v, normal = plane_from_points(
        [[0, 0, 0], [1, 0, 0], [0, 0, 1]]
    )
    np.testing.assert_allclose(origin, [0, 0, 0])
    np.testing.assert_allclose(np.abs(normal), [0, 1, 0])
    assert signed_distance_to_plane([0, 2, 0], origin, normal) == pytest.approx(normal[1] * 2)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_crossing_hysteresis_is_single_positive_to_negative_event():
    assert not crossing_event(0.004, -0.004, already_crossed=False, hysteresis_m=0.005)
    assert crossing_event(0.006, -0.006, already_crossed=False, hysteresis_m=0.005)
    assert not crossing_event(-0.006, -0.007, already_crossed=True, hysteresis_m=0.005)


def test_contact_aggregation_rejects_proximity_and_preserves_pair_metadata():
    result = aggregate_robot_door_contacts(
        [
            {"robot_geom": "robot0_hand", "door_geom": "microwave_door", "distance": 0.01},
            {
                "robot_geom": "robot0_hand",
                "robot_body": "robot0_link",
                "door_geom": "microwave_door_handle",
                "door_body": "microwave_door",
                "distance": -0.002,
                "position": [1, 2, 3],
                "normal": [0, 1, 0],
            },
        ]
    )
    assert result["active"] and result["pair_count"] == 1
    assert result["min_distance_m"] == pytest.approx(-0.002)
    assert result["pairs"][0]["door_body"] == "microwave_door"


def _synthetic_trajectory(steps=25):
    n = steps + 1
    raw = np.full(n, -1.0)
    threshold = raw_threshold_from_normalized()
    eef = np.tile(np.array([[0.0, 1.0, 0.0]]), (n, 1))
    normal = np.tile(np.array([[0.0, 1.0, 0.0]]), (n, 1))
    zeros3 = np.zeros((n, 3))
    zeros2 = np.zeros((n, 2))
    trajectory = {
        "diagnostic__door_joint_raw": raw,
        "diagnostic__door_joint_initial_raw": raw.copy(),
        "diagnostic__door_joint_success_threshold_raw": np.full(n, threshold),
        "diagnostic__door_joint_open_fraction": door_open_fraction_from_raw(raw),
        "diagnostic__door_closing_progress": np.zeros(n),
        "diagnostic__eef_world_pos": eef,
        "diagnostic__door_plane_origin": zeros3.copy(),
        "diagnostic__door_plane_normal": normal,
        "diagnostic__eef_signed_distance_m": np.ones(n),
        "diagnostic__eef_on_exterior_side": np.ones(n, dtype=bool),
        "diagnostic__door_plane_crossed": np.zeros(n, dtype=bool),
        "diagnostic__eef_plane_projection_uv_m": zeros2.copy(),
        "diagnostic__door_site_span_uv_m": np.ones((n, 2)),
        "diagnostic__robot_door_contact_active": np.zeros(n, dtype=bool),
        "diagnostic__robot_door_contact_pair_count": np.zeros(n, dtype=np.int64),
        "diagnostic__robot_door_contact_min_distance_m": np.zeros(n),
        "diagnostic__robot_door_contact_position": zeros3.copy(),
        "diagnostic__robot_door_contact_normal": zeros3.copy(),
        "diagnostic__door_stall": np.r_[np.zeros(steps, dtype=bool), True],
        "diagnostic__eef_stall": np.r_[np.zeros(steps, dtype=bool), True],
        "diagnostic__action_stall": np.r_[np.zeros(steps, dtype=bool), True],
        "diagnostic__contact_stall": np.zeros(n, dtype=bool),
        "diagnostic__stall": np.r_[np.zeros(steps, dtype=bool), True],
        "diagnostic__eef_window_displacement_m": np.r_[np.zeros(steps), 0.0],
        "diagnostic__door_window_progress_range": np.r_[np.zeros(steps), 0.0],
        "diagnostic__action_norm": np.full(steps, 1.2),
        "diagnostic__action_delta_norm": np.zeros(steps),
        "diagnostic__eef_step_displacement_m": np.zeros(steps),
    }
    return trajectory


def test_stall_window_and_serialized_diagnostic_validator():
    trajectory = _synthetic_trajectory()
    summary = {
        "first_crossing_step": -1,
        "first_contact_step": -1,
        "last_contact_step": -1,
        "contact_duration_steps": 0,
        "first_stall_step": 25,
    }
    validate_diagnostic_trajectory(trajectory, 25, summary=summary)


def test_case_list_is_exactly_four_failures_and_four_controls():
    root = Path(__file__).resolve().parents[1]
    case_path = root / "mobipi/configs/diagnostics/close_single_door_door_crossing_20260806.json"
    candidate_path = root / "mobipi/configs/paired_candidates/close_single_door_layout1_style1_frozen.json"
    payload = json.loads(case_path.read_text())
    candidates = load_candidate_config(candidate_path)
    validated = validate_case_list(
        payload,
        candidate_ids=[item["candidate_id"] for item in candidates["candidates"]],
        candidate_config_sha256=payload["candidate_config_sha256"],
    )
    assert len(validated["cases"]) == 8
    assert sha256_file(case_path)


def test_pure_diagnostics_do_not_touch_numpy_rng():
    state = np.random.get_state()
    normalize_closing_progress(-1.0, [-1.0, -0.5], raw_threshold_from_normalized())
    plane_from_points([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
    aggregate_robot_door_contacts([])
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
