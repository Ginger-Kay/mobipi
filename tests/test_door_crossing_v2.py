import numpy as np
import pytest

from mobipi.utils.door_crossing_v2 import CrossingV2Config, analyze_crossing_v2


def test_deadband_latch_records_interpolated_crossing_and_not_v1_jump():
    distance = np.array([0.008, 0.001, -0.006], dtype=np.float64)
    joint = np.array([0.0, 0.2, 0.4])
    progress = np.array([0.0, 0.1, 0.2])
    uv = np.array([[0.01, 0.01], [0.02, 0.02], [0.03, 0.03]])
    span = np.array([0.05, 0.05])
    result = analyze_crossing_v2(
        distance,
        door_joint_raw=joint,
        door_progress=progress,
        eef_plane_projection_uv_m=uv,
        door_site_span_uv_m=span,
    )
    assert result["legacy_v1_crossing_count"] == 0
    assert result["crossing_count"] == 1
    event = result["events"][0]
    assert event["direction"] == "positive_to_negative"
    assert event["crossing_step"] == 2
    assert event["deadband_steps"] == 1
    assert event["door_joint_raw"] == pytest.approx(0.22857142857142856)
    assert event["door_progress"] == pytest.approx(0.1142857142857143)
    assert event["finite_region_valid"] is True


def test_initial_or_always_one_sided_series_has_no_crossing():
    for distance in (np.array([0.006, 0.001, 0.004]), np.array([-0.006, -0.001, -0.004])):
        result = analyze_crossing_v2(distance)
        assert result["crossing_count"] == 0
        assert result["relative_plane_side_transition"] is False


def test_reverse_crossing_and_jitter_do_not_double_count_deadband():
    distance = np.array([0.006, 0.001, -0.006, -0.001, 0.001, 0.006, 0.004, -0.006])
    result = analyze_crossing_v2(distance)
    assert result["crossing_directions"] == ["positive_to_negative", "negative_to_positive", "positive_to_negative"]
    assert result["crossing_count"] == 3
    assert result["reverse_crossing"] is True


def test_exact_thresholds_are_inclusive():
    result = analyze_crossing_v2(np.array([0.005, 0.0, -0.005]))
    assert result["crossing_count"] == 1
    assert result["events"][0]["crossing_step"] == 2


def test_nan_length_and_shape_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        analyze_crossing_v2(np.array([0.006, np.nan]))
    with pytest.raises(ValueError, match="shape"):
        analyze_crossing_v2(np.array([0.006, -0.006]), door_progress=np.array([0.0]))
    with pytest.raises(ValueError, match="width"):
        analyze_crossing_v2(np.array([0.006, -0.006]), eef_plane_projection_uv_m=np.zeros((2, 3)))


def test_geometry_labels_plane_motion_and_finite_region_without_mutation_or_rng():
    distance = np.array([0.006, -0.006])
    uv = np.array([[0.01, 0.01], [0.01, 0.01]])
    span = np.array([[0.02, 0.02], [0.02, 0.02]])
    eef = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    origin = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    normal = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    originals = [item.copy() for item in (distance, uv, span, eef, origin, normal)]
    state = np.random.get_state()
    result = analyze_crossing_v2(
        distance,
        eef_plane_projection_uv_m=uv,
        door_site_span_uv_m=span,
        eef_world_pos=eef,
        door_plane_origin=origin,
        door_plane_normal=normal,
    )
    after = np.random.get_state()
    assert all(np.array_equal(before, after_item) for before, after_item in zip(state, after))
    assert all(np.array_equal(before, after_item) for before, after_item in zip(originals, (distance, uv, span, eef, origin, normal)))
    event = result["events"][0]
    assert event["finite_region_valid"] is True
    assert event["plane_motion"]["ambiguity"] is True


def test_config_validation():
    assert CrossingV2Config().as_dict()["hysteresis_m"] == 0.005
    with pytest.raises(ValueError):
        CrossingV2Config(hysteresis_m=0.0)
