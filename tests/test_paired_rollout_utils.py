import json
from pathlib import Path

import numpy as np
import pytest

from mobipi.utils.paired_rollout_utils import (
    base_manifest,
    candidate_run_id,
    compare_payloads,
    load_candidate_config,
    pose_error,
    requested_pose,
    resolve_seeds,
    stable_hash,
    target_relative_pose,
    validate_candidate,
    validate_manifest,
)


def candidate_config():
    return {
        "schema_version": "1.0",
        "review_status": "proposed",
        "coordinate_frame": "world",
        "units": {"translation": "m", "yaw": "rad"},
        "bounds": {"max_abs_dx_m": 0.1, "max_abs_dy_m": 0.1, "max_abs_dyaw_rad": 0.2},
        "candidates": [{"candidate_id": "C0", "requested_transform": [0.0, 0.0, 0.0]}],
    }


def test_seed_scopes_do_not_override_each_other():
    seeds, warning = resolve_seeds(3, None, 4, 5, 6)
    assert seeds == {
        "checkpoint_seed": 3,
        "environment_seed": 4,
        "candidate_seed": 5,
        "evaluation_seed": 6,
    }
    assert warning is None


def test_legacy_seed_maps_only_to_checkpoint_seed():
    seeds, warning = resolve_seeds(None, 7, 4, 5, 6)
    assert seeds["checkpoint_seed"] == 7
    assert seeds["environment_seed"] == 4
    assert warning and "checkpoint_seed" in warning


def test_named_candidate_config_parsing(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(candidate_config()))
    assert load_candidate_config(path)["candidates"][0]["candidate_id"] == "C0"


def test_requested_actual_pose_serialization():
    requested = requested_pose([1.0, 2.0, 0.25], [0.1, -0.1, 0.2], "world")
    error = pose_error(requested, requested + np.array([1e-4, -1e-4, 1e-4]))
    assert json.loads(json.dumps({"requested": requested.tolist(), "error": error}))
    assert error["translation_l2_m"] < 0.001


def test_manifest_fields_and_stable_run_id():
    run_id = candidate_run_id("parent", 2, "C 0")
    assert run_id == "parent-env002-C-0"
    manifest = base_manifest(
        experiment_id="exp",
        run_id=run_id,
        scene_id="scene",
        candidate={"candidate_id": "C0"},
        research={"commit": "r", "branch": "main", "remote": "x", "dirty": False},
        code={"commit": "c", "branch": "b", "remote": "x", "dirty": False},
        environment={"python_executable": "/python"},
        protocol={"config_uri": "/config", "config_sha256": "abc"},
        seeds={"checkpoint_seed": 1, "environment_seed": 2, "candidate_seed": 3, "evaluation_seed": 4},
        command="python run.py",
        output_root="/output",
    )
    validate_manifest(manifest)


def test_fingerprint_is_stable_and_tolerance_aware():
    first, _ = stable_hash({"b": [1.0000001], "a": 2}, tolerance=1e-6)
    second, _ = stable_hash({"a": 2, "b": [1.0000002]}, tolerance=1e-6)
    assert first == second


def test_nested_invariant_comparison_uses_numeric_tolerance():
    result = compare_payloads(
        {"pose": [1.0, 2.0], "schema": {"shape": [2]}},
        {"pose": [1.0005, 2.0], "schema": {"shape": [2]}},
        tolerance=0.001,
    )
    assert result["matched"]
    assert result["numeric_max_abs_diff"] == pytest.approx(0.0005)


def test_invalid_candidate_is_rejected():
    config = candidate_config()
    invalid = {"candidate_id": "bad", "requested_transform": [0.2, 0.0, 0.0]}
    with pytest.raises(ValueError, match="exceeds bounds"):
        validate_candidate(invalid, config)


def test_nominal_base_transform_rotates_translation():
    pose = requested_pose([0.0, 0.0, np.pi / 2], [0.1, 0.0, 0.0], "nominal_base")
    np.testing.assert_allclose(pose[:2], [0.0, 0.1], atol=1e-8)


def test_target_relative_pose_uses_fixture_frame():
    class Fixture:
        name = "cabinet"
        pos = np.array([1.0, 2.0, 0.0])
        rot = np.pi / 2

    class RawEnv:
        door_fxtr = Fixture()

    result = target_relative_pose(RawEnv(), [1.0, 3.0, np.pi])
    np.testing.assert_allclose(result["xy_yaw"], [1.0, 0.0, np.pi / 2], atol=1e-8)
