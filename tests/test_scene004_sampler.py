from collections import deque

import numpy as np
import pytest

from mobiwam.scene004_sampler import paired_state_schema, persist_source_snapshot_atomic


def snapshot(length=5):
    return {
        "env_state": {"model": "<mujoco/>", "states": np.arange(length, dtype=float), "ep_meta": "{}"},
        "state_schema": {"nq": 2, "nv": 2, "expected_flattened_length": 5, "actual_flattened_length": length, "schema": "time_qpos_qvel"},
        "obs_history": {"x": deque([np.ones((1, 2))])},
        "python_rng": (3, (), None),
        "numpy_rng": ("MT19937", np.array([1], dtype=np.uint32), 0, 0, 0.0),
        "policy_evidence": {"no_actuation": True, "chunk_sha256": "abc"},
    }


def test_paired_state_schema_rejects_mismatch():
    with pytest.raises(ValueError, match="state length"):
        paired_state_schema(snapshot(4))


def test_atomic_snapshot_is_camera_independent(tmp_path):
    source = {"source_id": "CloseDrawer-l1-seed00-e", "stratum": "E-compatible", "geometry": {}}
    provenance = {"task": "CloseDrawer", "cell": 1, "environment_seed": 0}
    path, meta = persist_source_snapshot_atomic(tmp_path, source, snapshot(), provenance)
    assert path.is_dir()
    assert meta["camera_independent"] is True
    assert meta["state_schema"]["expected_flattened_length"] == 5
    assert all((path / name).exists() for name in meta["files"])
