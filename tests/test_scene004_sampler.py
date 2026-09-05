from collections import deque

import numpy as np
import pytest

from mobiwam.scene004_sampler import (
    expansion_reuse_members,
    paired_state_schema,
    persist_source_snapshot_atomic,
    promote_validated_snapshot_group,
    write_source_snapshot_in_group,
)


def snapshot(length=5):
    return {
        "env_state": {"model": "<mujoco/>", "states": np.arange(length, dtype=float), "ep_meta": "{}"},
        "state_schema": {"nq": 2, "nv": 2, "expected_flattened_length": 5, "actual_flattened_length": length, "schema": "time_qpos_qvel"},
        "obs_history": {"x": deque([np.ones((1, 2)) for _ in range(10)])},
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


def test_group_promotes_only_after_all_three_receipts(tmp_path):
    temporary = tmp_path / "attempt" / ".group.tmp"
    receipts = {}
    for suffix, stratum in (("e", "E-compatible"), ("d", "D-required"), ("a", "A-required")):
        source = {"source_id": f"source-{suffix}", "stratum": stratum, "geometry": {}}
        write_source_snapshot_in_group(temporary / source["source_id"], source, snapshot(), {"task": "T", "cell": 1, "environment_seed": 0})
        receipt = tmp_path / "attempt" / source["source_id"] / "receipt.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text('{"status":"pass"}\n')
        receipts[source["source_id"]] = receipt
    canonical = tmp_path / "canonical" / "group"
    aggregate = promote_validated_snapshot_group(temporary, canonical, receipts)
    assert aggregate["complete_snapshot"] is True
    assert canonical.is_dir() and not temporary.exists()
    assert all(__import__("json").loads(path.read_text())["complete_snapshot"] for path in canonical.glob("*/snapshot-meta.json"))


def test_failed_validation_leaves_no_canonical_and_attempt_does_not_overwrite(tmp_path):
    temporary = tmp_path / "attempt" / ".group.tmp"
    source = {"source_id": "source-e", "stratum": "E-compatible", "geometry": {}}
    provenance = {"task": "T", "cell": 1, "environment_seed": 0}
    write_source_snapshot_in_group(temporary / "source-e", source, snapshot(), provenance)
    with pytest.raises(FileExistsError):
        write_source_snapshot_in_group(temporary / "source-e", source, snapshot(), provenance)
    with pytest.raises(ValueError, match="exactly three"):
        promote_validated_snapshot_group(temporary, tmp_path / "canonical" / "group", {})
    assert not (tmp_path / "canonical" / "group").exists()


def test_expansion_reuse_requires_manifest_schedule_and_full_criteria(tmp_path):
    # Use real directories so membership cannot be established by identity alone.
    good = {"task": "T", "cell": 1, "environment_seed": 3, "dimension_closed": True,
            "eligible_for_final_matching": True, "source_paths": []}
    # Path checks are part of the frozen SP0 criteria.
    roots = [str(tmp_path / suffix) for suffix in ("e", "d", "a")]
    for root in roots:
        __import__("pathlib").Path(root).mkdir()
    good["source_paths"] = roots
    wrong_seed = {**good, "environment_seed": 6}
    incomplete = {**good, "environment_seed": 4, "dimension_closed": False}
    accepted = {("T", 1, 3), ("T", 1, 4), ("T", 1, 6)}
    assert expansion_reuse_members([good, wrong_seed, incomplete], {"T": [1]}, accepted) == {("T", 1, 3)}
