"""Atomic persistence helpers for corrected SCENE-004 source snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mobiwam.scene004 import canonical_hash


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_state_schema(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    schema = dict(snapshot["state_schema"])
    expected = int(schema["expected_flattened_length"])
    actual = int(len(np.asarray(snapshot["env_state"]["states"])))
    if actual != expected or actual != int(schema["actual_flattened_length"]):
        raise ValueError(f"paired snapshot state length {actual} != {expected}")
    return schema


def persist_source_snapshot_atomic(
    root: Path,
    source: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Write one source family member to a temporary child and atomically promote it."""
    final = root / str(source["source_id"])
    temporary = root / f".{source['source_id']}.tmp-{os.getpid()}"
    if final.exists() or temporary.exists():
        raise FileExistsError(f"snapshot target already exists: {final}")
    temporary.mkdir(parents=True)
    try:
        schema = paired_state_schema(snapshot)
        (temporary / "model.xml").write_text(str(snapshot["env_state"]["model"]))
        np.save(temporary / "sim_state.npy", np.asarray(snapshot["env_state"]["states"]))
        history = {key: np.concatenate(list(values), axis=0) for key, values in snapshot["obs_history"].items()}
        np.savez_compressed(temporary / "frame_history.npz", **history)
        rng = {name: value for name, value in snapshot.items() if name not in {"env_state", "obs_history", "policy_evidence", "state_schema"}}
        with (temporary / "rng_state.pkl").open("wb") as handle:
            pickle.dump(rng, handle, protocol=pickle.HIGHEST_PROTOCOL)
        (temporary / "policy-evidence.json").write_text(json.dumps(snapshot["policy_evidence"], indent=2, sort_keys=True) + "\n")
        (temporary / "source-record.json").write_text(json.dumps(source, indent=2, sort_keys=True, default=str) + "\n")
        (temporary / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n")
        (temporary / "ep_meta.json").write_text(str(snapshot["env_state"].get("ep_meta", "{}")))
        names = ["model.xml", "sim_state.npy", "frame_history.npz", "rng_state.pkl", "policy-evidence.json", "source-record.json", "provenance.json", "ep_meta.json"]
        files = {name: {"bytes": (temporary / name).stat().st_size, "sha256": sha256(temporary / name)} for name in names}
        metadata = {
            "complete_snapshot": False,
            "source_id": source["source_id"],
            "stratum": source["stratum"],
            "task": provenance["task"],
            "cell": provenance["cell"],
            "environment_seed": provenance["environment_seed"],
            "state_schema": schema,
            "files": files,
            "camera_independent": True,
            "env_reset_calls_after_constructor": 0,
            "env_step_calls": 0,
            "route_outcome_reads": 0,
        }
        metadata["snapshot_hash"] = canonical_hash(metadata)
        (temporary / "snapshot-meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        temporary.rename(final)
        return final, metadata
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def write_source_snapshot_in_group(
    directory: Path,
    source: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an incomplete source child inside an attempt-scoped group.

    The caller must validate every E/D/A child before promoting the group.
    """
    if directory.exists():
        raise FileExistsError(f"snapshot attempt child already exists: {directory}")
    directory.mkdir(parents=True)
    schema = paired_state_schema(snapshot)
    (directory / "model.xml").write_text(str(snapshot["env_state"]["model"]))
    np.save(directory / "sim_state.npy", np.asarray(snapshot["env_state"]["states"]))
    history_frame_counts = {key: len(values) for key, values in snapshot["obs_history"].items()}
    if not history_frame_counts or set(history_frame_counts.values()) != {10}:
        raise ValueError(f"snapshot history is not exactly 10 frames: {history_frame_counts}")
    history = {key: np.concatenate(list(values), axis=0) for key, values in snapshot["obs_history"].items()}
    np.savez_compressed(directory / "frame_history.npz", **history)
    rng = {name: value for name, value in snapshot.items() if name not in {"env_state", "obs_history", "policy_evidence", "state_schema"}}
    with (directory / "rng_state.pkl").open("wb") as handle:
        pickle.dump(rng, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (directory / "policy-evidence.json").write_text(json.dumps(snapshot["policy_evidence"], indent=2, sort_keys=True) + "\n")
    (directory / "source-record.json").write_text(json.dumps(source, indent=2, sort_keys=True, default=str) + "\n")
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n")
    (directory / "ep_meta.json").write_text(str(snapshot["env_state"].get("ep_meta", "{}")))
    names = ["model.xml", "sim_state.npy", "frame_history.npz", "rng_state.pkl", "policy-evidence.json", "source-record.json", "provenance.json", "ep_meta.json"]
    files = {name: {"bytes": (directory / name).stat().st_size, "sha256": sha256(directory / name)} for name in names}
    metadata = {
        "complete_snapshot": False,
        "source_id": source["source_id"],
        "stratum": source["stratum"],
        "task": provenance["task"],
        "cell": provenance["cell"],
        "environment_seed": provenance["environment_seed"],
        "state_schema": schema,
        "history_frame_counts": history_frame_counts,
        "files": files,
        "camera_independent": True,
        "env_reset_calls_after_constructor": 0,
        "env_step_calls": 0,
        "route_outcome_reads": 0,
    }
    metadata["snapshot_hash"] = canonical_hash(metadata)
    (directory / "snapshot-meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def promote_validated_snapshot_group(
    temporary_group: Path,
    canonical_group: Path,
    validation_receipts: Mapping[str, Path],
) -> dict[str, Any]:
    """Mark all children complete and atomically promote one E/D/A family."""
    if canonical_group.exists():
        raise FileExistsError(f"canonical snapshot group already exists: {canonical_group}")
    children = sorted(path for path in temporary_group.iterdir() if path.is_dir() and (path / "snapshot-meta.json").exists())
    if len(children) != 3 or set(validation_receipts) != {path.name for path in children}:
        raise ValueError("validated snapshot group must contain exactly three E/D/A children")
    receipt_hashes: dict[str, str] = {}
    child_hashes: dict[str, str] = {}
    for child in children:
        receipt = validation_receipts[child.name]
        payload = json.loads(receipt.read_text())
        if payload.get("status") != "pass":
            raise ValueError(f"validator did not pass for {child.name}")
        receipt_hashes[child.name] = sha256(receipt)
        meta_path = child / "snapshot-meta.json"
        meta = json.loads(meta_path.read_text())
        meta["complete_snapshot"] = True
        meta["roundtrip_receipt_sha256"] = receipt_hashes[child.name]
        meta["snapshot_hash"] = canonical_hash({key: value for key, value in meta.items() if key != "snapshot_hash"})
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        child_hashes[child.name] = meta["snapshot_hash"]
    aggregate = {
        "complete_snapshot": True,
        "children": child_hashes,
        "roundtrip_receipts": receipt_hashes,
    }
    aggregate["aggregate_hash"] = canonical_hash(aggregate)
    (temporary_group / "group-receipt.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    canonical_group.parent.mkdir(parents=True, exist_ok=True)
    temporary_group.rename(canonical_group)
    return aggregate


def expansion_reuse_members(
    candidate_records: list[Mapping[str, Any]],
    selected_cells: Mapping[str, list[int]],
    accepted_candidate_keys: set[tuple[str, int, int]],
) -> set[tuple[str, int, int]]:
    """Filter pre-audited reuse candidates against the v1.4 frozen schedule."""
    members = set()
    for record in candidate_records:
        key = (str(record["task"]), int(record["cell"]), int(record["environment_seed"]))
        if key not in accepted_candidate_keys:
            continue
        if key[1] not in selected_cells.get(key[0], []) or key[2] not in range(3, 6):
            continue
        if not record.get("dimension_closed") or not record.get("eligible_for_final_matching"):
            continue
        paths = record.get("source_paths", [])
        if len(paths) != 3 or not all(Path(path).is_dir() for path in paths):
            continue
        members.add(key)
    return members
