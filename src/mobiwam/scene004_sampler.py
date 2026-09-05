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
