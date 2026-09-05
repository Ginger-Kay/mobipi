#!/usr/bin/env python3
"""Fresh-process paired snapshot validator; never constructs or resets a task env."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    import mujoco

    meta = json.loads((args.snapshot / "snapshot-meta.json").read_text())
    model = mujoco.MjModel.from_xml_string((args.snapshot / "model.xml").read_text())
    state = np.load(args.snapshot / "sim_state.npy", allow_pickle=False)
    expected = 1 + model.nq + model.nv
    if len(state) != expected:
        raise ValueError(f"state length {len(state)} != {expected}")
    data = mujoco.MjData(model)
    data.time = state[0]
    data.qpos[:] = state[1 : 1 + model.nq]
    data.qvel[:] = state[1 + model.nq : expected]
    mujoco.mj_forward(model, data)
    restored = np.concatenate([[data.time], np.asarray(data.qpos), np.asarray(data.qvel)])
    if not np.array_equal(restored, state):
        raise ValueError("round-trip state is not exact")
    with np.load(args.snapshot / "frame_history.npz", allow_pickle=False) as history:
        if not history.files or not all(np.asarray(history[key]).size for key in history.files):
            raise ValueError("frame history is empty")
    if not meta.get("history_frame_counts") or set(meta["history_frame_counts"].values()) != {10}:
        raise ValueError("frame history does not contain exactly 10 frames per key")
    rng = pickle.loads((args.snapshot / "rng_state.pkl").read_bytes())
    if not isinstance(rng, dict) or "python_rng" not in rng or "numpy_rng" not in rng:
        raise ValueError("RNG state is incomplete")
    hashes = {name: sha256(args.snapshot / name) for name in meta["files"]}
    if hashes != {name: row["sha256"] for name, row in meta["files"].items()}:
        raise ValueError("snapshot file checksum mismatch")
    receipt = {
        "status": "pass", "snapshot": str(args.snapshot), "nq": model.nq, "nv": model.nv,
        "expected_state_length": expected, "actual_state_length": len(state), "roundtrip_exact": True,
        "file_hashes": hashes, "env_reset_calls": 0, "env_step_calls": 0, "route_outcome_reads": 0,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
