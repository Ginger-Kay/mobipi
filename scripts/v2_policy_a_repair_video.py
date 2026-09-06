#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path("/share/jhk/MobiWAM")
CODE = ROOT / "Mobipi"
V1 = ROOT / "artifacts/MMWAM-OBC-002/v1-source-video-v1.0.1/20260906T060000Z-v1-video-compat-v1.0.1"
TRANSACTIONS = {
    "CloseDrawer": {
        "qualification": ROOT / "artifacts/MMWAM-OBC-001/runs/20260831T182801Z-pilot-r2-close-drawer-a1/collection/transactions/source-000015.json",
        "display": None,
    },
    "CloseSingleDoor": {
        "qualification": ROOT / "artifacts/MMWAM-OBC-001/runs/20260831T182800Z-pilot-r2-close-single-door-a1/collection/transactions/source-000027.json",
        "display": ROOT / "artifacts/MMWAM-OBC-001/runs/20260831T182800Z-pilot-r2-close-single-door-a1/collection/transactions/source-000019.json",
    },
}


def load_v1():
    path = CODE / "scripts/v1_source_video_run.py"
    spec = importlib.util.spec_from_file_location("v1_source_video_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def code_commit() -> str:
    return subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()


def source_spec(task: str, role: str) -> dict:
    transaction = TRANSACTIONS[task][role]
    if transaction is None:
        freeze = json.loads((V1 / "source-selection-freeze-v1.0.json").read_text())
        row = freeze["selected"][task]["primary"]
        return {
            "source_id": row["source_id"], "snapshot_path": row["snapshot_path"],
            "task_id": task, "layout_id": row["cell"],
            "environment_seed": row["environment_seed"], "transaction": None,
        }
    payload = json.loads(transaction.read_text())
    row = payload["source_state"]
    if row["split"] not in ("train",):
        raise RuntimeError("V2 source must be an existing development train source")
    return {
        "source_id": row["source_state_id"], "snapshot_path": row["snapshot_path"],
        "task_id": task, "layout_id": row["layout_id"],
        "environment_seed": row["environment_seed"], "transaction": str(transaction),
    }


def make_adapter(root: Path, task: str, role: str, save_video: bool):
    from mobiwam.adapters.mobipi import create_adapter

    v1 = load_v1()
    camera = json.loads((V1 / "camera-freeze-v1.0.json").read_text())["cameras"][task]
    config = v1.adapter_config(task, root, code_commit(), camera)
    config["output_root"] = str(root / role / task)
    config["save_video"] = bool(save_video)
    config["external_camera_width"] = 1920
    config["external_camera_height"] = 1080
    spec = source_spec(task, role)
    adapter = create_adapter(output_root=root / role / task, config=config)
    snapshot = adapter.load_frozen_source_state(
        Path(spec["snapshot_path"]), source_id=spec["source_id"], task_id=task,
        layout_id=int(spec["layout_id"]), environment_seed=int(spec["environment_seed"]),
    )
    return adapter, snapshot, spec


def metrics(record) -> dict:
    with np.load(record.state_trace_path, allow_pickle=False) as trace:
        base = np.asarray(trace["base_positions"], float)
        contacts = np.asarray(trace["manipulation_contacts"], bool)
        progress = np.asarray(trace["fixture_progress"], float)
        manifold = np.asarray(trace["manifold_errors_m"], float)
        solver = [str(x) for x in trace["solver_status"]]
    path = float(np.linalg.norm(np.diff(base, axis=0), axis=1).sum()) if len(base) > 1 else 0.0
    net = float(np.linalg.norm(base[-1] - base[0])) if len(base) > 1 else 0.0
    return {
        "base_net_m": net, "base_path_m": path,
        "contact_fraction": float(contacts.mean()) if len(contacts) else 0.0,
        "contact_steps": int(contacts.sum()),
        "joint_progress_monotonic_fraction": float(np.mean(np.diff(progress) >= -1e-4)) if len(progress) > 1 else 0.0,
        "manifold_error_p95_m": float(np.percentile(manifold, 95)) if len(manifold) else None,
        "solver_status": sorted(set(solver)), "step_count": int(len(base)),
    }


def qualify(root: Path, task: str, version: int) -> None:
    adapter, snapshot, spec = make_adapter(root, task, "qualification", False)
    try:
        restore = adapter.restore_source_state(snapshot)
        if not restore.passed:
            raise RuntimeError("qualification restore failed")
        record = adapter.execute_articulation_a(
            snapshot, policy_seed=202609060 + version, route_seed=202609070 + version,
            repeat_index=version - 1, candidate_id=f"a_articulation_v{version}",
            base_target_net_m=0.22 if version == 1 else 0.205,
            travel_cap_m=0.46, stable_contact_steps=3,
        )
        row = asdict(record); row["runtime_metrics"] = metrics(record)
        passed = bool(
            row["success"] and row["candidate_params"]["stable_contact_established"]
            and row["runtime_metrics"]["contact_steps"] >= 3 and not row["collision"]
            and not row["candidate_params"]["action_saturated"]
            and row["runtime_metrics"]["joint_progress_monotonic_fraction"] >= 0.90
            and row["runtime_metrics"]["base_path_m"] <= 0.465
        )
        write(root / "qualification" / task / f"a-v{version}-record.json", row)
        write(root / "qualification" / task / f"a-v{version}-receipt.json", {
            "task": task, "version": version, "source": spec, "passed": passed,
            "success": row["success"], "failure": row["failure_type"],
            "collision": row["collision"], "metrics": row["runtime_metrics"],
            "candidate_params": row["candidate_params"],
        })
        if not passed:
            raise SystemExit(3)
    finally:
        if adapter.env is not None:
            close = getattr(adapter.env, "close", None)
            if callable(close): close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["qualify"])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--task", choices=list(TRANSACTIONS), required=True)
    parser.add_argument("--version", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    qualify(args.artifact_root, args.task, args.version)


if __name__ == "__main__":
    main()
