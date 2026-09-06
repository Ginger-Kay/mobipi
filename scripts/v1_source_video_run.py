#!/usr/bin/env python3
"""MMWAM-OBC-002 V1-only source/video feasibility runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mobiwam.adapters.mobipi import create_adapter


CODE = Path("/share/jhk/MobiWAM/Mobipi")
PARENT = Path("/share/jhk/MobiWAM/artifacts/MMWAM-OBC-002/planner-min-v1.5.1/20260905T174401Z-planner-min-v1.5.1")
CHECKPOINTS = {
    "CloseDrawer": Path("/share/jhk/MobiWAM/checkpoints/MMWAM-OBC-001/robocasa/bc_xfmr/04-12-CloseDrawer/seed_1_CloseDrawer_mg-300/20250413055056/models/model_epoch_1000.pth"),
    "CloseSingleDoor": Path("/share/jhk/MobiWAM/checkpoints/inherited/chensiyu-20260830/robocasa/bc_xfmr/04-12-CloseSingleDoor/seed_1_CloseSingleDoor_mg-300/20250413055045/models/model_epoch_1000.pth"),
}
CHECKPOINT_ROOTS = {
    "CloseDrawer": Path("/share/jhk/MobiWAM/checkpoints/MMWAM-OBC-001"),
    "CloseSingleDoor": Path("/share/jhk/MobiWAM/checkpoints/inherited/chensiyu-20260830"),
}
ROUTE_ORDERS = {
    "CloseDrawer": ["E", "D", "A", "A0"],
    "CloseSingleDoor": ["A0", "A", "D", "E"],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def append(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(CODE), *args], text=True).strip()


def primary_rows(row: Mapping[str, Any]) -> dict[str, Any]:
    proposals = {proposal["candidate_id"]: proposal for proposal in row["proposals"]}
    return {family: proposals[row["primary"][family]] for family in "EDA"}


def strict_complete(row: Mapping[str, Any]) -> bool:
    try:
        primary = primary_rows(row)
    except (KeyError, TypeError):
        return False
    return bool(
        row.get("stage_receipt", {}).get("precontact")
        and all(primary[family].get("hard_valid") for family in "EDA")
    )


def ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    primary = primary_rows(row)
    stratum = {"A-required": 0, "D-required": 1, "E-required": 2, "E-compatible": 2}[str(row["stratum"])]
    return (
        stratum,
        -min(float(primary[family]["minimum_continuous_clearance_m"]) for family in "EDA"),
        -min(float(primary[family]["minimum_manipulability_or_joint_margin"]) for family in "EDA"),
        -min(float(primary[family]["minimum_policy_view_compatibility"]) for family in "EDA"),
        sum(float(primary[family]["total_planned_time_s"]) for family in "EDA"),
        str(row["source_id"]),
    )


def route_union_xy(row: Mapping[str, Any]) -> np.ndarray:
    primary = primary_rows(row)
    points: list[list[float]] = []
    for family in "EDA":
        proposal = primary[family]
        trajectory = proposal.get("planner_result", {}).get("trajectory", {})
        points.extend(trajectory.get("base_xy_m", []))
        points.extend(proposal.get("navigation_path_xy_m", []))
        points.extend(proposal.get("guide_path_xy_m", []))
        points.extend([point[:2] for point in row.get("task_space_chain", {}).get("approach_points_world", [])])
        points.extend([point[:2] for point in row.get("task_space_chain", {}).get("precontact_points_world", [])])
        points.extend([point[:2] for point in row.get("task_space_chain", {}).get("manipulation_points_world", [])])
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"source {row['source_id']} has no 2-D route envelope")
    return value


def camera_for(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    union = np.vstack([route_union_xy(row) for row in rows])
    lower = union.min(axis=0) - 0.35
    upper = union.max(axis=0) + 0.35
    center = (lower + upper) / 2.0
    fovy = 60.0
    tangent = math.tan(math.radians(fovy) / 2.0)
    needed_height = max(
        (upper[1] - lower[1]) / (2.0 * 0.90 * tangent),
        (upper[0] - lower[0]) / (2.0 * 0.90 * tangent * 16.0 / 9.0),
        2.5,
    )
    height = min(3.8, needed_height)
    visible_half_y = height * tangent * 0.90
    visible_half_x = visible_half_y * 16.0 / 9.0
    coverage = bool(
        np.all(lower >= center - [visible_half_x, visible_half_y])
        and np.all(upper <= center + [visible_half_x, visible_half_y])
    )
    focal_px = 0.5 * 1080.0 / tangent
    projected_base_diameter_px = 0.50 * focal_px / height
    return {
        "camera_type": "external_world_frame_fixed",
        "name": "freeview",
        "native_width": 1920,
        "native_height": 1080,
        "upscale_ratio": 1.0,
        "position_world": [float(center[0]), float(center[1]), float(height)],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "fovy_deg": fovy,
        "envelope_lower_xy_m": lower.tolist(),
        "envelope_upper_xy_m": upper.tolist(),
        "fit_visible_half_xy_m": [float(visible_half_x), float(visible_half_y)],
        "estimated_base_diameter_px": float(projected_base_diameter_px),
        "analytic_coverage_passed": coverage,
        "selection_inputs": [row["source_id"] for row in rows],
        "outcome_blind": True,
    }


def freeze(root: Path, code_commit: str) -> None:
    if git("rev-parse", "HEAD") != code_commit:
        raise RuntimeError("code commit mismatch at freeze")
    if git("status", "--porcelain"):
        raise RuntimeError("code tree must be clean before freeze")
    rows = [json.loads(line) for line in (PARENT / "candidate-plan-records.jsonl").read_text().splitlines() if line.strip()]
    strict = [row for row in rows if strict_complete(row)]
    if len(strict) != 31:
        raise RuntimeError(f"authoritative strict count changed: {len(strict)}")
    selected: dict[str, Any] = {}
    cameras: dict[str, Any] = {}
    candidate_audit: dict[str, Any] = {}
    for task in ROUTE_ORDERS:
        eligible = sorted([row for row in strict if row["task"] == task], key=ranking_key)
        if not eligible:
            raise RuntimeError(f"no strict-complete source for {task}")
        primary = eligible[0]
        reserves = [row for row in eligible[1:] if row["cell"] == primary["cell"]][:2]
        camera = camera_for([primary, *reserves])
        if not camera["analytic_coverage_passed"]:
            raise RuntimeError(f"no fixed analytic camera coverage for {task}")
        selected[task] = {
            "primary": primary,
            "ordered_reserves": reserves,
            "ranking_key": list(ranking_key(primary)),
            "selection_status": "primary_frozen_pending_no_actuation_probe",
        }
        cameras[task] = camera
        candidate_audit[task] = [
            {
                "source_id": row["source_id"],
                "strict_complete": strict_complete(row),
                "ranking_key": list(ranking_key(row)),
                "selected_role": (
                    "primary" if row is primary else "reserve" if row in reserves else "not_selected"
                ),
                "exclusion_reason": None if row is primary or row in reserves else "lower_outcome_blind_rank_or_different_cell_than_primary_reserve_set",
            }
            for row in eligible
        ]
    source_freeze = {
        "schema_version": "source-selection-freeze-v1.0",
        "frozen_at": now(),
        "code_commit": code_commit,
        "parent_candidate_ledger": str(PARENT / "candidate-plan-records.jsonl"),
        "parent_candidate_ledger_sha256": sha(PARENT / "candidate-plan-records.jsonl"),
        "authoritative_strict_complete": 31,
        "ranking_rule": ["A-required", "D-required", "E-compatible", "larger route-union minimum continuous clearance", "larger minimum manipulability or joint margin", "larger minimum policy-view compatibility", "shorter planned time", "source ID ascending"],
        "outcome_fields_read": [],
        "selected": selected,
        "candidate_audit": candidate_audit,
    }
    write(root / "source-selection-freeze-v1.0.json", source_freeze)
    write(root / "route-order-freeze-v1.0.json", {"schema_version": "route-order-freeze-v1.0", "frozen_at": now(), "orders": ROUTE_ORDERS, "planned_logical_routes": 8, "route_attempt_cap": 2, "global_actual_attempt_cap": 16})
    write(root / "camera-freeze-v1.0.json", {"schema_version": "camera-freeze-v1.0", "frozen_at": now(), "cameras": cameras, "camera_motion": "forbidden", "native_resolution": [1920, 1080]})
    config = {
        "schema_version": "v1-source-video-runtime-config-v1.0",
        "frozen_at": now(),
        "code_commit": code_commit,
        "policy_seed_by_task": {"CloseDrawer": 202609060, "CloseSingleDoor": 202609061},
        "route_seed_by_task": {"CloseDrawer": 202609062, "CloseSingleDoor": 202609063},
        "horizon": 500,
        "video_fps": 20,
        "external_camera_native": [1920, 1080],
        "E_A0_equivalence_tolerance": 1e-6,
        "base_lock_net_max_m": 0.02,
        "D_A_net_min_m": 0.30,
        "D_pre_manipulation_path_fraction_min": 0.90,
        "A_min_chunks": 3,
        "collision_rule": "no newly observed mobile-base collision",
    }
    write(root / "runtime-config-v1.0.json", config)
    hashes = {name: sha(root / name) for name in ("source-selection-freeze-v1.0.json", "route-order-freeze-v1.0.json", "camera-freeze-v1.0.json", "runtime-config-v1.0.json")}
    write(root / "freeze-hashes-v1.0.json", hashes)


def adapter_config(task: str, root: Path, code_commit: str, camera: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = CHECKPOINTS[task]
    return {
        "mobipi_repo": str(CODE),
        "mobipi_upstream_commit": "19b130b8ada3f7e029918449c12d433e9e629ca1",
        "robocasa_commit": "426bc4dbbadec923d37752b012ba1152d25f8716",
        "checkpoint_root": str(CHECKPOINT_ROOTS[task]),
        "data_root": "/share/jhk/MobiWAM/data/inherited/chensiyu-20260830",
        "env_name": task,
        "policy_name": "bc_xfmr",
        "checkpoint_seed": 1,
        "dataset_name": "mg-300",
        "policy_checkpoint_path": str(checkpoint),
        "policy_checkpoint_hash": sha(checkpoint),
        "code_commit": code_commit,
        "output_root": str(root),
        "horizon": 500,
        "history_length": 10,
        "legacy_navigation": False,
        "settle_linear_threshold_mps": 0.005,
        "settle_angular_threshold_radps": 0.02,
        "settle_max_steps": 200,
        "save_video": True,
        "video_fps": 20,
        "video_observation_key": "robot0_agentview_right_image",
        "external_world_camera": True,
        "external_camera_name": camera["name"],
        "external_camera_width": camera["native_width"],
        "external_camera_height": camera["native_height"],
        "external_camera_position_world": camera["position_world"],
        "external_camera_quaternion": camera["quaternion_wxyz"],
        "external_camera_fovy_deg": camera["fovy_deg"],
        "render_gpu_device_id": 0,
        "strict_torch_determinism": True,
        "schedule_checksum": sha(root / "route-order-freeze-v1.0.json"),
    }


def load_frozen(root: Path, task: str, *, save_config: bool = True):
    freeze_data = json.loads((root / "source-selection-freeze-v1.0.json").read_text())
    camera = json.loads((root / "camera-freeze-v1.0.json").read_text())["cameras"][task]
    selected = freeze_data["selected"][task]["primary"]
    code_commit = str(freeze_data["code_commit"])
    config = adapter_config(task, root / "workers" / task, code_commit, camera)
    if save_config:
        write(root / "workers" / task / "adapter-config.json", config)
    adapter = create_adapter(output_root=root / "workers" / task, config=config)
    snapshot = adapter.load_frozen_source_state(
        Path(selected["snapshot_path"]),
        source_id=selected["source_id"],
        task_id=task,
        layout_id=int(selected["cell"]),
        environment_seed=int(selected["environment_seed"]),
    )
    return adapter, snapshot, selected, camera


def probe(root: Path, task: str) -> None:
    from PIL import Image

    started = now()
    adapter = None
    try:
        adapter, snapshot, selected, camera = load_frozen(root, task)
        restore = adapter.restore_source_state(snapshot)
        if not restore.passed:
            raise RuntimeError("frozen source restore mismatch")
        frame = adapter._capture_frame(adapter._stacked_observation())
        frame_path = root / "workers" / task / "no-actuation-camera-probe.png"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame, mode="RGB").save(frame_path)
        receipt = {
            "task": task,
            "source_id": selected["source_id"],
            "started_at": started,
            "ended_at": now(),
            "status": "pass",
            "restore": asdict(restore),
            "parent_snapshot_hash": selected["snapshot_hash"],
            "runtime_snapshot_hash": snapshot.record.snapshot_hash,
            "runtime_observation_hash": snapshot.record.observation_hash,
            "camera": camera,
            "frame_path": str(frame_path),
            "frame_bytes": frame_path.stat().st_size,
            "frame_sha256": sha(frame_path),
            "frame_shape": list(frame.shape),
            "env_step_calls": 0,
            "outcome_reads": 0,
        }
        write(root / "workers" / task / "no-actuation-probe-receipt.json", receipt)
    except Exception as error:
        write(root / "workers" / task / "no-actuation-probe-receipt.json", {"task": task, "started_at": started, "ended_at": now(), "status": "mechanical_failure", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "env_step_calls": 0, "outcome_reads": 0})
        raise
    finally:
        if adapter is not None and adapter.env is not None:
            closer = getattr(adapter.env, "close", None)
            if callable(closer):
                closer()


def trace_metrics(record: Any) -> dict[str, Any]:
    with np.load(record.state_trace_path, allow_pickle=False) as trace:
        base = np.asarray(trace["base_positions"], dtype=float)
        phases = [str(value) for value in trace["phases"]]
        observation_hashes = [str(value) for value in trace["observation_hashes"]]
        desired = np.asarray(trace["desired_eef_poses"], dtype=float)
        eef = np.asarray(trace["eef_poses"], dtype=float)
    if len(base):
        initial = np.asarray(record.candidate_params.get("initial_base_xy_m", base[0]), dtype=float)
        net = float(np.linalg.norm(base[-1] - initial))
        segments = np.diff(np.vstack([initial, base]), axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        path = float(lengths.sum())
        manipulation_mask = np.asarray([phase in {"MANIPULATE", "ASSIST_ACTIVE"} for phase in phases])
        manipulation_path = float(lengths[manipulation_mask].sum()) if len(lengths) == len(manipulation_mask) else 0.0
    else:
        net = path = manipulation_path = 0.0
    eef_error = float(np.percentile(np.linalg.norm(eef[:, :3, 3] - desired[:, :3, 3], axis=1), 95)) if len(eef) and len(desired) == len(eef) else None
    return {"actual_base_net_m": net, "actual_base_path_m": path, "manipulation_base_path_m": manipulation_path, "pre_manipulation_path_fraction": (path - manipulation_path) / path if path > 1e-12 else 0.0, "step_count": int(len(base)), "phases": phases, "observation_hashes": observation_hashes, "eef_intent_position_error_p95_m": eef_error}


def run_task(root: Path, task: str) -> None:
    adapter, snapshot, selected, _camera = load_frozen(root, task)
    config = json.loads((root / "runtime-config-v1.0.json").read_text())
    primary = primary_rows(selected)
    policy_seed = int(config["policy_seed_by_task"][task])
    route_seed = int(config["route_seed_by_task"][task])
    order = ROUTE_ORDERS[task]
    worker_root = root / "workers" / task
    restore0 = adapter.restore_source_state(snapshot)
    if not restore0.passed:
        raise RuntimeError("initial source restore failed")
    nominal = adapter.sample_nominal_policy(snapshot, policy_seed)
    results = []
    try:
        for repeat_index, route in enumerate(order):
            restore = adapter.restore_source_state(snapshot)
            if not restore.passed:
                raise RuntimeError(f"restore failed before {route}")
            initial_base = adapter._origin_pose()[:2, 3].copy()
            attempt_id = f"{task}-{route}-attempt-1"
            append(root / "route-attempt-ledger.jsonl", {"attempt_id": attempt_id, "task": task, "route": route, "attempt": 1, "started_at": now(), "status": "started_env_step_attempt", "source_id": selected["source_id"], "restore": asdict(restore)})
            if route == "E":
                record = adapter.execute_e(snapshot, nominal, policy_seed=policy_seed, route_seed=route_seed, repeat_index=repeat_index)
            elif route == "A0":
                record = adapter.execute_a0(snapshot, nominal, policy_seed=policy_seed, route_seed=route_seed, repeat_index=repeat_index)
            elif route == "D":
                proposal = primary["D"]
                params = {"planned_path_world_xy_m": proposal["navigation_path_xy_m"], "steps_per_waypoint": 50, "position_tolerance_m": 0.02, "command_gain": 1.0}
                record = adapter.execute_d(snapshot, policy_seed=policy_seed, route_seed=route_seed, repeat_index=repeat_index, candidate_id=proposal["candidate_id"], candidate_params=params)
            elif route == "A":
                proposal = primary["A"]
                base_path = proposal["planner_result"]["trajectory"]["base_xy_m"]
                chunks = max(3, math.ceil(len(base_path) / len(nominal.chunk)))
                params = {"planned_base_path_world_xy_m": base_path, "persistent_chunks": chunks, "total_travel_cap_m": 0.46, "parallel_cap_m": 0.45}
                record = adapter.execute_a(snapshot, nominal, policy_seed=policy_seed, route_seed=route_seed, repeat_index=repeat_index, candidate_id=proposal["candidate_id"], candidate_params=params)
            else:
                raise AssertionError(route)
            row = asdict(record)
            row["route_label"] = route
            row["parent_snapshot_hash"] = selected["snapshot_hash"]
            row["initial_restore"] = asdict(restore)
            row["candidate_params"] = {**row["candidate_params"], "initial_base_xy_m": initial_base.tolist()}
            row["runtime_metrics"] = trace_metrics(type("RecordView", (), {**row})())
            path = worker_root / "route-records" / f"{route}.json"
            write(path, row)
            append(root / "route-attempt-ledger.jsonl", {"attempt_id": attempt_id, "task": task, "route": route, "attempt": 1, "ended_at": now(), "status": "semantic_complete", "record_path": str(path), "record_sha256": sha(path), "success": row["success"], "collision": row["collision"], "failure_type": row["failure_type"]})
            results.append(row)
        by_route = {row["route_label"]: row for row in results}
        e = by_route["E"]
        a0 = by_route["A0"]
        with np.load(e["action_trace_path"], allow_pickle=False) as e_action, np.load(a0["action_trace_path"], allow_pickle=False) as a0_action:
            action_error = float(np.max(np.abs(e_action["actions"] - a0_action["actions"]))) if e_action["actions"].shape == a0_action["actions"].shape else float("inf")
        with np.load(e["state_trace_path"], allow_pickle=False) as e_state, np.load(a0["state_trace_path"], allow_pickle=False) as a0_state:
            state_error = float(np.max(np.abs(e_state["states"] - a0_state["states"]))) if e_state["states"].shape == a0_state["states"].shape else float("inf")
            observation_equal = bool(np.array_equal(e_state["observation_hashes"], a0_state["observation_hashes"]))
        audit = {
            "task": task,
            "source_id": selected["source_id"],
            "route_order": order,
            "route_count": len(results),
            "initial_restore_hashes_identical": len({row["initial_restore"]["snapshot_hash"] for row in results}) == 1 and len({row["initial_restore"]["observation_hash"] for row in results}) == 1 and len({row["initial_restore"]["controller_hash"] for row in results}) == 1 and len({row["initial_restore"]["contact_hash"] for row in results}) == 1,
            "E_A0_equivalence": {"action_max_abs_error": action_error, "state_max_abs_error": state_error, "observation_hash_sequences_equal": observation_equal, "passed": action_error <= 1e-6 and state_error <= 1e-6 and observation_equal},
            "route_semantics": {
                "E": e["runtime_metrics"]["actual_base_net_m"] <= 0.02,
                "A0": a0["runtime_metrics"]["actual_base_net_m"] <= 0.02,
                "D": by_route["D"]["runtime_metrics"]["actual_base_net_m"] >= 0.30 and by_route["D"]["runtime_metrics"]["pre_manipulation_path_fraction"] >= 0.90 and bool(by_route["D"]["candidate_params"].get("post_dock_policy_ready")),
                "A": by_route["A"]["runtime_metrics"]["actual_base_net_m"] >= 0.30 and int(by_route["A"]["candidate_params"].get("assist_chunk_count", 0)) >= 3,
            },
            "no_collision": all(not row["collision"] for row in results),
            "completed_at": now(),
        }
        audit["semantic_gate_passed"] = bool(audit["initial_restore_hashes_identical"] and audit["E_A0_equivalence"]["passed"] and all(audit["route_semantics"].values()) and audit["no_collision"])
        write(worker_root / "task-semantic-audit.json", audit)
        write(worker_root / "task-completion.json", {"task": task, "status": "completed", "source_id": selected["source_id"], "route_records": 4, "raw_external_videos": 4, "raw_policy_videos": 4, "semantic_gate_passed": audit["semantic_gate_passed"], "completed_at": now()})
    finally:
        if adapter.env is not None:
            closer = getattr(adapter.env, "close", None)
            if callable(closer):
                closer()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "probe", "run-task"])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--code-commit")
    parser.add_argument("--task", choices=list(ROUTE_ORDERS))
    args = parser.parse_args()
    if args.command == "freeze":
        if not args.code_commit:
            parser.error("freeze requires --code-commit")
        freeze(args.artifact_root, args.code_commit)
    elif args.command == "probe":
        if not args.task:
            parser.error("probe requires --task")
        probe(args.artifact_root, args.task)
    else:
        if not args.task:
            parser.error("run-task requires --task")
        run_task(args.artifact_root, args.task)


if __name__ == "__main__":
    main()
