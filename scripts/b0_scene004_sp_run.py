#!/usr/bin/env python3
"""Deadline-minimum SCENE-004 v1.4 U2 runner."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from b0_scene004_u2_scan import (
    CELLS,
    TASKS,
    camera_grid,
    cell_rank,
    close_env,
    compile_reset_call,
    create_env,
    envelope_points,
    load_task,
    topdown_overlay,
)
from mobiwam.b0_scene_compiler import camera_projection_metrics
from mobiwam.scene004 import independent_reason_vector, select_cell_camera
from mobiwam.scene004_renderer import camera_payload_hash, load_snapshot_sim
from mobiwam.scene004_sampler import (
    expansion_reuse_members,
    promote_validated_snapshot_group,
    write_source_snapshot_in_group,
)
from robosuite.utils.camera_utils import get_camera_transform_matrix


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validator_command(repo: Path, snapshot: Path, receipt: Path) -> list[str]:
    return [
        sys.executable,
        str(repo / "scripts/b0_scene004_snapshot_validate.py"),
        str(snapshot),
        "--receipt",
        str(receipt),
    ]


def persist_and_validate_group(
    root: Path,
    repo: Path,
    group: dict[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
    call_index: int,
    attempt: int,
    code_commit: str,
) -> dict[str, Any]:
    provenance = {
        "task": group["task"], "cell": group["cell"], "environment_seed": group["environment_seed"],
        "call_index": call_index, "attempt": attempt, "state_sampling_call": group["state_sampling_call"],
        "constructor_count": 1, "explicit_reset_count": 0, "code_commit": code_commit,
        "captured_at": utcnow(), "env_step_calls": 0, "route_outcome_reads": 0,
        "call_path": {
            "sampling": "create_env_from_checkpoint_metadata constructor",
            "wrapper": "initialize_constructor_frame_stack -> update_obs(reset=True) -> 10-frame history",
            "inner_env_reset": False,
            "wrapper_reset": False,
            "explicit_reset": False,
        },
    }
    stem = f"{group['task']}-l{group['cell']}-seed{group['environment_seed']:02d}"
    attempt_root = root / "snapshot-attempts" / f"newcall{call_index:02d}" / f"attempt{attempt}" / stem
    temporary_group = attempt_root / f".{stem}.tmp"
    canonical_group = root / "canonical-snapshot-groups" / stem
    receipts = {}
    try:
        for source in group["sources"]:
            child = temporary_group / source["source_id"]
            write_source_snapshot_in_group(child, source, snapshots[source["stratum"]], provenance)
        # Validate only after all three children have been materialized.
        for source in group["sources"]:
            child = temporary_group / source["source_id"]
            receipt = child / "roundtrip-receipt.json"
            result = subprocess.run(
                validator_command(repo, child, receipt), capture_output=True, text=True,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
            )
            (child / "roundtrip-validator.log").write_text(result.stdout + result.stderr)
            if result.returncode != 0:
                raise RuntimeError(f"round-trip failed for {source['source_id']}: {result.returncode}")
            receipts[source["source_id"]] = receipt
        aggregate = promote_validated_snapshot_group(temporary_group, canonical_group, receipts)
    except Exception:
        if temporary_group.exists():
            failure_group = attempt_root / f"{stem}.failed"
            if failure_group.exists():
                raise FileExistsError(f"failure attempt path already exists: {failure_group}")
            temporary_group.rename(failure_group)
        raise
    for source in group["sources"]:
        child = canonical_group / source["source_id"]
        metadata = json.loads((child / "snapshot-meta.json").read_text())
        source["snapshot_path"] = str(child)
        source["snapshot_hash"] = metadata["snapshot_hash"]
        source["roundtrip_receipt"] = str(child / "roundtrip-receipt.json")
        source["camera"] = {"passed": None, "camera_hash": None}
        source["reason_vector"]["predicates"]["camera"] = False
        source["reason_vector"] = independent_reason_vector(source["reason_vector"]["predicates"])
    group["camera"] = {"passed": None, "camera_hash": None}
    group["complete_snapshot"] = True
    group["snapshot_aggregate_hash"] = aggregate["aggregate_hash"]
    group["complete_seed_group"] = False
    group["corrected_sampler"] = provenance
    group.pop("envelope_points_xyz", None)
    write_json(root / "group-records" / f"{group['task']}-l{group['cell']}-seed{group['environment_seed']:02d}.json", group)
    return group


def points_for(group: Mapping[str, Any]) -> np.ndarray:
    return envelope_points(list(group["sources"]), group["fixture"])


def evaluate_camera_snapshot(group: Mapping[str, Any], anchor: np.ndarray, pose: Any) -> dict[str, Any]:
    sim, _ = load_snapshot_sim(Path(group["sources"][0]["snapshot_path"]))
    camera_id = sim.model.camera_name2id("freeview")
    sim.model.cam_pos[camera_id] = [*(anchor + np.asarray(pose.center_offset_xy)), pose.height_m]
    sim.model.cam_quat[camera_id] = [1.0, 0.0, 0.0, 0.0]
    sim.model.cam_fovy[camera_id] = pose.fov_deg
    sim.forward()
    transform = get_camera_transform_matrix(sim, "freeview", 1080, 1920)
    metrics = camera_projection_metrics(points_for(group), transform, width=1920, height=1080, border_fraction=0.05)
    focal = 0.5 * 1080 / math.tan(math.radians(pose.fov_deg) / 2.0)
    diameter = 2.0 * float(group["robot_radius_m"]) * focal / pose.height_m
    metrics["base_projected_diameter_px"] = diameter
    metrics["passed"] = bool(metrics["passed"] and diameter >= 120.0)
    return metrics


def reduce_camera(task: str, cell: int, groups: list[dict[str, Any]]) -> dict[str, Any]:
    anchor = np.vstack([points_for(group) for group in groups])[:, :2].mean(axis=0)
    evaluations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pose in camera_grid():
        for group in groups:
            evaluations[pose.camera_id].append(evaluate_camera_snapshot(group, anchor, pose))
    selected = select_cell_camera(f"{task}-l{cell}", evaluations)
    pose = selected["selected"]["pose"]
    payload = {"cell_key": f"{task}-l{cell}", "anchor_xy": anchor.tolist(), "pose": pose}
    payload["camera_hash"] = camera_payload_hash(payload)
    selected["camera_payload"] = payload
    return selected


def render_group(root: Path, repo: Path, group: dict[str, Any], camera: Mapping[str, Any], phase: str) -> dict[str, Any]:
    metric = evaluate_camera_snapshot(
        group,
        np.asarray(camera["camera_payload"]["anchor_xy"]),
        next(pose for pose in camera_grid() if pose.camera_id == camera["camera_payload"]["pose"]["camera_id"]),
    )
    for source in group["sources"]:
        source["camera"] = {"passed": bool(metric["passed"]), "camera_hash": camera["camera_payload"]["camera_hash"]}
        source["reason_vector"]["predicates"]["camera"] = bool(metric["passed"])
        source["reason_vector"] = independent_reason_vector(source["reason_vector"]["predicates"])
    payload = {
        "task": group["task"], "cell": group["cell"], "environment_seed": group["environment_seed"],
        "camera_payload": camera["camera_payload"],
        "sources": [{"source_id": s["source_id"], "stratum": s["stratum"], "snapshot_path": s["snapshot_path"]} for s in group["sources"]],
    }
    stem = f"{group['task']}-l{group['cell']}-seed{group['environment_seed']:02d}"
    payload_path = root / "renderer-payloads" / f"{stem}.json"
    write_json(payload_path, payload)
    output_dir = root / "native-frames" / phase / stem
    receipt = root / "renderer-receipts" / f"{stem}.json"
    log = root / "renderer-logs" / f"{stem}.log"
    receipt.parent.mkdir(parents=True, exist_ok=True); log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(repo / "scripts/b0_scene004_renderer_worker.py"), "--payload", str(payload_path),
               "--output-dir", str(output_dir), "--receipt", str(receipt), "--device-id", os.environ["MUJOCO_EGL_DEVICE_ID"]]
    result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
    log.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"renderer failed for {stem}: {result.returncode}")
    rendered = json.loads(receipt.read_text())
    by_id = {row["source_id"]: row for row in rendered["frames"]}
    for source in group["sources"]:
        row = by_id[source["source_id"]]
        source["camera"].update({"frame_path": row["frame_path"], "frame_sha256": row["frame_sha256"],
                                  "native_width": 1920, "native_height": 1080, "upscale_ratio": 1.0})
    group["camera"] = {"passed": bool(metric["passed"]), "camera_hash": camera["camera_payload"]["camera_hash"],
                       "selected": camera["camera_payload"]["pose"], "anchor_xy": camera["camera_payload"]["anchor_xy"],
                       "minimum_border_fraction": metric["min_border_fraction"],
                       "minimum_base_projected_diameter_px": metric["base_projected_diameter_px"]}
    group["complete_seed_group"] = all(s["reason_vector"]["passed"] for s in group["sources"])
    group["independent_failure_reasons"] = sorted({x for s in group["sources"] for x in s["reason_vector"]["failure_reasons"]})
    overlay = topdown_overlay(group); overlay_path = root / "topdown-overlays" / f"{stem}.png"
    overlay_path.parent.mkdir(parents=True, exist_ok=True); overlay.save(overlay_path)
    group["topdown_overlay"] = {"path": str(overlay_path), "sha256": sha256(overlay_path)}
    return group


def canonicalize_reuse(record: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(record)
    for source in row["sources"]:
        source["camera"] = {"passed": None, "camera_hash": None}
        source["reason_vector"]["predicates"]["camera"] = False
        source["reason_vector"] = independent_reason_vector(source["reason_vector"]["predicates"])
    row["camera"] = {"passed": None, "camera_hash": None}
    row["complete_seed_group"] = False
    row["reuse_source"] = "SCENE-004-v1.1 dimension-closed canonical snapshot"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--parent-screening", type=Path, required=True)
    parser.add_argument("--parent-expansion", type=Path, required=True)
    parser.add_argument("--expansion-candidates", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--new-reset-cap", type=int, required=True)
    args = parser.parse_args(); root = args.artifact_root; repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != args.code_commit:
        raise RuntimeError("code commit mismatch")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("code worktree is dirty")
    reuse = json.loads(args.reuse_manifest.read_text())
    accepted = {(x["task"], x["cell"], x["environment_seed"]) for x in reuse["accepted"]}
    parent_screen = {(r["task"], r["cell"], r["environment_seed"]): r for r in load_jsonl(args.parent_screening)}
    parent_exp_records = load_jsonl(args.parent_expansion)
    parent_exp: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in parent_exp_records:
        parent_exp.setdefault((row["task"], row["cell"], row["environment_seed"]), row)
    candidate_manifest = json.loads(args.expansion_candidates.read_text())
    candidate_keys = {
        (row["task"], int(row["cell"]), int(row["environment_seed"]))
        for row in candidate_manifest["candidates"]
        if row.get("dimension_closed") and row.get("eligible_for_final_matching")
    }
    new_resets = 0; mechanical = 0; screening: list[dict[str, Any]] = []
    reset_ledger = root / "new-reset-ledger.jsonl"
    for task in sorted(TASKS):
        config, model, policy, env_meta, shape_meta = load_task(task)
        for cell in CELLS:
            for seed in range(3):
                key = (task, cell, seed)
                if key in accepted:
                    screening.append(canonicalize_reuse(parent_screen[key])); continue
                attempt = 1
                while True:
                    if new_resets >= args.new_reset_cap: raise RuntimeError("new reset cap exceeded")
                    new_resets += 1; env = None; started = utcnow()
                    try:
                        env = create_env(config, env_meta, shape_meta, cell, seed)
                        group, snapshots = compile_reset_call(task, cell, seed, env, policy, args.scene_root,
                            root / "snapshot-attempts" / f"newcall{new_resets:02d}" / f"attempt{attempt}" /
                            f"{task}-l{cell}-seed{seed:02d}" / "raw-source", perform_explicit_reset=False)
                        group = persist_and_validate_group(root, repo, group, snapshots, new_resets, attempt, args.code_commit)
                        close_env(env); env = None
                        append_jsonl(reset_ledger, {"new_call_index": new_resets, "phase": "screening", "task": task, "cell": cell,
                            "environment_seed": seed, "attempt": attempt, "state_sampling_call": "task_environment_constructor",
                            "explicit_reset_calls": 0, "status": "complete_snapshot_family", "started_at": started, "ended_at": utcnow(),
                            "env_step_calls": 0, "route_outcome_reads": 0})
                        screening.append(group); break
                    except Exception as error:
                        if env is not None:
                            try: close_env(env)
                            except Exception: pass
                        append_jsonl(reset_ledger, {"new_call_index": new_resets, "phase": "screening", "task": task, "cell": cell,
                            "environment_seed": seed, "attempt": attempt, "status": "mechanical_failure", "error": str(error),
                            "traceback": traceback.format_exc(), "started_at": started, "ended_at": utcnow(), "env_step_calls": 0,
                            "route_outcome_reads": 0})
                        mechanical += 1
                        fingerprint = hashlib.sha256(traceback.format_exc().encode()).hexdigest()
                        write_json(root / "sampling-budget-exhausted.json", {
                            "machine_verdict": "sampling_budget_exhausted_hold", "phase": "screening",
                            "task": task, "cell": cell, "environment_seed": seed, "new_call_index": new_resets,
                            "attempt": attempt, "traceback_fingerprint": fingerprint,
                            "prior_sp_sampling_failures": 5, "v1_4_sampling_failures": 1,
                            "shared_failure_cap": 6, "remaining_failure_calls": 0,
                            "env_step_calls": 0, "route_outcome_reads": 0, "ended_at": utcnow(),
                        })
                        write_json(root / "u2-decision.json", {"machine_verdict": "sampling_budget_exhausted_hold",
                            "R_screen": len(accepted), "actual_new_calls": new_resets, "global_total_calls": 55 + new_resets,
                            "screening_groups_available": len(screening), "expansion_groups": 0,
                            "env_step_calls": 0, "route_rollouts": 0, "route_outcome_reads": 0})
                        write_json(root / "status.json", {"status": "completed", "unit": "M2",
                            "verdict": "sampling_budget_exhausted_hold", "updated_at": utcnow(),
                            "not_run_gate": ["remaining_U2", "M3", "M4", "M5", "M6", "M7"]})
                        return 4
        del policy, model; gc.collect()
    cameras = {}
    for task in sorted(TASKS):
        for cell in CELLS:
            groups = [g for g in screening if g["task"] == task and g["cell"] == cell]
            camera = reduce_camera(task, cell, groups); cameras[(task, cell)] = camera
            write_json(root / "cameras" / f"{task}-l{cell}.json", camera)
            for group in groups: render_group(root, repo, group, camera, "screening")
    screening.sort(key=lambda r: (r["task"], r["cell"], r["environment_seed"]))
    (root / "screening-records.jsonl").write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in screening))
    selected = {task: sorted(CELLS, key=lambda cell: cell_rank(screening, task, cell))[:2] for task in sorted(TASKS)}
    write_json(root / "screening-cell-ranking.json", {"selected_cells": selected,
        "ranking_rule": ["complete_seed_groups_desc", "worst_signed_clearance_desc", "fixed_camera_min_border_desc", "joint_margin_desc", "layout_id_asc"],
        "ranked": {task: [{"cell": cell, "rank_key": cell_rank(screening, task, cell)} for cell in sorted(CELLS, key=lambda c: cell_rank(screening, task, c))] for task in sorted(TASKS)}})
    exp_reuse = expansion_reuse_members(candidate_manifest["candidates"], selected, candidate_keys)
    exp_reuse &= set(parent_exp)
    r_exp = len(exp_reuse); baseline = (30 - len(accepted)) + (12 - r_exp); absolute_cap = baseline + 1
    write_json(root / "reuse-expansion-manifest-v1.4.json", {"R_expansion_min": r_exp, "accepted": [list(x) for x in sorted(exp_reuse)],
        "selection_cells": selected, "seed_range": [3, 4, 5], "candidate_manifest": str(args.expansion_candidates),
        "candidate_manifest_sha256": sha256(args.expansion_candidates), "selection_rule": "pre-audited full criteria and final schedule match", "frozen_at": utcnow()})
    write_json(root / "exact-new-reset-budget.json", {"R_screen": len(accepted), "R_expansion": r_exp,
        "screening_new": 30 - len(accepted), "expansion_new": 12 - r_exp, "baseline_new": baseline,
        "prior_sp_sampling_failures": 5, "remaining_failure_calls": 1, "absolute_new_cap": absolute_cap,
        "pre_expansion_outer_cap": args.new_reset_cap, "global_historical_calls": 55,
        "actual_new_before_expansion": new_resets, "passed": absolute_cap <= args.new_reset_cap})
    if absolute_cap > args.new_reset_cap: raise RuntimeError("exact expansion budget exceeds frozen cap")
    expansion: list[dict[str, Any]] = []
    for task in sorted(TASKS):
        config, model, policy, env_meta, shape_meta = load_task(task)
        for cell in sorted(selected[task]):
            for seed in range(3, 6):
                key = (task, cell, seed)
                if key in exp_reuse:
                    group = canonicalize_reuse(parent_exp[key])
                    expansion.append(render_group(root, repo, group, cameras[(task, cell)], "expansion")); continue
                attempt = 1
                while True:
                    if new_resets >= absolute_cap: raise RuntimeError("exact new reset cap exceeded")
                    new_resets += 1; env = None; started = utcnow()
                    try:
                        env = create_env(config, env_meta, shape_meta, cell, seed)
                        group, snapshots = compile_reset_call(task, cell, seed, env, policy, args.scene_root,
                            root / "snapshot-attempts" / f"newcall{new_resets:02d}" / f"attempt{attempt}" /
                            f"{task}-l{cell}-seed{seed:02d}" / "raw-source", perform_explicit_reset=False)
                        group = persist_and_validate_group(root, repo, group, snapshots, new_resets, attempt, args.code_commit)
                        close_env(env); env = None
                        group = render_group(root, repo, group, cameras[(task, cell)], "expansion")
                        append_jsonl(reset_ledger, {"new_call_index": new_resets, "phase": "expansion", "task": task, "cell": cell,
                            "environment_seed": seed, "attempt": attempt, "state_sampling_call": "task_environment_constructor",
                            "explicit_reset_calls": 0, "status": "complete_snapshot_family", "started_at": started, "ended_at": utcnow(),
                            "env_step_calls": 0, "route_outcome_reads": 0})
                        expansion.append(group); break
                    except Exception as error:
                        if env is not None:
                            try: close_env(env)
                            except Exception: pass
                        append_jsonl(reset_ledger, {"new_call_index": new_resets, "phase": "expansion", "task": task, "cell": cell,
                            "environment_seed": seed, "attempt": attempt, "status": "mechanical_failure", "error": str(error),
                            "traceback": traceback.format_exc(), "started_at": started, "ended_at": utcnow(), "env_step_calls": 0,
                            "route_outcome_reads": 0})
                        mechanical += 1
                        fingerprint = hashlib.sha256(traceback.format_exc().encode()).hexdigest()
                        write_json(root / "sampling-budget-exhausted.json", {
                            "machine_verdict": "sampling_budget_exhausted_hold", "phase": "expansion",
                            "task": task, "cell": cell, "environment_seed": seed, "new_call_index": new_resets,
                            "attempt": attempt, "traceback_fingerprint": fingerprint,
                            "prior_sp_sampling_failures": 5, "v1_4_sampling_failures": 1,
                            "shared_failure_cap": 6, "remaining_failure_calls": 0,
                            "env_step_calls": 0, "route_outcome_reads": 0, "ended_at": utcnow(),
                        })
                        write_json(root / "u2-decision.json", {"machine_verdict": "sampling_budget_exhausted_hold",
                            "R_screen": len(accepted), "R_expansion_min": r_exp, "actual_new_calls": new_resets,
                            "global_total_calls": 55 + new_resets, "screening_groups": len(screening),
                            "expansion_groups_available": len(expansion), "env_step_calls": 0,
                            "route_rollouts": 0, "route_outcome_reads": 0})
                        write_json(root / "status.json", {"status": "completed", "unit": "M2",
                            "verdict": "sampling_budget_exhausted_hold", "updated_at": utcnow(),
                            "not_run_gate": ["remaining_U2", "M3", "M4", "M5", "M6", "M7"]})
                        return 4
        del policy, model; gc.collect()
    expansion.sort(key=lambda r: (r["task"], r["cell"], r["environment_seed"]))
    (root / "expansion-records.jsonl").write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in expansion))
    summary = {}
    insufficient = False
    for task in sorted(TASKS):
        summary[task] = {}
        for cell in selected[task]:
            groups = [g for g in screening + expansion if g["task"] == task and g["cell"] == cell]
            complete = [g for g in groups if g["complete_seed_group"]]
            summary[task][str(cell)] = {"group_count": len(groups), "complete_count": len(complete),
                "complete_seeds": [g["environment_seed"] for g in complete]}
            insufficient |= len(complete) < 6
    verdict = "scene_compiler_insufficient" if insufficient else "u2_source_freeze_pass"
    decision = {"machine_verdict": verdict, "R_screen": len(accepted), "R_expansion": r_exp,
        "actual_new_calls": new_resets, "global_total_calls": 55 + new_resets, "v1_4_sampling_failures": mechanical,
        "screening_groups": len(screening), "expansion_groups": len(expansion), "selected_cells": selected,
        "selected_cell_summary": summary, "env_step_calls": 0, "route_rollouts": 0, "route_outcome_reads": 0}
    write_json(root / "u2-decision.json", decision)
    write_json(root / "status.json", {"status": "completed", "unit": "SP4", "verdict": verdict, "updated_at": utcnow(), **decision})
    return 0 if verdict == "u2_source_freeze_pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
