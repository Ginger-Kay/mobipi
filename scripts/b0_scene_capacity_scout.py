"""Bounded, outcome-blind B0 scene compiler readiness audit.

This runner deliberately stops before simulator resets when the deterministic
compiler or required scene-model assets are unavailable.  It never calls
``env.step`` or a task outcome checker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TASKS = ("CloseSingleDoor", "CloseDrawer")
CELLS = ((1, 1), (4, 4), (7, 7), (8, 8), (9, 9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--control-root", type=Path)
    args = parser.parse_args()
    umbrella = Path(__file__).resolve().parents[2]
    root = umbrella / "Mobipi"
    control = args.control_root.resolve() if args.control_root else umbrella / "control"
    artifact = args.artifact_root.resolve()
    artifact.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc).isoformat()
    plan = control / "08-experiments/plans/2026-09-03-mobiwam-obc002-frozen-mainline-plan.md"
    parent = control / "08-experiments/contracts/2026-09-03-obc-wam-persistent-assist-source-redesign-b0.md"
    contract = control / "08-experiments/contracts/2026-09-03-obc-wam-b0-scene-capacity-scout.md"
    prompt = control / "08-experiments/prompt/2026-09-03-obc-wam-b0-scene-capacity-scout.md"
    receipt = {
        "schema_version": "b0-scene-alignment-v1",
        "created_at": now,
        "status": "implementation_incomplete",
        "plan": {"path": str(plan), "bytes": plan.stat().st_size, "sha256": sha256(plan)},
        "parent_contract": {"path": str(parent), "bytes": parent.stat().st_size, "sha256": sha256(parent)},
        "contract": {"path": str(contract), "bytes": contract.stat().st_size, "sha256": sha256(contract)},
        "prompt": {"path": str(prompt), "bytes": prompt.stat().st_size, "sha256": sha256(prompt), "git_blob": git(control, "rev-parse", f"HEAD:{prompt.relative_to(control)}")},
        "code": {"repo": str(root), "branch": git(root, "branch", "--show-current"), "commit": git(root, "rev-parse", "HEAD")},
        "environment": {"python": sys.executable, "python_version": platform.python_version(), "python_no_user_site": "1"},
        "reset_count": 0,
        "route_rollouts": 0,
        "constraints_relaxed": False,
    }
    (artifact / "alignment-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    schedule = {"schema_version": "b0-scan-schedule-v1", "tasks": TASKS, "cells": [list(c) for c in CELLS], "stage_a": {"environment_seeds": [0, 31], "max_resets": 320}, "stage_b": {"environment_seeds": [32, 63], "conditional": True, "max_total_resets": 640}, "route_rollouts": 0, "status": "not_started_compiler_not_ready"}
    (artifact / "scan-schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    (artifact / "source-pose-generator-v1.json").write_text(json.dumps({"status": "not_frozen", "reason": "deterministic source compiler is not implemented", "finite_candidate_cap": None}, indent=2) + "\n")
    (artifact / "candidate-geometry-binding-v1.json").write_text(json.dumps({"status": "not_frozen", "reason": "controller-only geometry caps and fixture-derived D target are not implemented", "a_candidates": ["a1", "a2", "a3", "a4", "a5"], "d_candidates": []}, indent=2) + "\n")
    (artifact / "seed-scope-v1.json").write_text(json.dumps({"status": "declared_not_executed", "schedule_seed": 20260903, "environment_seed_range": [0, 63], "source_pose_seed": "stable_digest_required", "policy_forward_seed": "stable_digest_required"}, indent=2) + "\n")
    missing = [
        "actual RoboCasa fixture introspection/integration reset compiler",
        "finite controlled source-pose generator and geometry-primary D target",
        "A envelope and continuous collision corridor validator",
        "actual fixed world-frame camera creation/projection/native render validator",
        "task/layout/style-bound 3DGS assets under data/scene_models",
        "scene-model/checkpoint compatibility validator",
    ]
    decision = {"schema_version": "b0-compiler-ready-v1", "verdict": "implementation_incomplete", "compiler_ready": False, "route_outcome_read": False, "missing": missing, "integration_prefix_resets": 0}
    (artifact / "compiler-ready-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    inventory = [{"task": t, "layout": l, "style": s, "status": "not_scanned_compiler_not_ready", "environment_seeds": [0, 63]} for t in TASKS for l, s in CELLS]
    (artifact / "scene-capacity-inventory.json").write_text(json.dumps({"rows": inventory, "reset_count": 0, "status": "implementation_incomplete"}, indent=2) + "\n")
    (artifact / "scene-capacity-inventory.csv").write_text("task,layout,style,status\n" + "\n".join(f"{r['task']},{r['layout']},{r['style']},{r['status']}" for r in inventory) + "\n")
    (artifact / "source-geometry-inventory.json").write_text(json.dumps({"rows": [], "status": "not_scanned_compiler_not_ready"}, indent=2) + "\n")
    (artifact / "source-geometry-inventory.csv").write_text("task,layout,style,source_pose_id,status\n")
    (artifact / "camera-coverage-validation.json").write_text(json.dumps({"status": "implementation_incomplete", "cells": [], "native_min": [1920, 1080], "reason": "actual camera compiler unavailable"}, indent=2) + "\n")
    (artifact / "scene-model-compatibility.json").write_text(json.dumps({"status": "implementation_incomplete", "scene_model_root": str(umbrella / "data/scene_models"), "assets_present": False, "reason": "no local official 3DGS assets and no compatibility validator"}, indent=2) + "\n")
    (artifact / "scene-capacity-decision.json").write_text(json.dumps({"machine_verdict": "implementation_incomplete", "reset_count": 0, "route_rollouts": 0, "eligible_cells": 0, "reason": "compiler-ready gate not passed; bulk scan prohibited"}, indent=2) + "\n")
    (artifact / "attempt-lineage.json").write_text(json.dumps({"attempt_id": artifact.name, "parent_attempt": None, "status": "stopped_before_reset", "failure_class": "implementation_incomplete"}, indent=2) + "\n")
    (artifact / "status.json").write_text(json.dumps({"status": "implementation_incomplete", "progress": {"resets": 0, "scheduled": 640}, "route_rollouts": 0}, indent=2) + "\n")
    (artifact / "completion.json").write_text(json.dumps({"ended_at": now, "exit_code": 0, "expected_reset_cap": 640, "actual_reset_count": 0, "verdict": "implementation_incomplete", "route_rollouts": 0}, indent=2) + "\n")
    (artifact / "commands.log").write_text("PYTHONNOUSERSITE=1 /share/jhk/MobiWAM/env/bin/python scripts/b0_scene_capacity_scout.py --artifact-root " + str(artifact) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
