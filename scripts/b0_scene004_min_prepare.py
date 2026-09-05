#!/usr/bin/env python3
"""Create the fail-closed M0 inheritance receipt for SCENE-004 MIN-v1.4."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_SP = (
    "reuse-screening-manifest.json",
    "screening-budget-freeze.json",
    "semantic-equivalence-receipt.json",
    "new-reset-ledger.jsonl",
    "sampler-roundtrip-failure.json",
    "global-reset-summary.json",
    "completion.json",
    "run-manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--sp-parent", type=Path, required=True)
    parser.add_argument("--ri-parent", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()

    inventory_path = args.sp_parent / "artifact-checksum-inventory.json"
    inventory = json.loads(inventory_path.read_text())
    expected = {row["path"]: row for row in inventory["files"]}
    checks = []
    for name in REQUIRED_SP:
        path = args.sp_parent / name
        row = expected.get(name, {})
        actual = sha256(path) if path.is_file() else None
        checks.append({"path": str(path), "exists": path.is_file(), "actual_sha256": actual,
                       "inventory_sha256": row.get("sha256"), "passed": actual == row.get("sha256")})

    inherited = [
        args.sp_parent / "zero-reset-camera-smoke-receipt.json",
        args.ri_parent / "renderer-isolation-qualification.json",
    ]
    for path in inherited:
        parent_inventory = json.loads((path.parents[0 if path.parent == args.sp_parent else 0] / "artifact-checksum-inventory.json").read_text())
        parent_expected = {row["path"]: row for row in parent_inventory["files"]}
        relative = str(path.relative_to(path.parent))
        # Both inherited receipts are top-level files in their parent run.
        actual = sha256(path) if path.is_file() else None
        checks.append({"path": str(path), "exists": path.is_file(), "actual_sha256": actual,
                       "inventory_sha256": parent_expected.get(relative, {}).get("sha256"),
                       "passed": actual == parent_expected.get(relative, {}).get("sha256")})

    plan = args.control / "08-experiments/plans/2026-09-03-mobiwam-obc002-frozen-mainline-plan.md"
    contract = args.control / "08-experiments/contracts/2026-09-05-obc-wam-b0-scene-004-deadline-n48-chained-continuation.md"
    prompt = args.control / "08-experiments/prompt/2026-09-05-obc-wam-b0-scene-004-deadline-n48-chained-continuation.md"
    code_head = git(args.code, "rev-parse", "HEAD")
    payload = {
        "contract_id": "MMWAM-OBC-002-B0-SCENE-004-MIN-v1.4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {"plan": {"path": str(plan), "sha256": sha256(plan)},
                  "contract": {"path": str(contract), "sha256": sha256(contract)},
                  "prompt": {"path": str(prompt), "sha256": sha256(prompt)}},
        "hash_assertions": {"plan": sha256(plan) == "b061d8facc1b7f72352afd1ebd90123378374ba25dcc15f0563afd2e7471813a",
                            "contract": sha256(contract) == "aef3e6894664d3f63caeca49708e3799bedf6ad864c1cc75f2fd601ba9fb8572"},
        "research": {"commit": git(args.control, "rev-parse", "HEAD"), "branch": git(args.control, "branch", "--show-current"),
                     "origin": git(args.control, "remote", "get-url", "origin"), "clean": not bool(git(args.control, "status", "--porcelain")),
                     "parity": git(args.control, "rev-list", "--left-right", "--count", "HEAD...@{upstream}")},
        "code": {"commit": code_head, "branch": git(args.code, "branch", "--show-current"),
                 "origin": git(args.code, "remote", "get-url", "origin"), "clean": not bool(git(args.code, "status", "--porcelain")),
                 "parity": git(args.code, "rev-list", "--left-right", "--count", "HEAD...@{upstream}"),
                 "minimum_commit_is_ancestor": subprocess.run(["git", "-C", str(args.code), "merge-base", "--is-ancestor",
                     "e34eb296f029aeae612950d51a3e65e14e71cbd7", code_head]).returncode == 0},
        "sp_receipt_checks": checks,
        "all_inherited_receipts_verified": all(row["passed"] for row in checks),
        "frozen_lineage": {"R_screen": 17, "old_scene004_calls": 50, "sp_v1_3_failed_sampling_calls": 5,
                           "shared_sampling_failure_cap": 6, "remaining_failure_calls": 1},
        "zero_state_sampling_calls": True,
    }
    passed = (payload["all_inherited_receipts_verified"] and all(payload["hash_assertions"].values())
              and payload["research"]["clean"] and payload["research"]["parity"] == "0\t0"
              and payload["code"]["clean"] and payload["code"]["parity"] == "0\t0"
              and payload["code"]["minimum_commit_is_ancestor"])
    payload["passed"] = passed
    write(args.artifact_root / "alignment-receipt.json", payload)
    if not passed:
        write(args.artifact_root / "status.json", {"status": "completed", "verdict": "provenance_invalid_hold",
              "updated_at": datetime.now(timezone.utc).isoformat()})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
