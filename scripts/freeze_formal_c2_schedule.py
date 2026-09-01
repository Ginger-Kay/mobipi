#!/usr/bin/env python3
"""Freeze the outcome-blind MMWAM-OBC-001-C2 formal source schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TASKS = (
    "CloseSingleDoor",
    "CloseDrawer",
    "TurnOnFaucet",
    "TurnOnMicrowave",
)
LAYOUTS = (1, 4, 7, 8, 9)
SIGMAS = (0.0, 0.1, 0.2)
LOCKED_LAYOUTS = {
    "CloseSingleDoor": 1,
    "CloseDrawer": 8,
    "TurnOnFaucet": 4,
    "TurnOnMicrowave": 8,
}
SPLIT_COUNTS_PER_TASK = {
    "train": 66,
    "validation": 12,
    "calibration": 18,
    "locked_test": 24,
}
REPEATS_BY_SPLIT = {
    "train": 2,
    "validation": 2,
    "calibration": 3,
    "locked_test": 3,
}
FORBIDDEN_ELIGIBILITY_FIELDS = frozenset(
    {
        "success",
        "hard_valid",
        "irreversible_failure",
        "collision",
        "contact_loss",
        "progress_delta",
        "route_cost",
        "candidate_id",
        "route_type",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stratum(source_index: int) -> tuple[int, float]:
    return (
        LAYOUTS[(source_index // len(SIGMAS)) % len(LAYOUTS)],
        SIGMAS[source_index % len(SIGMAS)],
    )


def canonical_row(row: dict[str, Any], *, split: str | None = None) -> dict[str, Any]:
    source_index = int(row["source_index"])
    layout, sigma = stratum(source_index)
    result = {
        "source_index": source_index,
        "source_state_id": str(row["source_state_id"]),
        "environment_seed": int(row["environment_seed"]),
        "layout_id": layout,
        "base_noise_sigma": sigma,
        "snapshot_hash": str(row["snapshot_hash"]),
        "observation_hash": str(row["observation_hash"]),
    }
    if split is not None:
        result["split"] = split
        result["repeats_per_candidate"] = REPEATS_BY_SPLIT[split]
    return result


def stable_rows(
    rows: Iterable[dict[str, Any]], *, seed: int, task: str, purpose: str
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: stable_hash(
            [seed, task, purpose, int(row["source_index"])]
        ),
    )


def balanced_select(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: int,
    task: str,
    purpose: str,
) -> list[dict[str, Any]]:
    remaining = list(rows)
    selected: list[dict[str, Any]] = []
    strata: Counter[tuple[int, float]] = Counter()
    layouts: Counter[int] = Counter()
    sigmas: Counter[float] = Counter()
    while len(selected) < count:
        if not remaining:
            raise RuntimeError(f"{task}: exhausted eligible rows for {purpose}")
        row = min(
            remaining,
            key=lambda candidate: (
                strata[stratum(int(candidate["source_index"]))],
                layouts[stratum(int(candidate["source_index"]))[0]],
                sigmas[stratum(int(candidate["source_index"]))[1]],
                stable_hash(
                    [seed, task, purpose, int(candidate["source_index"])]
                ),
            ),
        )
        remaining.remove(row)
        selected.append(row)
        layout, sigma = stratum(int(row["source_index"]))
        strata[(layout, sigma)] += 1
        layouts[layout] += 1
        sigmas[sigma] += 1
    return selected


def validate_candidate_grid(grid: dict[str, Any]) -> None:
    expected = {
        "execute_candidates": ["e0"],
        "dock_candidates": [f"d{index}" for index in range(5)],
        "assist_candidates": [f"a{index}" for index in range(5)],
    }
    for field, candidate_ids in expected.items():
        observed = [str(row.get("candidate_id")) for row in grid.get(field, [])]
        if observed != candidate_ids:
            raise RuntimeError(f"candidate grid {field} differs from frozen v2 support")
    if int(grid.get("seeds_per_candidate", 0)) != 2:
        raise RuntimeError("the immutable v2 candidate grid must retain pilot repeats=2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility-manifest", type=Path, required=True)
    parser.add_argument("--split-feasibility-decision", type=Path, required=True)
    parser.add_argument("--base-config-root", type=Path, required=True)
    parser.add_argument("--candidate-grid", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--selection-seed", type=int, default=20260901)
    parser.add_argument("--split-seed", type=int, default=20260901)
    args = parser.parse_args()

    if args.split_seed != 20260901:
        parser.error("C2 freezes split seed 20260901")
    if len(args.code_commit) != 40:
        parser.error("code commit must be a full 40-character Git object ID")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")

    grid = json.loads(args.candidate_grid.read_text(encoding="utf-8"))
    validate_candidate_grid(grid)
    grid_sha256 = sha256_file(args.candidate_grid)
    environment_seed_start = int(grid.get("environment_seed_start", -1))
    if environment_seed_start < 0:
        raise RuntimeError("candidate grid lacks a non-negative environment seed start")
    eligibility = json.loads(args.eligibility_manifest.read_text(encoding="utf-8"))
    if eligibility.get("status") != "pass":
        raise RuntimeError("eligibility manifest is not a pass")
    if not eligibility.get("pilot_indices_excluded", False):
        raise RuntimeError("eligibility manifest does not exclude pilot/G0 indices")
    if eligibility.get("selection_uses_rollout_outcomes") is not False:
        raise RuntimeError("eligibility selection is not outcome blind")

    feasibility = json.loads(
        args.split_feasibility_decision.read_text(encoding="utf-8")
    )
    if feasibility.get("status") != "pass_task_specific_layout_holdout":
        raise RuntimeError("split feasibility decision is not a pass")
    if feasibility.get("selection_uses_rollout_outcomes") is not False:
        raise RuntimeError("split feasibility decision is not outcome blind")
    if feasibility.get("proposed_task_specific_held_out_locked_layouts") != LOCKED_LAYOUTS:
        raise RuntimeError("task-specific locked layouts differ from frozen G0 decision")

    base_manifest_path = args.base_config_root / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_commit = str(base_manifest["code_commit"])
    if base_manifest.get("candidate_grid", {}).get("sha256") != grid_sha256:
        raise RuntimeError("base config manifest candidate grid differs")

    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    reserves_by_task: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, Path] = {}
    for task in TASKS:
        task_entry = eligibility["tasks"][task]
        report_path = Path(task_entry["report"])
        if sha256_file(report_path) != str(task_entry["report_sha256"]):
            raise RuntimeError(f"{task}: eligibility report hash differs")
        report_paths[task] = report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = [dict(row) for row in report["sources"] if row.get("eligible")]
        if any(FORBIDDEN_ELIGIBILITY_FIELDS.intersection(row) for row in rows):
            raise RuntimeError(f"{task}: eligibility rows contain rollout outcomes")
        if len(rows) != int(task_entry["eligible_source_count"]):
            raise RuntimeError(f"{task}: eligible count differs from manifest")
        if len({int(row["source_index"]) for row in rows}) != len(rows):
            raise RuntimeError(f"{task}: duplicate eligible source index")
        for row in rows:
            index = int(row["source_index"])
            layout, sigma = stratum(index)
            expected_id = (
                f"{task}-l{layout}-sig{sigma:.2f}"
                f"-seed{environment_seed_start + index}"
            )
            if str(row["source_state_id"]) != expected_id:
                raise RuntimeError(f"{task}: source identity/stratum mismatch at {index}")

        locked_layout = LOCKED_LAYOUTS[task]
        locked: list[dict[str, Any]] = []
        for sigma in SIGMAS:
            candidates = [
                row
                for row in rows
                if stratum(int(row["source_index"])) == (locked_layout, sigma)
            ]
            ordered = stable_rows(
                candidates,
                seed=args.selection_seed,
                task=task,
                purpose=f"locked-sigma-{sigma:.2f}",
            )
            if len(ordered) < 8:
                raise RuntimeError(f"{task}: locked layout lacks 8 rows for sigma {sigma}")
            locked.extend(ordered[:8])

        development_pool = [
            row
            for row in rows
            if stratum(int(row["source_index"]))[0] != locked_layout
        ]
        development = balanced_select(
            development_pool,
            count=96,
            seed=args.selection_seed,
            task=task,
            purpose="development-source-selection",
        )
        validation = balanced_select(
            development,
            count=SPLIT_COUNTS_PER_TASK["validation"],
            seed=args.split_seed,
            task=task,
            purpose="validation-split",
        )
        validation_ids = {int(row["source_index"]) for row in validation}
        after_validation = [
            row for row in development if int(row["source_index"]) not in validation_ids
        ]
        calibration = balanced_select(
            after_validation,
            count=SPLIT_COUNTS_PER_TASK["calibration"],
            seed=args.split_seed,
            task=task,
            purpose="calibration-split",
        )
        calibration_ids = {int(row["source_index"]) for row in calibration}
        train = [
            row
            for row in after_validation
            if int(row["source_index"]) not in calibration_ids
        ]
        rows_with_split = (
            [canonical_row(row, split="train") for row in train]
            + [canonical_row(row, split="validation") for row in validation]
            + [canonical_row(row, split="calibration") for row in calibration]
            + [canonical_row(row, split="locked_test") for row in locked]
        )
        rows_with_split.sort(key=lambda row: int(row["source_index"]))
        counts = Counter(str(row["split"]) for row in rows_with_split)
        if dict(counts) != SPLIT_COUNTS_PER_TASK:
            raise RuntimeError(f"{task}: split counts differ: {dict(counts)}")
        locked_rows = [row for row in rows_with_split if row["split"] == "locked_test"]
        if {int(row["layout_id"]) for row in locked_rows} != {locked_layout}:
            raise RuntimeError(f"{task}: locked layout transfer is not isolated")
        if Counter(float(row["base_noise_sigma"]) for row in locked_rows) != Counter(
            {0.0: 8, 0.1: 8, 0.2: 8}
        ):
            raise RuntimeError(f"{task}: locked sigma balance differs")
        selected_by_task[task] = rows_with_split

        used = {int(row["source_index"]) for row in rows_with_split}
        remaining = [row for row in rows if int(row["source_index"]) not in used]
        reserves_by_task[task] = {
            "selection_uses_rollout_outcomes": False,
            "replacement_rule": "replace only an initial-invalid source before any route outcome; append ledger and retain the original split",
            "development": [
                canonical_row(row)
                for row in stable_rows(
                    [
                        row
                        for row in remaining
                        if stratum(int(row["source_index"]))[0] != locked_layout
                    ],
                    seed=args.selection_seed,
                    task=task,
                    purpose="development-replacement-reserve",
                )
            ],
            "locked_by_sigma": {
                f"{sigma:.2f}": [
                    canonical_row(row)
                    for row in stable_rows(
                        [
                            row
                            for row in remaining
                            if stratum(int(row["source_index"]))
                            == (locked_layout, sigma)
                        ],
                        seed=args.selection_seed,
                        task=task,
                        purpose=f"locked-replacement-reserve-{sigma:.2f}",
                    )
                ]
                for sigma in SIGMAS
            },
        }

    global_counts = Counter(
        row["split"] for rows in selected_by_task.values() for row in rows
    )
    expected_global = {
        split: count * len(TASKS) for split, count in SPLIT_COUNTS_PER_TASK.items()
    }
    if dict(global_counts) != expected_global:
        raise RuntimeError(f"global split counts differ: {dict(global_counts)}")
    total_rollouts = sum(
        int(global_counts[split]) * REPEATS_BY_SPLIT[split] * 11
        for split in SPLIT_COUNTS_PER_TASK
    )
    if total_rollouts != 12408:
        raise RuntimeError(f"formal rollout count differs: {total_rollouts}")

    scientific_schedule = {
        "schema_version": "1.0",
        "experiment_id": "MMWAM-OBC-001",
        "contract_id": "MMWAM-OBC-001-C2",
        "code_commit": args.code_commit,
        "candidate_grid_sha256": grid_sha256,
        "selection_seed": args.selection_seed,
        "split_seed": args.split_seed,
        "source_count": 480,
        "sources_per_task": 120,
        "split_counts": expected_global,
        "split_counts_per_task": SPLIT_COUNTS_PER_TASK,
        "repeats_by_split": REPEATS_BY_SPLIT,
        "seed_stride_per_source": 3,
        "candidate_support": {"E": 1, "D": 5, "A": 5},
        "total_rollouts": total_rollouts,
        "locked_layouts": LOCKED_LAYOUTS,
        "selected_sources": selected_by_task,
        "replacement_reserves": reserves_by_task,
    }
    schedule_checksum = stable_hash(scientific_schedule)

    temporary = args.output_root.with_name(args.output_root.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary output: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "configs").mkdir()
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": "MMWAM-OBC-001",
        "contract_id": "MMWAM-OBC-001-C2",
        "status": "frozen_not_launched",
        "review_status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_test_read": False,
        "selection_uses_rollout_outcomes": False,
        "pilot_and_g0_sources_excluded": True,
        "source_atomic": True,
        "code_commit": args.code_commit,
        "base_config_code_commit": base_commit,
        "source_count": 480,
        "sources_per_task": 120,
        "split_counts": expected_global,
        "split_counts_per_task": SPLIT_COUNTS_PER_TASK,
        "repeats_by_split": REPEATS_BY_SPLIT,
        "seed_stride_per_source": 3,
        "candidate_support": {"E": 1, "D": 5, "A": 5},
        "total_rollouts": total_rollouts,
        "schedule_checksum": schedule_checksum,
        "scientific_schedule": scientific_schedule,
        "inputs": {
            "eligibility_manifest": str(args.eligibility_manifest),
            "eligibility_manifest_sha256": sha256_file(args.eligibility_manifest),
            "split_feasibility_decision": str(args.split_feasibility_decision),
            "split_feasibility_decision_sha256": sha256_file(
                args.split_feasibility_decision
            ),
            "base_config_manifest": str(base_manifest_path),
            "base_config_manifest_sha256": sha256_file(base_manifest_path),
            "candidate_grid": str(args.candidate_grid),
            "candidate_grid_sha256": grid_sha256,
        },
        "replacement_policy": {
            "allowed_only_before_any_route_outcome": True,
            "append_only_ledger": str(args.output_root / "replacement-ledger.jsonl"),
            "locked_replacement_preserves_layout_and_sigma": True,
            "development_replacement_preserves_nonlocked_layout_scope": True,
        },
        "tasks": {},
    }

    for task in TASKS:
        selected = selected_by_task[task]
        split_map = {str(row["source_state_id"]): str(row["split"]) for row in selected}
        config_path = args.base_config_root / f"{task}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if str(config.get("code_commit")) != base_commit:
            raise RuntimeError(f"{task}: base config commit differs from manifest")
        config.update(
            {
                "code_commit": args.code_commit,
                "contract_id": "MMWAM-OBC-001-C2",
                "data_role": "formal_paired_collection",
                "source_split_map": split_map,
                "schedule_checksum": schedule_checksum,
                "candidate_grid_sha256": grid_sha256,
                "formal_repeats_by_split": REPEATS_BY_SPLIT,
                "seed_stride_per_source": 3,
                "labeler_version": "mobipi-four-task-collision-risk-v2",
            }
        )
        output_config = temporary / "configs" / f"{task}.json"
        output_config.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        indices_by_split = {
            split: [
                int(row["source_index"]) for row in selected if row["split"] == split
            ]
            for split in SPLIT_COUNTS_PER_TASK
        }
        indices_file = temporary / f"{task}-source-indices-by-split.json"
        indices_file.write_text(
            json.dumps(indices_by_split, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reserve_file = temporary / f"{task}-replacement-reserve.json"
        reserve_file.write_text(
            json.dumps(reserves_by_task[task], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["tasks"][task] = {
            "locked_layout": LOCKED_LAYOUTS[task],
            "eligibility_report": str(report_paths[task]),
            "eligibility_report_sha256": sha256_file(report_paths[task]),
            "selected_sources": selected,
            "split_counts": dict(Counter(row["split"] for row in selected)),
            "config": str(args.output_root / "configs" / f"{task}.json"),
            "config_sha256": sha256_file(output_config),
            "source_indices_by_split": str(
                args.output_root / f"{task}-source-indices-by-split.json"
            ),
            "source_indices_by_split_sha256": sha256_file(indices_file),
            "replacement_reserve": str(
                args.output_root / f"{task}-replacement-reserve.json"
            ),
            "replacement_reserve_sha256": sha256_file(reserve_file),
        }

    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.rename(temporary, args.output_root)
    print(args.output_root / "manifest.json")
    print(schedule_checksum)


if __name__ == "__main__":
    main()
