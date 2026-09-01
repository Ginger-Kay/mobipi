#!/usr/bin/env python3
"""Plan C2 dynamic two-source, source-atomic formal collection shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


TASKS = (
    "CloseSingleDoor",
    "CloseDrawer",
    "TurnOnFaucet",
    "TurnOnMicrowave",
)
SPLITS = ("train", "validation", "calibration", "locked_test")
REPEATS = {"train": 2, "validation": 2, "calibration": 3, "locked_test": 3}
PHASES = {
    "train": "train_validation",
    "validation": "train_validation",
    "calibration": "calibration",
    "locked_test": "locked_deferred",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--candidate-grid", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--sources-per-shard", type=int, default=2)
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--visible-gpus", type=int, default=4)
    args = parser.parse_args()

    if args.sources_per_shard != 2:
        parser.error("C2 throughput decision freezes two sources per shard")
    if args.workers_per_gpu != 3 or args.visible_gpus != 4:
        parser.error("G0 throughput decision freezes 3 workers on each of 4 GPUs")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")

    schedule = json.loads(args.schedule_manifest.read_text(encoding="utf-8"))
    if schedule.get("status") != "frozen_not_launched":
        raise RuntimeError("formal schedule is not frozen-not-launched")
    if str(schedule.get("code_commit")) != args.code_commit:
        raise RuntimeError("schedule code commit differs")
    if stable_hash(schedule["scientific_schedule"]) != schedule["schedule_checksum"]:
        raise RuntimeError("scientific schedule checksum differs")
    if schedule.get("split_counts") != {
        "train": 264,
        "validation": 48,
        "calibration": 72,
        "locked_test": 96,
    }:
        raise RuntimeError("schedule split counts differ from C2")
    if schedule.get("repeats_by_split") != REPEATS:
        raise RuntimeError("schedule repeats differ from C2")

    grid_sha256 = sha256_file(args.candidate_grid)
    if grid_sha256 != schedule["inputs"]["candidate_grid_sha256"]:
        raise RuntimeError("candidate grid hash differs from frozen schedule")

    temporary = args.output_root.with_name(args.output_root.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary output: {temporary}")
    temporary.mkdir(parents=True)
    shards: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    source_keys: set[tuple[str, int]] = set()
    phase_order = {"train_validation": 0, "calibration": 1, "locked_deferred": 2}

    for task in TASKS:
        rows = schedule["tasks"][task]["selected_sources"]
        for split in SPLITS:
            split_rows = sorted(
                (row for row in rows if row["split"] == split),
                key=lambda row: int(row["source_index"]),
            )
            for start in range(0, len(split_rows), args.sources_per_shard):
                chunk = split_rows[start : start + args.sources_per_shard]
                shard_number = start // args.sources_per_shard
                shard_id = f"{task}-{split}-shard-{shard_number:02d}"
                indices = [int(row["source_index"]) for row in chunk]
                ids = [str(row["source_state_id"]) for row in chunk]
                keys = {(task, index) for index in indices}
                if source_ids.intersection(ids) or source_keys.intersection(keys):
                    raise RuntimeError(f"duplicate source in {shard_id}")
                source_ids.update(ids)
                source_keys.update(keys)
                index_path = temporary / f"{shard_id}-source-indices.json"
                index_path.write_text(
                    json.dumps(indices, indent=2) + "\n", encoding="utf-8"
                )
                repeats = REPEATS[split]
                phase = PHASES[split]
                shards.append(
                    {
                        "shard_id": shard_id,
                        "task": task,
                        "split": split,
                        "phase": phase,
                        "queue_round": shard_number,
                        "source_count": len(chunk),
                        "source_indices": str(
                            args.output_root / f"{shard_id}-source-indices.json"
                        ),
                        "source_indices_sha256": sha256_file(index_path),
                        "source_state_ids": ids,
                        "repeats_per_candidate": repeats,
                        "seed_stride_per_source": 3,
                        "candidate_support": {"E": 1, "D": 5, "A": 5},
                        "expected_rollouts": len(chunk) * 11 * repeats,
                        "source_atomic": True,
                        "logical_launch_unit": True,
                        "attempt_budget": 5,
                        "launch_status": (
                            "sealed_until_locked_freeze"
                            if split == "locked_test"
                            else "not_launched"
                        ),
                    }
                )

    shards.sort(
        key=lambda row: (
            phase_order[str(row["phase"])],
            int(row["queue_round"]),
            TASKS.index(str(row["task"])),
            SPLITS.index(str(row["split"])),
        )
    )
    if len(source_ids) != 480:
        raise RuntimeError(f"planned {len(source_ids)} sources, expected 480")
    if sum(int(row["expected_rollouts"]) for row in shards) != 12408:
        raise RuntimeError("planned rollout count differs from 12,408")
    if len(shards) != 240:
        raise RuntimeError(f"planned {len(shards)} shards, expected 240")

    phase_source_counts = Counter()
    phase_rollout_counts = Counter()
    for shard in shards:
        phase_source_counts[str(shard["phase"])] += int(shard["source_count"])
        phase_rollout_counts[str(shard["phase"])] += int(shard["expected_rollouts"])
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": "MMWAM-OBC-001",
        "contract_id": "MMWAM-OBC-001-C2",
        "stage": "S3_formal_paired_collection",
        "status": "planned_not_launched",
        "review_status": "pending",
        "locked_test_read": False,
        "code_commit": args.code_commit,
        "distributed_multi_gpu": False,
        "independent_single_gpu_workers": True,
        "dynamic_queue": True,
        "workers_per_gpu": 3,
        "visible_gpus": 4,
        "total_workers": 12,
        "minimum_free_memory_fraction": 0.15,
        "sources_per_shard": 2,
        "source_atomic": True,
        "attempt_budget_per_shard": 5,
        "source_count": 480,
        "shard_count": len(shards),
        "total_rollouts": 12408,
        "phase_source_counts": dict(phase_source_counts),
        "phase_rollout_counts": dict(phase_rollout_counts),
        "phase_order": ["train_validation", "calibration", "locked_deferred"],
        "locked_rule": "locked shards are not launchable until model, calibration, thresholds, and statistics freeze",
        "formal_schedule_manifest": str(args.schedule_manifest),
        "formal_schedule_manifest_sha256": sha256_file(args.schedule_manifest),
        "formal_schedule_checksum": schedule["schedule_checksum"],
        "candidate_grid": str(args.candidate_grid),
        "candidate_grid_sha256": grid_sha256,
        "shards": shards,
    }
    manifest["plan_checksum"] = stable_hash(manifest)
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.rename(temporary, args.output_root)
    print(args.output_root / "manifest.json")
    print(manifest["plan_checksum"])


if __name__ == "__main__":
    main()
