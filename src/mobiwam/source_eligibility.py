from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from mobiwam.adapters.mobipi import (
    SourceStateIneligibleError,
    create_adapter,
)


def probe_source_eligibility(
    adapter: Any,
    source_indices: Sequence[int],
    *,
    environment_seed_start: int = 0,
) -> dict[str, Any]:
    indices = [int(index) for index in source_indices]
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise ValueError("source_indices must be non-empty, unique, and non-negative")
    if environment_seed_start < 0:
        raise ValueError("environment_seed_start must be non-negative")

    rows: list[dict[str, Any]] = []
    for source_index in indices:
        environment_seed = environment_seed_start + source_index
        adapter.prepare_source_state(source_index, environment_seed)
        try:
            snapshot = adapter.capture_source_state()
        except SourceStateIneligibleError as exc:
            rows.append(
                {
                    "source_index": source_index,
                    "environment_seed": environment_seed,
                    "eligible": False,
                    "reason": str(exc),
                }
            )
            continue
        rows.append(
            {
                "source_index": source_index,
                "environment_seed": environment_seed,
                "eligible": True,
                "reason": None,
                "source_state_id": snapshot.record.source_state_id,
                "snapshot_hash": snapshot.record.snapshot_hash,
                "observation_hash": snapshot.record.observation_hash,
            }
        )

    eligible_indices = [row["source_index"] for row in rows if row["eligible"]]
    return {
        "schema_version": "1.0",
        "probed_source_count": len(rows),
        "eligible_source_count": len(eligible_indices),
        "ineligible_source_count": len(rows) - len(eligible_indices),
        "eligible_source_indices": eligible_indices,
        "sources": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Mobi-pi source states before candidate rollouts"
    )
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--eligible-indices", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--source-count", type=int, required=True)
    parser.add_argument("--environment-seed-start", type=int, default=0)
    args = parser.parse_args()

    if args.start_index < 0 or args.source_count <= 0:
        parser.error("start-index must be non-negative and source-count must be positive")
    config = json.loads(args.adapter_config.read_text(encoding="utf-8"))
    adapter = create_adapter(output_root=args.output_root, config=config)
    report = probe_source_eligibility(
        adapter,
        range(args.start_index, args.start_index + args.source_count),
        environment_seed_start=args.environment_seed_start,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.eligible_indices.parent.mkdir(parents=True, exist_ok=True)
    args.eligible_indices.write_text(
        json.dumps(report["eligible_source_indices"], indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
