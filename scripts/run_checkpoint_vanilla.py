#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mobiwam.adapters.mobipi import create_adapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Mobi-pi vanilla from checkpoint-carried metadata and stats"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=1)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = create_adapter(output_root=args.output_root, config=config)
    adapter.prepare_source_state(args.source_index, args.environment_seed)
    snapshot = adapter.capture_source_state()
    restore = adapter.restore_source_state(snapshot)
    if not restore.passed:
        raise RuntimeError(
            "snapshot restore failed before vanilla rollout: "
            f"state expected={snapshot.opaque_handle.snapshot_hash} actual={restore.snapshot_hash}; "
            f"observation expected={snapshot.opaque_handle.observation_hash} actual={restore.observation_hash}; "
            f"controller expected={snapshot.opaque_handle.controller_hash} actual={restore.controller_hash}; "
            f"contact expected={snapshot.opaque_handle.contact_hash} actual={restore.contact_hash}"
        )
    record = adapter.run_vanilla(snapshot, policy_seed=args.policy_seed)
    output = args.output_root / "vanilla_record.json"
    output.write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    if not record.hard_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
