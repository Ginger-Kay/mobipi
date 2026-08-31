from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import paired_cluster_bootstrap, risk_coverage_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-shot locked source-level evaluation")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    args = parser.parse_args()
    data_file = np.load(args.predictions, allow_pickle=False)
    data = {name: data_file[name] for name in data_file.files}
    if set(data["split"].astype(str)) != {"locked_test"}:
        raise ValueError("locked evaluator refuses non-locked-test rows")
    source_ids = data["source_ids"].astype(str)
    effects = {source_id: float(data["selected_utility"][index] - data["baseline_utility"][index]) for index, source_id in enumerate(source_ids)}
    strata = {source_id: str(data["strata"][index]) for index, source_id in enumerate(source_ids)}
    bootstrap = paired_cluster_bootstrap(effects, strata=strata, resamples=10_000, seed=args.bootstrap_seed)
    curve = risk_coverage_curve(success=data["success"], irreversible=data["irreversible_risk"], uncertainty=data["uncertainty"])
    record = {
        "split": "locked_test", "locked_test_opened": True,
        "paired_effect": bootstrap.__dict__,
        "risk_coverage": {name: value.tolist() for name, value in curve.items()},
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite locked evaluation: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
