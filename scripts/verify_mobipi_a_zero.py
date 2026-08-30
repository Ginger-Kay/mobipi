#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mobiwam.adapters.mobipi import create_adapter


def max_abs_error(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("inf")
    return float(np.max(np.abs(first - second))) if first.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify A(alpha=0) equals E")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["assist_fraction_toward_dock"] = 0.0
    config["assist_max_translation_m"] = 0.0
    config["assist_max_yaw_rad"] = 0.0
    adapter = create_adapter(output_root=args.output_root, config=config)
    adapter.prepare_source_state(args.source_index, args.environment_seed)
    snapshot = adapter.capture_source_state()

    restore_e = adapter.restore_source_state(snapshot)
    if not restore_e.passed:
        raise RuntimeError("snapshot restore failed before E")
    macro = adapter.sample_nominal_policy(snapshot, args.policy_seed)
    execute = adapter.execute_e(
        snapshot,
        macro,
        policy_seed=args.policy_seed,
        route_seed=0,
        repeat_index=0,
    )

    restore_a = adapter.restore_source_state(snapshot)
    if not restore_a.passed:
        raise RuntimeError("snapshot restore failed before A(0)")
    assist = adapter.execute_a(
        snapshot,
        macro,
        policy_seed=args.policy_seed,
        route_seed=0,
        repeat_index=0,
    )

    e_actions = np.load(execute.action_trace_path)["actions"]
    a_actions = np.load(assist.action_trace_path)["actions"]
    e_states = np.load(execute.state_trace_path)["states"]
    a_states = np.load(assist.state_trace_path)["states"]
    action_error = max_abs_error(e_actions, a_actions)
    state_error = max_abs_error(e_states, a_states)
    passed = (
        action_error <= args.atol
        and state_error <= args.atol
        and execute.success == assist.success
    )
    report = {
        "schema_version": "1.0",
        "passed": passed,
        "atol": args.atol,
        "action_shape_e": list(e_actions.shape),
        "action_shape_a0": list(a_actions.shape),
        "state_shape_e": list(e_states.shape),
        "state_shape_a0": list(a_states.shape),
        "max_abs_action_error": action_error,
        "max_abs_state_error": state_error,
        "success_e": execute.success,
        "success_a0": assist.success,
        "chunk_first_action_error": macro.evidence.max_abs_error,
    }
    report_path = args.output_root / "a_zero_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
