#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mobiwam.adapters.mobipi import (
    VOLATILE_EP_META_KEYS,
    _observation_hash,
    _state_hash,
    create_adapter,
)
from mobiwam.mobipi_actions import lock_base


DEFAULT_IMAGE_MAX_ABS_ERROR = 1.0 / 255.0 + 1e-6
DEFAULT_IMAGE_MAX_CHANGED_FRACTION = 1e-4


def parse_source_indices(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise ValueError("source indices must be a non-empty list of non-negative integers")
    if len(set(values)) != len(values):
        raise ValueError("source indices must be unique")
    return values


def probe_outcome(
    adapter: Any, action: Any
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    _, reward, done, _ = adapter.env.step(action)
    observation = {
        str(key): np.asarray(value).copy()
        for key, value in adapter._stacked_observation().items()
    }
    return {
        "state_hash": _state_hash(adapter.env.get_state()),
        "observation_hash": _observation_hash(observation),
        "reward": float(reward),
        "done": bool(done),
        "success": adapter._is_success(),
        "task_progress": adapter._task_progress(),
        "base_collision": adapter._base_collision(),
    }, observation


def compare_observations(
    reference: dict[str, np.ndarray], current: dict[str, np.ndarray]
) -> dict[str, object]:
    reference_keys = set(reference)
    current_keys = set(current)
    keys_match = reference_keys == current_keys
    per_key: dict[str, object] = {}
    all_exact = keys_match
    for key in sorted(reference_keys | current_keys):
        if key not in reference or key not in current:
            per_key[key] = {
                "present_in_reference": key in reference,
                "present_in_current": key in current,
                "exact_equal": False,
            }
            all_exact = False
            continue
        expected = np.asarray(reference[key])
        observed = np.asarray(current[key])
        shape_match = expected.shape == observed.shape
        dtype_match = expected.dtype == observed.dtype
        exact_equal = shape_match and dtype_match and np.array_equal(expected, observed)
        result: dict[str, object] = {
            "shape": list(observed.shape),
            "dtype": observed.dtype.str,
            "shape_match": shape_match,
            "dtype_match": dtype_match,
            "exact_equal": exact_equal,
        }
        if shape_match and np.issubdtype(expected.dtype, np.number) and np.issubdtype(
            observed.dtype, np.number
        ):
            difference = np.abs(
                observed.astype(np.float64) - expected.astype(np.float64)
            )
            changed = difference != 0
            result.update(
                {
                    "max_abs_error": float(difference.max(initial=0.0)),
                    "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
                    "changed_elements": int(changed.sum()),
                    "changed_fraction": float(changed.mean()) if changed.size else 0.0,
                }
            )
        per_key[key] = result
        all_exact = all_exact and exact_equal
    return {
        "keys_match": keys_match,
        "exact_equal": all_exact,
        "per_key": per_key,
    }


def compare_state_vectors(reference: np.ndarray, current: np.ndarray) -> dict[str, object]:
    expected = np.asarray(reference)
    observed = np.asarray(current)
    shape_match = expected.shape == observed.shape
    dtype_match = expected.dtype == observed.dtype
    result: dict[str, object] = {
        "expected_shape": list(expected.shape),
        "observed_shape": list(observed.shape),
        "expected_dtype": expected.dtype.str,
        "observed_dtype": observed.dtype.str,
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "exact_equal": shape_match and dtype_match and np.array_equal(expected, observed),
    }
    if not shape_match or not np.issubdtype(expected.dtype, np.number) or not np.issubdtype(
        observed.dtype, np.number
    ):
        return result
    difference = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
    changed_indices = np.flatnonzero(difference.reshape(-1) != 0)
    largest_indices = sorted(
        changed_indices.tolist(),
        key=lambda index: float(difference.reshape(-1)[index]),
        reverse=True,
    )[:16]
    expected_flat = expected.reshape(-1)
    observed_flat = observed.reshape(-1)
    result.update(
        {
            "max_abs_error": float(difference.max(initial=0.0)),
            "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
            "changed_elements": int(changed_indices.size),
            "changed_fraction": (
                float(changed_indices.size / difference.size) if difference.size else 0.0
            ),
            "largest_differences": [
                {
                    "flat_index": int(index),
                    "expected": float(expected_flat[index]),
                    "observed": float(observed_flat[index]),
                    "abs_error": float(difference.reshape(-1)[index]),
                }
                for index in largest_indices
            ],
        }
    )
    return result


def compare_state_metadata(reference: dict[str, Any], current: dict[str, Any]) -> dict[str, object]:
    expected_model = str(reference.get("model", ""))
    observed_model = str(current.get("model", ""))
    first_model_difference = next(
        (
            index
            for index, (expected, observed) in enumerate(
                zip(expected_model, observed_model, strict=False)
            )
            if expected != observed
        ),
        min(len(expected_model), len(observed_model))
        if expected_model != observed_model
        else None,
    )

    def stable_ep_meta(state: dict[str, Any]) -> dict[str, Any]:
        raw = state.get("ep_meta", {})
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise TypeError("episode metadata must be a JSON object")
        return {
            str(key): value
            for key, value in parsed.items()
            if key not in VOLATILE_EP_META_KEYS
        }

    expected_meta = stable_ep_meta(reference)
    observed_meta = stable_ep_meta(current)
    differing_meta_keys = sorted(
        key
        for key in set(expected_meta) | set(observed_meta)
        if expected_meta.get(key) != observed_meta.get(key)
    )
    result: dict[str, object] = {
        "model_equal": expected_model == observed_model,
        "expected_model_sha256": hashlib.sha256(expected_model.encode()).hexdigest(),
        "observed_model_sha256": hashlib.sha256(observed_model.encode()).hexdigest(),
        "expected_model_length": len(expected_model),
        "observed_model_length": len(observed_model),
        "first_model_difference": first_model_difference,
        "stable_ep_meta_equal": expected_meta == observed_meta,
        "differing_stable_ep_meta_keys": differing_meta_keys,
        "stable_ep_meta_differences": {
            key: {
                "expected": expected_meta.get(key),
                "observed": observed_meta.get(key),
            }
            for key in differing_meta_keys
        },
    }
    if first_model_difference is not None:
        start = max(0, first_model_difference - 120)
        end = first_model_difference + 120
        result["expected_model_snippet"] = expected_model[start:end]
        result["observed_model_snippet"] = observed_model[start:end]
    return result


def observation_within_tolerance(
    comparison: dict[str, object],
    *,
    image_max_abs_error: float,
    image_max_changed_fraction: float,
) -> bool:
    if not comparison["keys_match"]:
        return False
    per_key = comparison["per_key"]
    if not isinstance(per_key, dict):
        return False
    for key, raw_metrics in per_key.items():
        if not isinstance(raw_metrics, dict):
            return False
        if not raw_metrics.get("shape_match") or not raw_metrics.get("dtype_match"):
            return False
        if raw_metrics.get("exact_equal"):
            continue
        if not str(key).endswith("_image"):
            return False
        max_abs_error = raw_metrics.get("max_abs_error")
        changed_fraction = raw_metrics.get("changed_fraction")
        if not isinstance(max_abs_error, (float, int)) or not isinstance(
            changed_fraction, (float, int)
        ):
            return False
        if max_abs_error > image_max_abs_error:
            return False
        if changed_fraction > image_max_changed_fraction:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit repeated Mobi-pi snapshot restores and one-step outcomes"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-indices", default="0,5,10,15,20")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--policy-seed-start", type=int, default=1000)
    parser.add_argument(
        "--image-max-abs-error",
        type=float,
        default=DEFAULT_IMAGE_MAX_ABS_ERROR,
    )
    parser.add_argument(
        "--image-max-changed-fraction",
        type=float,
        default=DEFAULT_IMAGE_MAX_CHANGED_FRACTION,
    )
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("repeats must be at least two")
    if args.image_max_abs_error < 0:
        raise ValueError("image max absolute error must be non-negative")
    if not 0 <= args.image_max_changed_fraction <= 1:
        raise ValueError("image max changed fraction must be in [0, 1]")

    source_indices = parse_source_indices(args.source_indices)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    adapter = create_adapter(output_root=args.output_root, config=config)
    sources: list[dict[str, object]] = []

    for source_index in source_indices:
        adapter.prepare_source_state(source_index, source_index)
        snapshot = adapter.capture_source_state()
        macro = adapter.sample_nominal_policy(
            snapshot, args.policy_seed_start + source_index
        )
        probe_action = lock_base(macro.chunk[0])
        repeats: list[dict[str, object]] = []
        reference: dict[str, object] | None = None
        reference_observation: dict[str, np.ndarray] | None = None

        for repeat_index in range(args.repeats):
            restore = adapter.restore_source_state(snapshot)
            expected_env_state = snapshot.opaque_handle.env_state
            observed_env_state = adapter.env.get_state()
            state_comparison = compare_state_vectors(
                np.asarray(expected_env_state["states"]),
                np.asarray(observed_env_state["states"]),
            )
            metadata_comparison = compare_state_metadata(
                expected_env_state,
                observed_env_state,
            )
            restore_matches = (
                restore.passed
                and restore.snapshot_hash == snapshot.record.snapshot_hash
                and restore.observation_hash == snapshot.record.observation_hash
            )
            if restore_matches:
                outcome, observation = probe_outcome(adapter, probe_action)
            else:
                outcome, observation = None, None
            if reference is None and outcome is not None and observation is not None:
                reference = outcome
                reference_observation = observation
            observation_comparison = (
                compare_observations(reference_observation, observation)
                if reference_observation is not None and observation is not None
                else None
            )
            core_matches = (
                outcome is not None
                and reference is not None
                and all(
                    outcome[key] == reference[key]
                    for key in outcome
                    if key != "observation_hash"
                )
            )
            exact_probe_match = outcome == reference
            observation_tolerated = (
                observation_comparison is not None
                and observation_within_tolerance(
                    observation_comparison,
                    image_max_abs_error=args.image_max_abs_error,
                    image_max_changed_fraction=args.image_max_changed_fraction,
                )
            )
            probe_within_tolerance = core_matches and observation_tolerated
            repeat_passed = restore_matches and probe_within_tolerance
            repeats.append(
                {
                    "repeat_index": repeat_index,
                    "restore_passed": restore.passed,
                    "restore_matches_snapshot": restore_matches,
                    "snapshot_hash": restore.snapshot_hash,
                    "observation_hash": restore.observation_hash,
                    "restore_state_comparison": state_comparison,
                    "restore_metadata_comparison": metadata_comparison,
                    "probe_outcome": outcome,
                    "core_matches_first_probe": core_matches,
                    "observation_comparison_to_first": observation_comparison,
                    "observation_within_tolerance": observation_tolerated,
                    "matches_first_probe": exact_probe_match,
                    "probe_within_tolerance": probe_within_tolerance,
                    "passed": repeat_passed,
                }
            )

        sources.append(
            {
                "source_index": source_index,
                "source_state_id": snapshot.record.source_state_id,
                "environment_seed": snapshot.record.environment_seed,
                "repeats": repeats,
                "passed": all(bool(item["passed"]) for item in repeats),
            }
        )

    close = getattr(adapter.env, "close", None)
    if callable(close):
        close()
    report = {
        "schema_version": "1.1",
        "source_indices": source_indices,
        "repeats_per_source": args.repeats,
        "probe": "one frozen-policy action with base command zeroed",
        "observation_tolerance": {
            "non_image_keys": "exact",
            "image_max_abs_error": args.image_max_abs_error,
            "image_max_changed_fraction": args.image_max_changed_fraction,
        },
        "sources": sources,
        "passed": all(bool(source["passed"]) for source in sources),
    }
    report_path = args.output_root / "restore_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
