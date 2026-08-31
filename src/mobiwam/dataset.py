from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .records import (
    DataSplit,
    RouteRolloutRecord,
    RouteType,
    SourceStateRecord,
    Stage,
)


EXECUTABLE_ROUTES = (RouteType.EXECUTE, RouteType.DOCK, RouteType.ASSIST)


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    source_state_id: str | None = None


@dataclass(frozen=True)
class PairedCollectionReport:
    expected_source_states: int
    observed_source_states: int
    rollout_records: int
    route_counts: Mapping[str, int]
    split_counts: Mapping[str, int]
    derived_abstain_targets: int
    issues: Sequence[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": "pass" if self.ok else "fail",
            "expected_source_states": self.expected_source_states,
            "observed_source_states": self.observed_source_states,
            "rollout_records": self.rollout_records,
            "route_counts": dict(self.route_counts),
            "split_counts": dict(self.split_counts),
            "derived_abstain_targets": self.derived_abstain_targets,
            "issues": [asdict(issue) for issue in self.issues],
        }


def assign_group_split(
    source_state_id: str,
    *,
    seed: int = 0,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.15,
    calibration_fraction: float = 0.10,
) -> DataSplit:
    if not source_state_id:
        raise ValueError("source_state_id must not be empty")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in [0, 1)")
    if train_fraction + validation_fraction + calibration_fraction >= 1.0:
        raise ValueError("train, validation, and calibration fractions must sum to less than 1")

    digest = hashlib.sha256(f"{seed}:{source_state_id}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    if unit < train_fraction:
        return DataSplit.TRAIN
    if unit < train_fraction + validation_fraction:
        return DataSplit.VALIDATION
    if unit < train_fraction + validation_fraction + calibration_fraction:
        return DataSplit.CALIBRATION
    return DataSplit.LOCKED_TEST


def derive_abstain_targets(
    rollouts: Iterable[RouteRolloutRecord],
) -> dict[str, bool]:
    grouped: dict[str, list[RouteRolloutRecord]] = {}
    for rollout in rollouts:
        grouped.setdefault(rollout.source_state_id, []).append(rollout)
    return {
        source_state_id: not any(
            row.stage_eligible
            and row.hard_valid
            and row.restore_check_passed
            and row.success
            and not row.unsafe
            for row in rows
        )
        for source_state_id, rows in grouped.items()
    }


def validate_paired_collection(
    source_states: Sequence[SourceStateRecord],
    rollouts: Sequence[RouteRolloutRecord],
    *,
    expected_source_states: int = 100,
    repeats_per_route: int = 1,
    expected_repeats_by_route: Mapping[RouteType, int] | None = None,
    expected_candidates_by_route: Mapping[RouteType, Sequence[str]] | None = None,
    expected_source_splits: Mapping[str, DataSplit | str] | None = None,
    require_precontact: bool = True,
) -> PairedCollectionReport:
    if expected_source_states <= 0:
        raise ValueError("expected_source_states must be positive")
    if repeats_per_route <= 0:
        raise ValueError("repeats_per_route must be positive")
    route_repeats = (
        {route: repeats_per_route for route in EXECUTABLE_ROUTES}
        if expected_repeats_by_route is None
        else {
            route: int(expected_repeats_by_route.get(route, 0))
            for route in EXECUTABLE_ROUTES
        }
    )
    if min(route_repeats.values()) <= 0:
        raise ValueError("every executable route must have a positive repeat count")

    issues: list[ValidationIssue] = []
    sources: dict[str, SourceStateRecord] = {}
    normalized_source_splits = (
        None
        if expected_source_splits is None
        else {
            str(source_state_id): (
                split if isinstance(split, DataSplit) else DataSplit(str(split))
            )
            for source_state_id, split in expected_source_splits.items()
        }
    )
    for source in source_states:
        source.validate()
        if source.source_state_id in sources:
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_source_state",
                    "source_state_id appears more than once",
                    source.source_state_id,
                )
            )
        sources[source.source_state_id] = source
        expected_split = (
            assign_group_split(source.source_state_id)
            if normalized_source_splits is None
            else normalized_source_splits.get(source.source_state_id)
        )
        if expected_split is None:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_expected_source_split",
                    "source_state_id is absent from the frozen split map",
                    source.source_state_id,
                )
            )
        elif source.split is not expected_split:
            issues.append(
                ValidationIssue(
                    "error",
                    "noncanonical_group_split",
                    f"expected split {expected_split.value}, observed {source.split.value}",
                    source.source_state_id,
                )
            )
        if require_precontact and source.stage is not Stage.PRECONTACT:
            issues.append(
                ValidationIssue(
                    "error",
                    "pilot_requires_precontact",
                    "the 100 x E/D/A pilot requires precontact source states",
                    source.source_state_id,
                )
            )

    if normalized_source_splits is not None:
        unexpected_split_ids = sorted(set(normalized_source_splits).difference(sources))
        if unexpected_split_ids:
            issues.append(
                ValidationIssue(
                    "error",
                    "unexpected_source_split_map_entries",
                    "frozen split map contains source IDs absent from the collection: "
                    + ", ".join(unexpected_split_ids),
                )
            )

    if len(sources) != expected_source_states:
        issues.append(
            ValidationIssue(
                "error",
                "wrong_source_state_count",
                f"expected {expected_source_states}, observed {len(sources)}",
            )
        )

    route_counts = {route.value: 0 for route in EXECUTABLE_ROUTES}
    split_counts = {split.value: 0 for split in DataSplit}
    grouped: dict[str, dict[tuple[RouteType, int], RouteRolloutRecord]] = {}
    for rollout in rollouts:
        rollout.validate()
        source = sources.get(rollout.source_state_id)
        if source is None:
            issues.append(
                ValidationIssue(
                    "error",
                    "unknown_source_state",
                    "rollout references a source_state_id absent from the manifest",
                    rollout.source_state_id,
                )
            )
            continue

        route_counts[rollout.route_type.value] += 1
        split_counts[rollout.split.value] += 1
        metadata_pairs = {
            "task_id": (source.task_id, rollout.task_id),
            "task_family": (source.task_family, rollout.task_family),
            "episode_id": (source.episode_id, rollout.episode_id),
            "stage": (source.stage, rollout.stage),
            "split": (source.split, rollout.split),
            "environment_seed": (source.environment_seed, rollout.environment_seed),
            "policy_name": (source.policy_name, rollout.policy_name),
            "policy_checkpoint_hash": (
                source.policy_checkpoint_hash,
                rollout.policy_checkpoint_hash,
            ),
            "simulator_version": (source.simulator_version, rollout.simulator_version),
            "code_commit": (source.code_commit, rollout.code_commit),
            "snapshot_hash": (source.snapshot_hash, rollout.snapshot_hash),
            "observation_hash": (source.observation_hash, rollout.observation_hash),
            "snapshot_path": (source.snapshot_path, rollout.source_snapshot_path),
        }
        mismatched = [
            name for name, (expected, observed) in metadata_pairs.items() if expected != observed
        ]
        if mismatched:
            issues.append(
                ValidationIssue(
                    "error",
                    "branch_metadata_mismatch",
                    f"branch differs from source manifest: {', '.join(mismatched)}",
                    rollout.source_state_id,
                )
            )
        if not rollout.restore_check_passed:
            issues.append(
                ValidationIssue(
                    "error",
                    "restore_check_failed",
                    "paired data are invalid when source restore verification fails",
                    rollout.source_state_id,
                )
            )
        if not rollout.transform_check_passed:
            issues.append(
                ValidationIssue(
                    "error",
                    "transform_check_failed",
                    "action/transform closure must pass before collection",
                    rollout.source_state_id,
                )
            )
        if not rollout.labeler_version:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_labeler_version",
                    "every branch must identify the automatic labeler version",
                    rollout.source_state_id,
                )
            )
        if rollout.stage_eligible and rollout.hard_valid:
            required_artifacts = {
                "state_trace_path": rollout.state_trace_path,
                "action_trace_path": rollout.action_trace_path,
            }
            if rollout.schema_version != "1.0":
                required_artifacts["event_trace_path"] = rollout.event_trace_path
            missing_artifacts = sorted(
                name
                for name, value in required_artifacts.items()
                if not value
            )
            if missing_artifacts:
                issues.append(
                    ValidationIssue(
                        "error",
                        "missing_rollout_artifact",
                        f"executed branch is missing: {', '.join(missing_artifacts)}",
                        rollout.source_state_id,
                    )
                )
        if rollout.route_type is RouteType.EXECUTE:
            raw_displacement = rollout.candidate_params.get(
                "max_base_displacement_m", rollout.base_path_length_m
            )
            try:
                execute_displacement = float(raw_displacement)
            except (TypeError, ValueError):
                execute_displacement = float("nan")
            if not isfinite(execute_displacement) or execute_displacement < 0.0:
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_execute_base_displacement",
                        "E maximum base displacement must be finite and non-negative",
                        rollout.source_state_id,
                    )
                )
            elif execute_displacement >= 1e-3:
                issues.append(
                    ValidationIssue(
                        "error",
                        "execute_base_moved",
                        "E must remain strictly below 1 mm from its source base position",
                        rollout.source_state_id,
                    )
                )

        key = (rollout.route_type, rollout.repeat_index)
        branch_group = grouped.setdefault(rollout.source_state_id, {})
        if key in branch_group:
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_route_repeat",
                    f"duplicate {rollout.route_type.value} repeat {rollout.repeat_index}",
                    rollout.source_state_id,
                )
            )
        branch_group[key] = rollout

    expected_keys = {
        (route, repeat_index)
        for route, repeat_count in route_repeats.items()
        for repeat_index in range(repeat_count)
    }
    for source_state_id in sources:
        observed_keys = set(grouped.get(source_state_id, {}))
        missing = sorted(
            f"{route.value}:{repeat_index}"
            for route, repeat_index in expected_keys.difference(observed_keys)
        )
        extra = sorted(
            f"{route.value}:{repeat_index}"
            for route, repeat_index in observed_keys.difference(expected_keys)
        )
        if missing:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_paired_branch",
                    f"missing route repeats: {', '.join(missing)}",
                    source_state_id,
                )
            )
        if extra:
            issues.append(
                ValidationIssue(
                    "error",
                    "unexpected_route_repeat",
                    f"unexpected route repeats: {', '.join(extra)}",
                    source_state_id,
                )
            )
        if expected_candidates_by_route is not None:
            source_rows = list(grouped.get(source_state_id, {}).values())
            for route, expected_candidate_ids in expected_candidates_by_route.items():
                observed_candidate_ids = {
                    row.candidate_id for row in source_rows if row.route_type is route
                }
                expected_ids = set(expected_candidate_ids)
                if observed_candidate_ids != expected_ids:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "candidate_support_mismatch",
                            f"{route.value} expected {sorted(expected_ids)}, observed {sorted(observed_candidate_ids)}",
                            source_state_id,
                        )
                    )
                elif expected_ids and route_repeats[route] % len(expected_ids) == 0:
                    expected_count = route_repeats[route] // len(expected_ids)
                    counts = {
                        candidate_id: sum(
                            row.route_type is route and row.candidate_id == candidate_id
                            for row in source_rows
                        )
                        for candidate_id in expected_ids
                    }
                    if any(count != expected_count for count in counts.values()):
                        issues.append(
                            ValidationIssue(
                                "error",
                                "candidate_repeat_mismatch",
                                f"{route.value} expected {expected_count} repeats each, observed {counts}",
                                source_state_id,
                            )
                        )

    abstain = derive_abstain_targets(rollouts)
    return PairedCollectionReport(
        expected_source_states=expected_source_states,
        observed_source_states=len(sources),
        rollout_records=len(rollouts),
        route_counts=route_counts,
        split_counts=split_counts,
        derived_abstain_targets=sum(abstain.values()),
        issues=tuple(issues),
    )


def _load_jsonl(path: Path, factory: object) -> list[object]:
    records: list[object] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                records.append(factory.from_mapping(row))  # type: ignore[attr-defined]
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paired E/D/A route data")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-source-states", type=int, default=100)
    parser.add_argument("--repeats-per-route", type=int, default=1)
    parser.add_argument("--execute-repeats", type=int)
    parser.add_argument("--dock-repeats", type=int)
    parser.add_argument("--assist-repeats", type=int)
    parser.add_argument(
        "--source-split-map",
        type=Path,
        help="frozen JSON source_state_id to split mapping; defaults to legacy hash assignment",
    )
    parser.add_argument("--allow-contact", action="store_true")
    args = parser.parse_args()

    sources = _load_jsonl(args.sources, SourceStateRecord)
    rollouts = _load_jsonl(args.rollouts, RouteRolloutRecord)
    explicit_repeat_counts = (
        args.execute_repeats,
        args.dock_repeats,
        args.assist_repeats,
    )
    if any(value is not None for value in explicit_repeat_counts) and not all(
        value is not None for value in explicit_repeat_counts
    ):
        parser.error(
            "--execute-repeats, --dock-repeats, and --assist-repeats must be set together"
        )
    expected_repeats_by_route = (
        None
        if all(value is None for value in explicit_repeat_counts)
        else {
            RouteType.EXECUTE: args.execute_repeats,
            RouteType.DOCK: args.dock_repeats,
            RouteType.ASSIST: args.assist_repeats,
        }
    )
    expected_source_splits = None
    if args.source_split_map is not None:
        expected_source_splits = json.loads(
            args.source_split_map.read_text(encoding="utf-8")
        )
        if not isinstance(expected_source_splits, dict):
            parser.error("--source-split-map must contain a JSON object")
    report = validate_paired_collection(
        sources,  # type: ignore[arg-type]
        rollouts,  # type: ignore[arg-type]
        expected_source_states=args.expected_source_states,
        repeats_per_route=args.repeats_per_route,
        expected_repeats_by_route=expected_repeats_by_route,
        expected_source_splits=expected_source_splits,
        require_precontact=not args.allow_contact,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report.ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
