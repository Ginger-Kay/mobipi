from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .collector import PairedBranchCollector, PairedRouteAdapter, RouteCandidate
from .records import RouteRolloutRecord, RouteType, SourceStateRecord


TRANSACTION_SCHEMA_VERSION = "1.0"


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _record_mapping(record: object) -> dict[str, object]:
    return dict(asdict(record))


def _transaction_path(output_root: Path, source_index: int) -> Path:
    if source_index < 0:
        raise ValueError("source_index must be non-negative")
    return output_root / "transactions" / f"source-{source_index:06d}.json"


def _validate_transaction(row: Mapping[str, Any], path: Path) -> tuple[int, SourceStateRecord, list[RouteRolloutRecord]]:
    if str(row.get("schema_version")) != TRANSACTION_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported transaction schema")
    source_index = int(row["source_index"])
    source = SourceStateRecord.from_mapping(row["source_state"])
    rollouts = [RouteRolloutRecord.from_mapping(item) for item in row["rollouts"]]
    observed_routes = {rollout.route_type for rollout in rollouts}
    required_routes = {RouteType.EXECUTE, RouteType.DOCK, RouteType.ASSIST}
    if not rollouts or not required_routes.issubset(observed_routes):
        raise ValueError(f"{path}: transaction must contain E, D, and A")
    if any(item.source_state_id != source.source_state_id for item in rollouts):
        raise ValueError(f"{path}: rollout source_state_id mismatch")
    return source_index, source, rollouts


def load_transactions(output_root: Path) -> dict[int, tuple[SourceStateRecord, list[RouteRolloutRecord]]]:
    transactions: dict[int, tuple[SourceStateRecord, list[RouteRolloutRecord]]] = {}
    transaction_dir = output_root / "transactions"
    if not transaction_dir.exists():
        return transactions
    for path in sorted(transaction_dir.glob("source-*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            source_index, source, rollouts = _validate_transaction(row, path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid collection transaction {path}: {exc}") from exc
        if source_index in transactions:
            raise ValueError(f"duplicate source_index {source_index}")
        transactions[source_index] = source, rollouts
    return transactions


def commit_transaction(
    output_root: Path,
    *,
    source_index: int,
    source: SourceStateRecord,
    rollouts: list[RouteRolloutRecord],
) -> bool:
    source.validate()
    for rollout in rollouts:
        rollout.validate()
    transaction = {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "source_index": source_index,
        "source_state": _record_mapping(source),
        "rollouts": [_record_mapping(row) for row in rollouts],
    }
    content = _canonical_json(transaction)
    path = _transaction_path(output_root, source_index)
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return False
        raise FileExistsError(f"refusing to replace a different transaction: {path}")
    _atomic_write(path, content)
    return True


def materialize_manifests(output_root: Path) -> tuple[Path, Path]:
    transactions = load_transactions(output_root)
    sources: list[SourceStateRecord] = []
    rollouts: list[RouteRolloutRecord] = []
    source_ids: set[str] = set()
    for source_index in sorted(transactions):
        source, rows = transactions[source_index]
        if source.source_state_id in source_ids:
            raise ValueError(f"duplicate source_state_id: {source.source_state_id}")
        source_ids.add(source.source_state_id)
        sources.append(source)
        rollouts.extend(rows)

    manifest_dir = output_root / "manifests"
    source_path = manifest_dir / "source_states.jsonl"
    rollout_path = manifest_dir / "route_rollouts.jsonl"
    _atomic_write(
        source_path,
        "".join(json.dumps(_record_mapping(row), sort_keys=True) + "\n" for row in sources),
    )
    _atomic_write(
        rollout_path,
        "".join(json.dumps(_record_mapping(row), sort_keys=True) + "\n" for row in rollouts),
    )
    return source_path, rollout_path


def run_collection(
    adapter: PairedRouteAdapter,
    *,
    output_root: Path,
    start_index: int,
    source_count: int,
    repeats_per_route: int = 1,
    environment_seed_start: int = 0,
    policy_seed_start: int = 0,
    route_seed_start: int = 0,
) -> tuple[Path, Path]:
    if min(start_index, environment_seed_start, policy_seed_start, route_seed_start) < 0:
        raise ValueError("indices and seeds must be non-negative")
    if source_count <= 0 or repeats_per_route <= 0:
        raise ValueError("source_count and repeats_per_route must be positive")

    completed = load_transactions(output_root)
    collector = PairedBranchCollector(adapter)
    for offset in range(source_count):
        source_index = start_index + offset
        environment_seed = environment_seed_start + source_index
        if source_index in completed:
            source, rows = completed[source_index]
            if source.environment_seed != environment_seed:
                raise RuntimeError(
                    f"resume seed mismatch for source_index {source_index}: "
                    f"expected environment seed {environment_seed}, observed "
                    f"{source.environment_seed}"
                )
            expected_rows = 3 * repeats_per_route
            if len(rows) != expected_rows:
                raise RuntimeError(
                    f"resume repeat mismatch for source_index {source_index}: "
                    f"expected {expected_rows} rollouts, observed {len(rows)}"
                )
            for row in rows:
                seed_offset = source_index * repeats_per_route + row.repeat_index
                expected_policy_seed = policy_seed_start + seed_offset
                expected_route_seed = route_seed_start + seed_offset
                if (
                    row.policy_seed != expected_policy_seed
                    or row.route_seed != expected_route_seed
                ):
                    raise RuntimeError(
                        f"resume policy/route seed mismatch for source_index {source_index}"
                    )
            continue
        adapter.prepare_source_state(source_index, environment_seed)
        snapshot = adapter.capture_source_state()
        if snapshot.record.environment_seed != environment_seed:
            raise RuntimeError(
                "adapter source environment_seed does not match the requested seed"
            )

        rows: list[RouteRolloutRecord] = []
        for repeat_index in range(repeats_per_route):
            seed_offset = source_index * repeats_per_route + repeat_index
            rows.extend(
                collector.collect_from_snapshot(
                    snapshot,
                    policy_seed=policy_seed_start + seed_offset,
                    route_seed=route_seed_start + seed_offset,
                    repeat_index=repeat_index,
                )
            )
        commit_transaction(
            output_root,
            source_index=source_index,
            source=snapshot.record,
            rollouts=rows,
        )
        materialize_manifests(output_root)
    return materialize_manifests(output_root)


def _parse_candidates(
    rows: object,
    *,
    route: RouteType,
) -> tuple[RouteCandidate, ...]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"candidate grid requires a non-empty {route.value} list")
    candidates: list[RouteCandidate] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{route.value} candidate {index} must be an object")
        candidate = RouteCandidate(
            candidate_id=str(row.get("candidate_id", "")),
            candidate_params=dict(row.get("candidate_params", {})),
        )
        candidate.validate_for(route)
        candidates.append(candidate)
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError(f"duplicate {route.value} candidate_id")
    return tuple(candidates)


def run_candidate_grid_collection(
    adapter: PairedRouteAdapter,
    *,
    output_root: Path,
    source_indices: Sequence[int],
    candidate_grid: Mapping[str, Any],
    seeds_per_candidate_override: int | None = None,
    seed_stride_per_source: int | None = None,
    environment_seed_start: int = 0,
    policy_seed_start: int = 0,
    route_seed_start: int = 0,
) -> tuple[Path, Path]:
    if min(environment_seed_start, policy_seed_start, route_seed_start) < 0:
        raise ValueError("seeds must be non-negative")
    indices = [int(index) for index in source_indices]
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise ValueError("source_indices must be non-empty, unique, and non-negative")

    seeds_per_candidate = int(
        candidate_grid.get("seeds_per_candidate", 0)
        if seeds_per_candidate_override is None
        else seeds_per_candidate_override
    )
    if seeds_per_candidate <= 0:
        raise ValueError("seeds_per_candidate must be positive")
    seed_stride = (
        seeds_per_candidate
        if seed_stride_per_source is None
        else int(seed_stride_per_source)
    )
    if seed_stride < seeds_per_candidate:
        raise ValueError(
            "seed_stride_per_source must be at least seeds_per_candidate"
        )
    schedule_seed = int(candidate_grid.get("schedule_seed", -1))
    if schedule_seed < 0:
        raise ValueError("candidate grid requires a non-negative schedule_seed")
    dock_candidates = _parse_candidates(
        candidate_grid.get("dock_candidates"),
        route=RouteType.DOCK,
    )
    assist_candidates = _parse_candidates(
        candidate_grid.get("assist_candidates"),
        route=RouteType.ASSIST,
    )
    expected_rows = seeds_per_candidate * (
        1 + len(dock_candidates) + len(assist_candidates)
    )

    completed = load_transactions(output_root)
    collector = PairedBranchCollector(adapter)
    for source_index in indices:
        environment_seed = environment_seed_start + source_index
        source_policy_seed = policy_seed_start + source_index * seed_stride
        source_route_seed = route_seed_start + source_index * seed_stride
        if source_index in completed:
            source, rows = completed[source_index]
            if source.environment_seed != environment_seed:
                raise RuntimeError(
                    f"resume seed mismatch for source_index {source_index}"
                )
            if len(rows) != expected_rows:
                raise RuntimeError(
                    f"resume candidate-grid mismatch for source_index {source_index}: "
                    f"expected {expected_rows} rollouts, observed {len(rows)}"
                )
            continue

        adapter.prepare_source_state(source_index, environment_seed)
        snapshot = adapter.capture_source_state()
        if snapshot.record.environment_seed != environment_seed:
            raise RuntimeError(
                "adapter source environment_seed does not match the requested seed"
            )
        rows = list(
            collector.collect_candidate_grid(
                snapshot,
                policy_seed_start=source_policy_seed,
                route_seed_start=source_route_seed,
                seeds_per_candidate=seeds_per_candidate,
                dock_candidates=dock_candidates,
                assist_candidates=assist_candidates,
                schedule_seed=schedule_seed,
            )
        )
        commit_transaction(
            output_root,
            source_index=source_index,
            source=snapshot.record,
            rollouts=rows,
        )
        materialize_manifests(output_root)
    return materialize_manifests(output_root)


def _load_factory(specification: str) -> object:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("adapter factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect paired E/D/A source-state branches")
    parser.add_argument("--adapter-factory", required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--source-count", type=int, required=True)
    parser.add_argument("--repeats-per-route", type=int, default=1)
    parser.add_argument("--environment-seed-start", type=int, default=0)
    parser.add_argument("--policy-seed-start", type=int, default=0)
    parser.add_argument("--route-seed-start", type=int, default=0)
    parser.add_argument("--candidate-grid-config", type=Path)
    parser.add_argument(
        "--seeds-per-candidate-override",
        type=int,
        help="split-frozen C2 repeat count; candidate parameters still come from the immutable grid",
    )
    parser.add_argument(
        "--seed-stride-per-source",
        type=int,
        help="fixed source seed stride; C2 uses 3 for every split",
    )
    parser.add_argument("--source-indices", type=Path)
    args = parser.parse_args()

    config = json.loads(args.adapter_config.read_text(encoding="utf-8"))
    factory = _load_factory(args.adapter_factory)
    adapter = factory(output_root=args.output_root, config=config)  # type: ignore[operator]
    if args.candidate_grid_config is None:
        if args.seeds_per_candidate_override is not None:
            parser.error(
                "--seeds-per-candidate-override requires --candidate-grid-config"
            )
        if args.seed_stride_per_source is not None:
            parser.error("--seed-stride-per-source requires --candidate-grid-config")
        if args.source_indices is not None:
            parser.error("--source-indices requires --candidate-grid-config")
        run_collection(
            adapter,
            output_root=args.output_root,
            start_index=args.start_index,
            source_count=args.source_count,
            repeats_per_route=args.repeats_per_route,
            environment_seed_start=args.environment_seed_start,
            policy_seed_start=args.policy_seed_start,
            route_seed_start=args.route_seed_start,
        )
    else:
        candidate_grid = json.loads(
            args.candidate_grid_config.read_text(encoding="utf-8")
        )
        if args.source_indices is None:
            source_indices = list(
                range(args.start_index, args.start_index + args.source_count)
            )
        else:
            source_indices = json.loads(args.source_indices.read_text(encoding="utf-8"))
            if not isinstance(source_indices, list):
                parser.error("--source-indices must contain a JSON list")
            if len(source_indices) != args.source_count:
                parser.error("--source-count must match --source-indices length")
        run_candidate_grid_collection(
            adapter,
            output_root=args.output_root,
            source_indices=source_indices,
            candidate_grid=candidate_grid,
            seeds_per_candidate_override=args.seeds_per_candidate_override,
            seed_stride_per_source=args.seed_stride_per_source,
            environment_seed_start=args.environment_seed_start,
            policy_seed_start=args.policy_seed_start,
            route_seed_start=args.route_seed_start,
        )


if __name__ == "__main__":
    main()
