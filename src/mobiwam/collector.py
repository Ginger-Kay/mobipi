from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Protocol, Sequence

from .records import RouteRolloutRecord, RouteType, SourceStateRecord


@dataclass(frozen=True)
class SourceSnapshot:
    record: SourceStateRecord
    opaque_handle: Any


@dataclass(frozen=True)
class RestoreEvidence:
    passed: bool
    snapshot_hash: str
    observation_hash: str


@dataclass(frozen=True)
class RouteCandidate:
    candidate_id: str
    candidate_params: Mapping[str, Any]

    def validate_for(self, route: RouteType) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.candidate_id.lower().startswith(route.value.lower()):
            raise ValueError(
                f"{route.value} candidate_id must start with {route.value.lower()}"
            )


class PairedRouteAdapter(Protocol):
    def prepare_source_state(self, source_index: int, environment_seed: int) -> None: ...

    def capture_source_state(self) -> SourceSnapshot: ...

    def restore_source_state(self, snapshot: SourceSnapshot) -> RestoreEvidence: ...

    def sample_nominal_policy(
        self, snapshot: SourceSnapshot, policy_seed: int
    ) -> Any: ...

    def execute_e(
        self,
        snapshot: SourceSnapshot,
        nominal_chunk: Any,
        *,
        policy_seed: int,
        route_seed: int,
        repeat_index: int,
    ) -> RouteRolloutRecord: ...

    def execute_d(
        self,
        snapshot: SourceSnapshot,
        *,
        policy_seed: int,
        route_seed: int,
        repeat_index: int,
        candidate_id: str = "d0",
        candidate_params: Mapping[str, Any] | None = None,
    ) -> RouteRolloutRecord: ...

    def execute_a(
        self,
        snapshot: SourceSnapshot,
        nominal_chunk: Any,
        *,
        policy_seed: int,
        route_seed: int,
        repeat_index: int,
        candidate_id: str = "a0",
        candidate_params: Mapping[str, Any] | None = None,
    ) -> RouteRolloutRecord: ...


class PairedBranchCollector:
    """Collect one E/D/A tuple from exactly one source snapshot.

    E and A receive the same nominal policy sample. D deliberately does not:
    its adapter must complete dock-settle-history-reset and re-query the frozen
    policy from the real post-dock observation.
    """

    def __init__(self, adapter: PairedRouteAdapter):
        self.adapter = adapter

    def _restore_or_fail(self, snapshot: SourceSnapshot, route: RouteType) -> None:
        evidence = self.adapter.restore_source_state(snapshot)
        expected = snapshot.record
        if not evidence.passed:
            raise RuntimeError(f"restore verification failed before route {route.value}")
        if evidence.snapshot_hash != expected.snapshot_hash:
            raise RuntimeError(f"snapshot hash changed before route {route.value}")
        if evidence.observation_hash != expected.observation_hash:
            raise RuntimeError(f"observation hash changed before route {route.value}")

    @staticmethod
    def _validate_branch(
        snapshot: SourceSnapshot,
        branch: RouteRolloutRecord,
        route: RouteType,
        repeat_index: int,
        candidate_id: str | None = None,
    ) -> None:
        branch.validate()
        if branch.source_state_id != snapshot.record.source_state_id:
            raise RuntimeError("adapter returned a branch for a different source state")
        if branch.route_type is not route:
            raise RuntimeError(
                f"adapter returned route {branch.route_type.value}, expected {route.value}"
            )
        if branch.repeat_index != repeat_index:
            raise RuntimeError("adapter returned the wrong repeat_index")
        if candidate_id is not None and branch.candidate_id != candidate_id:
            raise RuntimeError(
                f"adapter returned candidate {branch.candidate_id}, expected {candidate_id}"
            )
        if not branch.restore_check_passed:
            raise RuntimeError("adapter record does not attest the verified restore")
        if route is RouteType.EXECUTE:
            raw_displacement = branch.candidate_params.get(
                "max_base_displacement_m", branch.base_path_length_m
            )
            try:
                displacement = float(raw_displacement)
            except (TypeError, ValueError):
                displacement = float("nan")
            if not isfinite(displacement) or displacement < 0.0 or displacement >= 1e-3:
                raise RuntimeError(
                    "E maximum base displacement must be finite and strictly below 1 mm"
                )

    def collect_from_snapshot(
        self,
        snapshot: SourceSnapshot,
        *,
        policy_seed: int,
        route_seed: int,
        repeat_index: int = 0,
    ) -> tuple[RouteRolloutRecord, ...]:
        if min(policy_seed, route_seed, repeat_index) < 0:
            raise ValueError("seeds and repeat_index must be non-negative")
        snapshot.record.validate()

        self._restore_or_fail(snapshot, RouteType.EXECUTE)
        nominal_chunk = self.adapter.sample_nominal_policy(snapshot, policy_seed)
        execute = self.adapter.execute_e(
            snapshot,
            nominal_chunk,
            policy_seed=policy_seed,
            route_seed=route_seed,
            repeat_index=repeat_index,
        )
        self._validate_branch(snapshot, execute, RouteType.EXECUTE, repeat_index)

        self._restore_or_fail(snapshot, RouteType.DOCK)
        dock = self.adapter.execute_d(
            snapshot,
            policy_seed=policy_seed,
            route_seed=route_seed,
            repeat_index=repeat_index,
        )
        self._validate_branch(snapshot, dock, RouteType.DOCK, repeat_index)

        self._restore_or_fail(snapshot, RouteType.ASSIST)
        assist = self.adapter.execute_a(
            snapshot,
            nominal_chunk,
            policy_seed=policy_seed,
            route_seed=route_seed,
            repeat_index=repeat_index,
        )
        self._validate_branch(snapshot, assist, RouteType.ASSIST, repeat_index)

        return execute, dock, assist

    def collect_candidate_grid(
        self,
        snapshot: SourceSnapshot,
        *,
        policy_seed_start: int,
        route_seed_start: int,
        seeds_per_candidate: int,
        dock_candidates: Sequence[RouteCandidate],
        assist_candidates: Sequence[RouteCandidate],
    ) -> tuple[RouteRolloutRecord, ...]:
        if min(policy_seed_start, route_seed_start) < 0:
            raise ValueError("seeds must be non-negative")
        if seeds_per_candidate <= 0:
            raise ValueError("seeds_per_candidate must be positive")
        if not dock_candidates or not assist_candidates:
            raise ValueError("candidate grids require at least one D and one A")
        for route, candidates in (
            (RouteType.DOCK, dock_candidates),
            (RouteType.ASSIST, assist_candidates),
        ):
            ids = [candidate.candidate_id for candidate in candidates]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {route.value} candidate_id")
            for candidate in candidates:
                candidate.validate_for(route)

        rows: list[RouteRolloutRecord] = []
        for seed_index in range(seeds_per_candidate):
            policy_seed = policy_seed_start + seed_index
            route_seed = route_seed_start + seed_index

            self._restore_or_fail(snapshot, RouteType.EXECUTE)
            nominal_chunk = self.adapter.sample_nominal_policy(snapshot, policy_seed)
            execute = self.adapter.execute_e(
                snapshot,
                nominal_chunk,
                policy_seed=policy_seed,
                route_seed=route_seed,
                repeat_index=seed_index,
            )
            self._validate_branch(
                snapshot,
                execute,
                RouteType.EXECUTE,
                seed_index,
                "e0",
            )
            rows.append(execute)

            for candidate_index, candidate in enumerate(dock_candidates):
                repeat_index = candidate_index * seeds_per_candidate + seed_index
                self._restore_or_fail(snapshot, RouteType.DOCK)
                dock = self.adapter.execute_d(
                    snapshot,
                    policy_seed=policy_seed,
                    route_seed=route_seed,
                    repeat_index=repeat_index,
                    candidate_id=candidate.candidate_id,
                    candidate_params=candidate.candidate_params,
                )
                self._validate_branch(
                    snapshot,
                    dock,
                    RouteType.DOCK,
                    repeat_index,
                    candidate.candidate_id,
                )
                rows.append(dock)

            for candidate_index, candidate in enumerate(assist_candidates):
                repeat_index = candidate_index * seeds_per_candidate + seed_index
                self._restore_or_fail(snapshot, RouteType.ASSIST)
                assist = self.adapter.execute_a(
                    snapshot,
                    nominal_chunk,
                    policy_seed=policy_seed,
                    route_seed=route_seed,
                    repeat_index=repeat_index,
                    candidate_id=candidate.candidate_id,
                    candidate_params=candidate.candidate_params,
                )
                self._validate_branch(
                    snapshot,
                    assist,
                    RouteType.ASSIST,
                    repeat_index,
                    candidate.candidate_id,
                )
                rows.append(assist)

        return tuple(rows)

    def collect_one(
        self,
        *,
        policy_seed: int,
        route_seed: int,
        repeat_index: int = 0,
    ) -> tuple[SourceStateRecord, tuple[RouteRolloutRecord, ...]]:
        snapshot = self.adapter.capture_source_state()
        rows = self.collect_from_snapshot(
            snapshot,
            policy_seed=policy_seed,
            route_seed=route_seed,
            repeat_index=repeat_index,
        )
        return snapshot.record, rows
