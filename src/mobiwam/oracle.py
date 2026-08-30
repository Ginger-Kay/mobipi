from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

from .records import CandidateOutcome, RouteType


DEPLOYABLE_ROUTES = (RouteType.EXECUTE, RouteType.DOCK, RouteType.ASSIST)


@dataclass(frozen=True)
class CandidateScore:
    snapshot_id: str
    route_type: RouteType
    candidate_id: str
    repeat_count: int
    success_rate: float
    unsafe_rate: float
    contact_loss_rate: float
    mean_completion_time_s: float
    mean_base_path_m: float


def aggregate_candidates(
    outcomes: Iterable[CandidateOutcome],
) -> list[CandidateScore]:
    grouped: dict[tuple[str, RouteType, str], list[CandidateOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[(outcome.snapshot_id, outcome.route_type, outcome.candidate_id)].append(
            outcome
        )

    scores: list[CandidateScore] = []
    for (snapshot_id, route_type, candidate_id), repeats in sorted(
        grouped.items(), key=lambda item: tuple(str(part) for part in item[0])
    ):
        stage_eligible = [r for r in repeats if r.stage_eligible]
        if not any(r.hard_valid for r in stage_eligible):
            continue
        scores.append(
            CandidateScore(
                snapshot_id=snapshot_id,
                route_type=route_type,
                candidate_id=candidate_id,
                repeat_count=len(stage_eligible),
                success_rate=fmean(float(r.success) for r in stage_eligible),
                unsafe_rate=fmean(float(r.unsafe) for r in stage_eligible),
                contact_loss_rate=fmean(
                    float(r.contact_loss) for r in stage_eligible
                ),
                mean_completion_time_s=fmean(
                    r.completion_time_s for r in stage_eligible
                ),
                mean_base_path_m=fmean(r.base_path_m for r in stage_eligible),
            )
        )
    return scores


def _preference_key(score: CandidateScore) -> tuple[float, float, float, float, float, str]:
    return (
        score.unsafe_rate,
        -score.success_rate,
        score.contact_loss_rate,
        score.mean_completion_time_s,
        score.mean_base_path_m,
        score.candidate_id,
    )


def select_within_route(scores: Iterable[CandidateScore]) -> list[CandidateScore]:
    grouped: dict[tuple[str, RouteType], list[CandidateScore]] = defaultdict(list)
    for score in scores:
        if score.route_type in DEPLOYABLE_ROUTES:
            grouped[(score.snapshot_id, score.route_type)].append(score)
    return [
        min(candidates, key=_preference_key)
        for _, candidates in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def select_route_oracle(within_route: Iterable[CandidateScore]) -> dict[str, CandidateScore]:
    grouped: dict[str, list[CandidateScore]] = defaultdict(list)
    for score in within_route:
        grouped[score.snapshot_id].append(score)
    return {
        snapshot_id: min(candidates, key=_preference_key)
        for snapshot_id, candidates in sorted(grouped.items())
    }


def choose_best_fixed_route(
    within_route: Iterable[CandidateScore], train_snapshot_ids: Sequence[str]
) -> RouteType:
    train_ids = set(train_snapshot_ids)
    if not train_ids:
        raise ValueError("train_snapshot_ids must not be empty")
    by_route: dict[RouteType, list[CandidateScore]] = defaultdict(list)
    for score in within_route:
        if score.snapshot_id in train_ids:
            by_route[score.route_type].append(score)
    if not by_route:
        raise ValueError("no route scores match the training snapshot ids")

    expected_states = len(train_ids)

    def fixed_key(item: tuple[RouteType, list[CandidateScore]]) -> tuple[float, ...]:
        route, scores = item
        support_penalty = expected_states - len({s.snapshot_id for s in scores})
        return (
            float(support_penalty),
            fmean(s.unsafe_rate for s in scores),
            -fmean(s.success_rate for s in scores),
            fmean(s.contact_loss_rate for s in scores),
            fmean(s.mean_completion_time_s for s in scores),
            fmean(s.mean_base_path_m for s in scores),
            float(DEPLOYABLE_ROUTES.index(route)),
        )

    return min(by_route.items(), key=fixed_key)[0]


def evaluate_fixed_route(
    within_route: Iterable[CandidateScore],
    route: RouteType,
    test_snapshot_ids: Sequence[str],
) -> dict[str, CandidateScore]:
    test_ids = set(test_snapshot_ids)
    return {
        score.snapshot_id: score
        for score in within_route
        if score.route_type == route and score.snapshot_id in test_ids
    }


def load_jsonl(path: Path) -> list[CandidateOutcome]:
    rows: list[CandidateOutcome] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(CandidateOutcome.from_mapping(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def write_oracle(path: Path, oracle: dict[str, CandidateScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for snapshot_id, score in sorted(oracle.items()):
            row = asdict(score)
            row["snapshot_id"] = snapshot_id
            row["route_type"] = score.route_type.value
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the paired route oracle")
    parser.add_argument("--input", type=Path, required=True, help="candidate outcomes JSONL")
    parser.add_argument("--output", type=Path, required=True, help="oracle JSONL")
    args = parser.parse_args()

    outcomes = load_jsonl(args.input)
    scores = aggregate_candidates(outcomes)
    within_route = select_within_route(scores)
    oracle = select_route_oracle(within_route)
    write_oracle(args.output, oracle)


if __name__ == "__main__":
    main()
