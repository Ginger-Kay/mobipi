from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .records import RouteType, Stage


@dataclass(frozen=True)
class CandidateFeatures:
    candidate_id: str
    route_type: RouteType
    hard_valid: bool
    visibility: float
    reachability: float
    joint_margin: float
    intent_error: float
    base_cost: float
    policy_compatibility: float = 0.0


def eligible(candidates: Iterable[CandidateFeatures], stage: Stage) -> list[CandidateFeatures]:
    return [
        candidate
        for candidate in candidates
        if candidate.hard_valid
        and not (stage is Stage.CONTACT and candidate.route_type is RouteType.DOCK)
    ]


def select_stage_rule(candidates: Iterable[CandidateFeatures], stage: Stage) -> CandidateFeatures:
    supported = eligible(candidates, stage)
    if not supported:
        raise ValueError("no eligible candidate; caller must derive X")
    return max(
        supported,
        key=lambda candidate: (
            candidate.joint_margin - candidate.intent_error,
            -candidate.base_cost,
            candidate.candidate_id,
        ),
    )


def select_geometry(candidates: Iterable[CandidateFeatures], stage: Stage) -> CandidateFeatures:
    supported = eligible(candidates, stage)
    if not supported:
        raise ValueError("no eligible candidate; caller must derive X")
    return max(
        supported,
        key=lambda candidate: (
            candidate.visibility
            + candidate.reachability
            + candidate.joint_margin
            - candidate.intent_error
            - 0.1 * candidate.base_cost,
            candidate.candidate_id,
        ),
    )


def select_mobipi_d_scorer(candidates: Iterable[CandidateFeatures], stage: Stage) -> CandidateFeatures:
    supported = eligible(candidates, stage)
    if not supported:
        raise ValueError("no eligible candidate; caller must derive X")
    docking = [candidate for candidate in supported if candidate.route_type is RouteType.DOCK]
    pool = docking or supported
    return max(
        pool,
        key=lambda candidate: (
            candidate.policy_compatibility + candidate.visibility,
            candidate.reachability,
            -candidate.base_cost,
            candidate.candidate_id,
        ),
    )
