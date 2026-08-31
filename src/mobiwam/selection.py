from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np

from .records import RouteType


@dataclass(frozen=True)
class PredictedCandidate:
    candidate_id: str
    route_type: RouteType
    hard_valid: bool
    success_probabilities: np.ndarray
    risk_probabilities: np.ndarray
    cost: tuple[float, float, float, float, float, float]


def empirical_bound(values: np.ndarray, *, lower: bool, confidence: float = 0.95) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) < 2:
        raise ValueError("empirical ensemble bounds require at least two model seeds")
    if not np.all((0.0 <= array) & (array <= 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    radius = z * float(array.std(ddof=1)) / np.sqrt(len(array))
    mean = float(array.mean())
    return max(0.0, mean - radius) if lower else min(1.0, mean + radius)


def select_minimum_cost_sufficient(
    candidates: Sequence[PredictedCandidate], *, tau_success: float, epsilon_risk: float
) -> PredictedCandidate | RouteType:
    if not 0.0 <= tau_success <= 1.0 or not 0.0 <= epsilon_risk <= 1.0:
        raise ValueError("operating thresholds must lie in [0, 1]")
    admissible = [
        candidate
        for candidate in candidates
        if candidate.hard_valid
        and empirical_bound(candidate.success_probabilities, lower=True) >= tau_success
        and empirical_bound(candidate.risk_probabilities, lower=False) <= epsilon_risk
    ]
    if not admissible:
        return RouteType.ABSTAIN
    return min(admissible, key=lambda candidate: (candidate.cost, candidate.candidate_id))
