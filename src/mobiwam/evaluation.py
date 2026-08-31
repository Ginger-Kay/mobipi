from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    source_count: int
    resamples: int
    seed: int


def paired_cluster_bootstrap(
    effects: Mapping[str, float],
    *,
    strata: Mapping[str, str] | None = None,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapResult:
    if not effects:
        raise ValueError("effects must contain at least one source state")
    if resamples < 10_000:
        raise ValueError("claim-bearing paired bootstrap requires at least 10,000 resamples")
    if strata is None:
        strata = {source_id: "all" for source_id in effects}
    if set(strata) != set(effects):
        raise ValueError("strata and effects must cover the same source IDs")

    grouped: dict[str, list[float]] = {}
    for source_id, effect in effects.items():
        grouped.setdefault(strata[source_id], []).append(float(effect))
    rng = np.random.default_rng(seed)
    boot_sum = np.zeros(resamples, dtype=np.float64)
    total = 0
    for values in grouped.values():
        array = np.asarray(values, dtype=np.float64)
        indices = rng.integers(0, len(array), size=(resamples, len(array)))
        boot_sum += array[indices].sum(axis=1)
        total += len(array)
    samples = boot_sum / total
    values = np.asarray(list(effects.values()), dtype=np.float64)
    return BootstrapResult(
        mean=float(values.mean()),
        ci_low=float(np.quantile(samples, 0.025)),
        ci_high=float(np.quantile(samples, 0.975)),
        source_count=len(values),
        resamples=resamples,
        seed=seed,
    )


def risk_coverage_curve(
    *, success: np.ndarray, irreversible: np.ndarray, uncertainty: np.ndarray
) -> dict[str, np.ndarray]:
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in (success, irreversible, uncertainty)]
    if len({array.shape for array in arrays}) != 1 or not arrays[0].size:
        raise ValueError("success, irreversible, and uncertainty require equal non-empty shapes")
    success_array, irreversible_array, uncertainty_array = arrays
    order = np.argsort(uncertainty_array, kind="stable")
    success_ordered = success_array[order]
    irreversible_ordered = irreversible_array[order]
    count = np.arange(1, len(order) + 1, dtype=np.float64)
    return {
        "coverage": count / len(order),
        "success_rate": np.cumsum(success_ordered) / count,
        "irreversible_rate": np.cumsum(irreversible_ordered) / count,
        "uncertainty_threshold": uncertainty_array[order],
    }
