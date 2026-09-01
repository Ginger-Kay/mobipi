from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


ALIGNMENT_FIELDS = frozenset(
    {
        "source_ids",
        "split",
        "is_a0",
        "task_ids",
        "task_families",
        "route_types",
        "candidate_ids",
        "repeat_indices",
    }
)
OBSERVABLE_FIELDS = frozenset(
    {
        "context",
        "option_ids",
        "candidate_params",
        "phase_ids",
        "duration",
        *ALIGNMENT_FIELDS,
    }
)
LABEL_FIELDS = frozenset(
    {
        "success",
        "irreversible_risk",
        "duration_cost",
        "typed_internal_states",
        "common_boundary_latent",
        "induced_ee_trajectory",
        "trajectory_valid",
        *ALIGNMENT_FIELDS,
    }
)
OUTCOME_FIELDS = LABEL_FIELDS.difference(ALIGNMENT_FIELDS)


def partition_feature_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    unknown = sorted(set(arrays).difference(OBSERVABLE_FIELDS | LABEL_FIELDS))
    if unknown:
        raise ValueError(f"unclassified feature arrays: {unknown}")
    missing_observable = sorted(OBSERVABLE_FIELDS.difference(arrays))
    missing_labels = sorted(LABEL_FIELDS.difference(arrays))
    if missing_observable:
        raise ValueError(f"observable partition lacks arrays: {missing_observable}")
    if missing_labels:
        raise ValueError(f"label partition lacks arrays: {missing_labels}")
    observable = {name: np.asarray(arrays[name]) for name in OBSERVABLE_FIELDS}
    labels = {name: np.asarray(arrays[name]) for name in LABEL_FIELDS}
    validate_feature_partitions(observable, labels)
    return observable, labels


def validate_feature_partitions(
    observable: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]
) -> None:
    leaked = sorted(OUTCOME_FIELDS.intersection(observable))
    if leaked:
        raise ValueError(f"simulator-only outcomes leaked into observable partition: {leaked}")
    missing_observable = sorted(OBSERVABLE_FIELDS.difference(observable))
    missing_labels = sorted(LABEL_FIELDS.difference(labels))
    if missing_observable:
        raise ValueError(f"observable partition lacks arrays: {missing_observable}")
    if missing_labels:
        raise ValueError(f"label partition lacks arrays: {missing_labels}")
    count = len(np.asarray(observable["source_ids"]))
    if count <= 0:
        raise ValueError("feature partitions must contain rows")
    if any(len(np.asarray(value)) != count for value in observable.values()):
        raise ValueError("observable arrays have inconsistent row counts")
    if any(len(np.asarray(value)) != count for value in labels.values()):
        raise ValueError("label arrays have inconsistent row counts")
    for name in ALIGNMENT_FIELDS:
        if not np.array_equal(np.asarray(observable[name]), np.asarray(labels[name])):
            raise ValueError(f"observable/label alignment differs for {name}")


def load_partition(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_feature_partitions(
    observable_path: Path, label_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    observable = load_partition(observable_path)
    labels = load_partition(label_path)
    validate_feature_partitions(observable, labels)
    merged = dict(observable)
    for name, value in labels.items():
        if name not in ALIGNMENT_FIELDS:
            merged[name] = value
    return observable, labels, merged
