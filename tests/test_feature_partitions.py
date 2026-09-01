import tempfile
import unittest
from pathlib import Path

import numpy as np

from mobiwam.feature_partitions import (
    ALIGNMENT_FIELDS,
    LABEL_FIELDS,
    OBSERVABLE_FIELDS,
    OUTCOME_FIELDS,
    load_feature_partitions,
    partition_feature_arrays,
    validate_feature_partitions,
)


def arrays(count: int = 3) -> dict[str, np.ndarray]:
    result = {
        "context": np.ones((count, 4, 8), dtype=np.float32),
        "option_ids": np.arange(count, dtype=np.int64),
        "candidate_params": np.ones((count, 16), dtype=np.float32),
        "phase_ids": np.zeros(count, dtype=np.int64),
        "duration": np.ones(count, dtype=np.float32),
        "success": np.ones(count, dtype=np.float32),
        "irreversible_risk": np.zeros(count, dtype=np.float32),
        "duration_cost": np.zeros((count, 6), dtype=np.float32),
        "typed_internal_states": np.zeros((count, 3, 8), dtype=np.float32),
        "common_boundary_latent": np.zeros((count, 8), dtype=np.float32),
        "induced_ee_trajectory": np.zeros((count, 16, 7), dtype=np.float32),
        "trajectory_valid": np.ones(count, dtype=np.float32),
        "source_ids": np.asarray([f"s{index}" for index in range(count)]),
        "split": np.asarray(["train"] * count),
        "is_a0": np.zeros(count, dtype=np.bool_),
        "task_ids": np.asarray(["task"] * count),
        "task_families": np.asarray(["family"] * count),
        "route_types": np.asarray(["E"] * count),
        "candidate_ids": np.asarray(["e0"] * count),
        "repeat_indices": np.zeros(count, dtype=np.int64),
    }
    self_check = set(result)
    assert self_check == OBSERVABLE_FIELDS | LABEL_FIELDS
    return result


class FeaturePartitionTest(unittest.TestCase):
    def test_outcomes_are_absent_from_observable_partition(self):
        observable, labels = partition_feature_arrays(arrays())
        self.assertFalse(OUTCOME_FIELDS.intersection(observable))
        self.assertTrue(OUTCOME_FIELDS.issubset(labels))
        self.assertTrue(ALIGNMENT_FIELDS.issubset(observable))
        self.assertTrue(ALIGNMENT_FIELDS.issubset(labels))

    def test_alignment_mismatch_fails_closed(self):
        observable, labels = partition_feature_arrays(arrays())
        labels["source_ids"] = labels["source_ids"].copy()
        labels["source_ids"][0] = "different"
        with self.assertRaisesRegex(ValueError, "alignment differs"):
            validate_feature_partitions(observable, labels)

    def test_round_trip_keeps_physical_partitions(self):
        observable, labels = partition_feature_arrays(arrays())
        with tempfile.TemporaryDirectory() as temporary:
            feature_path = Path(temporary) / "features.npz"
            label_path = Path(temporary) / "labels.npz"
            np.savez_compressed(feature_path, **observable)
            np.savez_compressed(label_path, **labels)
            observed, outcomes, merged = load_feature_partitions(
                feature_path, label_path
            )
        self.assertFalse(OUTCOME_FIELDS.intersection(observed))
        self.assertEqual(set(outcomes), LABEL_FIELDS)
        self.assertTrue(OUTCOME_FIELDS.issubset(merged))


if __name__ == "__main__":
    unittest.main()
