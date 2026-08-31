import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mobiwam.predict import (
    LOCKED_FREEZE_FIELDS,
    prediction_arrays,
    validate_locked_freeze,
)


class FakeEvaluator(torch.nn.Module):
    def forward(self, **inputs):
        batch = inputs["context"].shape[0]
        device = inputs["context"].device
        return {
            "success": torch.arange(batch, device=device).reshape(-1, 1).float(),
            "irreversible_risk": torch.zeros(batch, 1, device=device),
            "duration_cost": torch.zeros(batch, 6, device=device),
            "predictive_uncertainty": torch.ones(batch, 1, device=device),
        }


class PredictionExportTest(unittest.TestCase):
    def test_prediction_arrays_preserve_split_and_identity(self):
        count = 3
        data = {
            "context": np.ones((count, 4, 8), dtype=np.float32),
            "option_ids": np.asarray([0, 1, 2]),
            "candidate_params": np.ones((count, 4), dtype=np.float32),
            "phase_ids": np.zeros(count, dtype=np.int64),
            "duration": np.ones(count, dtype=np.float32),
            "source_ids": np.asarray(["s0", "s1", "s2"]),
            "split": np.asarray(["validation"] * count),
            "success": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            "irreversible_risk": np.zeros(count, dtype=np.float32),
            "duration_cost": np.zeros((count, 6), dtype=np.float32),
            "candidate_ids": np.asarray(["e0", "d0", "a0"]),
            "is_a0": np.asarray([False, False, False]),
        }
        result = prediction_arrays(
            FakeEvaluator(),
            data,
            np.arange(count),
            device=torch.device("cpu"),
            batch_size=2,
        )
        self.assertEqual(result["success_logits"].shape, (count, 1))
        self.assertEqual(result["uncertainty_logits"].shape, (count, 1))
        np.testing.assert_array_equal(result["source_ids"], data["source_ids"])
        np.testing.assert_array_equal(result["split"], data["split"])

    def test_locked_freeze_requires_all_frozen_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "freeze.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "locked_test_open_authorized": True,
                        **{field: "a" * 64 for field in LOCKED_FREEZE_FIELDS},
                    }
                )
            )
            freeze = validate_locked_freeze(path)
            self.assertEqual(freeze["status"], "frozen")
            freeze.pop("calibration_sha256")
            path.write_text(json.dumps(freeze))
            with self.assertRaisesRegex(ValueError, "lacks fields"):
                validate_locked_freeze(path)

    def test_locked_freeze_rejects_unfrozen_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "freeze.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "draft",
                        "locked_test_open_authorized": False,
                        **{field: "b" * 64 for field in LOCKED_FREEZE_FIELDS},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "not frozen"):
                validate_locked_freeze(path)


if __name__ == "__main__":
    unittest.main()
