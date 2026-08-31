import unittest

import numpy as np
import torch

from mobiwam.training import trajectory_supervision_loss


class TrajectorySupervisionTest(unittest.TestCase):
    def test_masked_loss_ignores_invalid_trace_rows(self):
        prediction = torch.zeros(2, 16, 7, requires_grad=True)
        data = {
            "induced_ee_trajectory": np.stack(
                [
                    np.ones((16, 7), dtype=np.float32),
                    np.full((16, 7), 100.0, dtype=np.float32),
                ]
            ),
            "trajectory_valid": np.asarray([1.0, 0.0], dtype=np.float32),
        }
        loss = trajectory_supervision_loss(
            {"induced_ee_trajectory": prediction},
            data,
            np.asarray([0, 1]),
            torch.device("cpu"),
        )
        self.assertAlmostEqual(float(loss.detach()), 0.5)
        loss.backward()
        self.assertGreater(float(prediction.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(prediction.grad[1].abs().sum()), 0.0)

    def test_missing_trajectory_targets_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            trajectory_supervision_loss(
                {"induced_ee_trajectory": torch.zeros(1, 16, 7)},
                {},
                np.asarray([0]),
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
