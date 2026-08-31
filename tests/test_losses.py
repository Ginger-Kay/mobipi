import unittest

import torch

from mobiwam.losses import OBCWAMLoss


class OBCWAMLossTest(unittest.TestCase):
    def test_all_frozen_terms_are_present_and_finite(self):
        batch = 4
        predictions = {
            "typed_internal_states": torch.randn(batch, 3, 8),
            "common_boundary_latent": torch.randn(batch, 8),
            "success": torch.randn(batch, 1),
            "irreversible_risk": torch.randn(batch, 1),
            "duration_cost": torch.randn(batch, 6),
        }
        targets = {
            "typed_internal_states": torch.randn(batch, 3, 8),
            "common_boundary_latent": torch.randn(batch, 8),
            "success": torch.rand(batch, 1),
            "irreversible_risk": torch.rand(batch, 1),
            "duration_cost": torch.rand(batch, 6),
            "pair_preferred": torch.tensor([1.0, 0.0]),
            "pair_indices": torch.tensor([[0, 1], [2, 3]]),
            "a0_indices": torch.tensor([2]),
            "e_indices": torch.tensor([0]),
        }
        output = OBCWAMLoss()(predictions, targets)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(
            set(output),
            {"loss", "event", "terminal", "paired_rank", "a0_equals_e", "boundary"},
        )


if __name__ == "__main__":
    unittest.main()
