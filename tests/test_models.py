import gc
import unittest

import torch

from mobiwam.models import (
    EvaluatorConfig,
    OBCWAM,
    TrajectoryOnlyEvaluator,
    ValueOnlyEvaluator,
    trainable_parameter_count,
)


class EvaluatorArchitectureTest(unittest.TestCase):
    def setUp(self):
        self.config = EvaluatorConfig(input_dim=64, candidate_dim=8)

    def test_obc_wam_shape_and_frozen_capacity(self):
        model = OBCWAM(self.config)
        self.assertEqual(model.backbone.encoder.num_layers, 4)
        self.assertEqual(model.backbone.encoder.layers[0].self_attn.num_heads, 8)
        count = trainable_parameter_count(model)
        self.assertGreaterEqual(count, 5_000_000)
        self.assertLessEqual(count, 10_000_000)
        batch = model(
            context=torch.randn(2, 4, 64),
            option_ids=torch.tensor([0, 2]),
            candidate_params=torch.randn(2, 8),
            phase_ids=torch.tensor([0, 1]),
            duration=torch.tensor([0.0, 0.5]),
        )
        self.assertEqual(batch["common_boundary_latent"].shape, (2, 256))
        self.assertEqual(batch["typed_internal_states"].shape, (2, 3, 512))
        self.assertEqual(batch["success"].shape, (2, 1))
        self.assertEqual(batch["duration_cost"].shape, (2, 6))
        del model
        gc.collect()

    def test_matched_controls_are_within_ten_percent(self):
        obc_count = trainable_parameter_count(OBCWAM(self.config))
        gc.collect()
        value_count = trainable_parameter_count(ValueOnlyEvaluator(self.config))
        gc.collect()
        trajectory_count = trainable_parameter_count(
            TrajectoryOnlyEvaluator(self.config)
        )
        for count in (value_count, trajectory_count):
            self.assertLessEqual(abs(count - obc_count) / obc_count, 0.10)


if __name__ == "__main__":
    unittest.main()
