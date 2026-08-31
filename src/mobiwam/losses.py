from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LossWeights:
    paired_rank: float = 1.0
    a0_equals_e: float = 1.0
    boundary: float = 1.0


class OBCWAMLoss(nn.Module):
    def __init__(self, weights: LossWeights = LossWeights()):
        super().__init__()
        self.weights = weights
        self.binary = nn.BCEWithLogitsLoss()
        self.regression = nn.SmoothL1Loss()

    def forward(
        self, predictions: Mapping[str, Tensor], targets: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        event = self.regression(
            predictions["typed_internal_states"], targets["typed_internal_states"]
        )
        terminal = (
            self.binary(predictions["success"], targets["success"])
            + self.binary(
                predictions["irreversible_risk"], targets["irreversible_risk"]
            )
            + self.regression(
                predictions["duration_cost"], targets["duration_cost"]
            )
        )
        pair_indices = targets["pair_indices"].long()
        success_logit = predictions["success"].reshape(-1)
        pair_margin = success_logit[pair_indices[:, 0]] - success_logit[pair_indices[:, 1]]
        paired_rank = self.binary(pair_margin, targets["pair_preferred"].reshape(-1))
        a0_latent = predictions["common_boundary_latent"][targets["a0_indices"].long()]
        e_latent = predictions["common_boundary_latent"][targets["e_indices"].long()]
        a0_equals_e = self.regression(a0_latent, e_latent)
        boundary = self.regression(
            predictions["common_boundary_latent"], targets["common_boundary_latent"]
        )
        loss = (
            event
            + terminal
            + self.weights.paired_rank * paired_rank
            + self.weights.a0_equals_e * a0_equals_e
            + self.weights.boundary * boundary
        )
        return {
            "loss": loss,
            "event": event,
            "terminal": terminal,
            "paired_rank": paired_rank,
            "a0_equals_e": a0_equals_e,
            "boundary": boundary,
        }
