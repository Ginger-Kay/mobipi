from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EvaluatorConfig:
    input_dim: int = 1024
    candidate_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    feedforward_dim: int = 2560
    dropout: float = 0.1
    num_options: int = 4
    num_events: int = 16
    num_phases: int = 32
    adapter_rank: int = 32
    trajectory_horizon: int = 16
    trajectory_dim: int = 7

    def validate(self) -> None:
        if min(self.hidden_dim, self.num_layers, self.num_heads) <= 0:
            raise ValueError("hidden size, layers, and heads must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.feedforward_dim <= self.hidden_dim:
            raise ValueError("feedforward_dim must exceed hidden_dim")
        if min(self.input_dim, self.candidate_dim, self.adapter_rank) <= 0:
            raise ValueError("feature and adapter dimensions must be positive")
        if min(self.num_options, self.num_events, self.num_phases) <= 0:
            raise ValueError("option, event, and phase counts must be positive")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "EvaluatorConfig":
        aliases = {
            "layers": "num_layers",
            "heads": "num_heads",
        }
        fields = set(cls.__dataclass_fields__)
        values: dict[str, Any] = {}
        for name, value in row.items():
            resolved = aliases.get(name, name)
            if resolved in fields:
                values[resolved] = value
        config = cls(**values)
        config.validate()
        return config


class OptionAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, num_options: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.ModuleList(
            nn.Linear(rank, hidden_dim, bias=False) for _ in range(num_options)
        )

    def forward(self, value: Tensor, option_ids: Tensor) -> Tensor:
        lowered = torch.nn.functional.gelu(self.down(value))
        stacked = torch.stack([module(lowered) for module in self.up], dim=1)
        batch_index = torch.arange(value.shape[0], device=value.device)
        return value + stacked[batch_index, option_ids]


class SharedEvaluatorBackbone(nn.Module):
    """One frozen-feature evaluator backbone shared by every learned control."""

    def __init__(self, config: EvaluatorConfig):
        super().__init__()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        self.context_projection = nn.Linear(config.input_dim, hidden)
        self.candidate_projection = nn.Linear(config.candidate_dim, hidden)
        self.duration_projection = nn.Linear(1, hidden)
        self.option_embedding = nn.Embedding(config.num_options, hidden)
        self.event_embedding = nn.Embedding(config.num_events, hidden)
        self.phase_embedding = nn.Embedding(config.num_phases, hidden)
        self.common_boundary_token = nn.Parameter(torch.empty(1, 1, hidden))
        self.typed_internal_tokens = nn.Parameter(torch.empty(1, 3, hidden))
        self.register_buffer(
            "option_event_ids",
            torch.tensor(
                [
                    [0, 1, 2],  # E: QUERY, EXECUTE, REPLAN
                    [3, 8, 1],  # D: MOVE, POST_DOCK_POLICY_READY, EXECUTE
                    [0, 7, 2],  # A: QUERY, ASSIST, REPLAN
                    [9, 9, 9],  # X: SAFE_EXIT
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, norm=nn.LayerNorm(hidden)
        )
        self.adapter = OptionAdapter(hidden, config.adapter_rank, config.num_options)
        nn.init.normal_(self.common_boundary_token, std=0.02)
        nn.init.normal_(self.typed_internal_tokens, std=0.02)

    def forward(
        self,
        *,
        context: Tensor,
        option_ids: Tensor,
        candidate_params: Tensor,
        phase_ids: Tensor,
        duration: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if context.ndim != 3:
            raise ValueError("context must have shape [batch, tokens, input_dim]")
        batch = context.shape[0]
        if option_ids.shape != (batch,) or phase_ids.shape != (batch,):
            raise ValueError("option_ids and phase_ids must have shape [batch]")
        if candidate_params.shape != (batch, self.config.candidate_dim):
            raise ValueError("candidate_params has the wrong shape")
        duration = duration.reshape(batch, 1)
        context_tokens = self.context_projection(context)
        option_token = self.option_embedding(option_ids).unsqueeze(1)
        phase_token = self.phase_embedding(phase_ids).unsqueeze(1)
        candidate_token = self.candidate_projection(candidate_params).unsqueeze(1)
        duration_token = self.duration_projection(duration).unsqueeze(1)
        common = self.common_boundary_token.expand(batch, -1, -1)
        event_ids = self.option_event_ids[option_ids]
        internal = self.typed_internal_tokens.expand(batch, -1, -1) + self.event_embedding(event_ids)
        tokens = torch.cat(
            [common, internal, option_token, phase_token, duration_token, candidate_token, context_tokens],
            dim=1,
        )
        encoded = self.encoder(tokens)
        common_latent = self.adapter(encoded[:, 0], option_ids)
        internal_states = encoded[:, 1:4]
        return common_latent, internal_states


def _head(hidden: int, output: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, output),
    )


class OBCWAM(nn.Module):
    HEAD_DIMS: Mapping[str, int] = {
        "policy_compatibility": 1,
        "visibility": 1,
        "reachability": 1,
        "contact_retention": 1,
        "intent_error": 2,
        "progress": 1,
        "success": 1,
        "irreversible_risk": 1,
        "duration_cost": 6,
        "predictive_uncertainty": 1,
    }

    def __init__(self, config: EvaluatorConfig = EvaluatorConfig()):
        super().__init__()
        self.config = config
        self.backbone = SharedEvaluatorBackbone(config)
        self.heads = nn.ModuleDict(
            {name: _head(config.hidden_dim, size) for name, size in self.HEAD_DIMS.items()}
        )
        self.event_state_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        common, internal = self.backbone(**inputs)
        outputs = {name: head(common) for name, head in self.heads.items()}
        outputs["common_boundary_latent"] = common
        outputs["typed_internal_states"] = self.event_state_projection(internal)
        return outputs


class ValueOnlyEvaluator(nn.Module):
    def __init__(self, config: EvaluatorConfig = EvaluatorConfig()):
        super().__init__()
        self.backbone = SharedEvaluatorBackbone(config)
        self.value_head = _head(config.hidden_dim, 8)

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        common, _ = self.backbone(**inputs)
        values = self.value_head(common)
        return {
            "success": values[:, 0:1],
            "irreversible_risk": values[:, 1:2],
            "duration_cost": values[:, 2:8],
            "common_boundary_latent": common,
        }


class TrajectoryOnlyEvaluator(nn.Module):
    def __init__(self, config: EvaluatorConfig = EvaluatorConfig()):
        super().__init__()
        self.config = config
        self.backbone = SharedEvaluatorBackbone(config)
        self.trajectory_head = _head(
            config.hidden_dim, config.trajectory_horizon * config.trajectory_dim
        )
        self.score_head = _head(config.hidden_dim, 8)

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        common, _ = self.backbone(**inputs)
        trajectory = self.trajectory_head(common).reshape(
            common.shape[0], self.config.trajectory_horizon, self.config.trajectory_dim
        )
        values = self.score_head(common)
        return {
            "induced_ee_trajectory": trajectory,
            "success": values[:, 0:1],
            "irreversible_risk": values[:, 1:2],
            "duration_cost": values[:, 2:8],
            "common_boundary_latent": common,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
