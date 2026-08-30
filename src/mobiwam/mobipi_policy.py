from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FutureChunkEvidence:
    chunk: np.ndarray
    official_first_action: np.ndarray
    max_abs_error: float

    @property
    def passed(self) -> bool:
        return self.max_abs_error <= 1e-6


def expose_future_action_chunk(
    rollout_policy: Any,
    observation: Mapping[str, Any],
    goal: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Expose BC_Transformer's future chunk through RolloutPolicy conversion."""

    chunk, _ = _single_forward_chunk_and_official(
        rollout_policy, observation, goal
    )
    return chunk


def _single_forward_chunk_and_official(
    rollout_policy: Any,
    observation: Mapping[str, Any],
    goal: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a full chunk and its official first action from one forward.

    The temporary get_action override changes only which tensor is returned by
    the frozen network. RolloutPolicy still performs its original observation
    preparation, action unnormalization, and rotation-format conversion.

    Some robomimic image randomizers can remain stochastic in evaluation mode.
    Re-running the network is therefore not a valid way to verify the first
    action. The second RolloutPolicy call replays step zero from the tensor
    captured by the first call and only repeats deterministic post-processing.
    """

    algo = rollout_policy.policy
    if not bool(getattr(algo, "supervise_all_steps", False)):
        raise RuntimeError("checkpoint does not supervise all transformer steps")
    if not bool(getattr(algo, "pred_future_acs", False)):
        config = getattr(getattr(algo, "algo_config", None), "transformer", None)
        if not bool(getattr(config, "pred_future_acs", False)):
            raise RuntimeError("checkpoint does not predict future actions")
    horizon = int(getattr(algo, "context_length", 0))
    if horizon <= 1:
        raise RuntimeError("invalid transformer context length")

    original_get_action = algo.get_action
    raw_chunk: Any | None = None

    def get_full_chunk(*, obs_dict: Any, goal_dict: Any = None) -> Any:
        nonlocal raw_chunk
        raw_chunk = algo.nets["policy"](
            obs_dict=obs_dict,
            actions=None,
            goal_dict=goal_dict,
        )

        return raw_chunk

    def replay_official_first_action(
        *, obs_dict: Any, goal_dict: Any = None
    ) -> Any:
        del obs_dict, goal_dict
        if raw_chunk is None:
            raise RuntimeError("future action chunk was not captured")
        return raw_chunk[:, 0, :]

    algo.get_action = get_full_chunk
    try:
        chunk = np.asarray(
            rollout_policy(
                copy.deepcopy(observation),
                goal=copy.deepcopy(goal),
                batched=False,
            )
        )
        algo.get_action = replay_official_first_action
        official = np.asarray(
            rollout_policy(
                copy.deepcopy(observation),
                goal=copy.deepcopy(goal),
                batched=False,
            )
        )
    finally:
        algo.get_action = original_get_action

    if chunk.ndim != 2 or chunk.shape[0] != horizon:
        raise RuntimeError(
            f"expected a [{horizon}, action_dim] future chunk, got {chunk.shape}"
        )
    return chunk, official


def sample_verified_future_chunk(
    rollout_policy: Any,
    observation: Mapping[str, Any],
    goal: Mapping[str, Any] | None = None,
    *,
    atol: float = 1e-6,
) -> FutureChunkEvidence:
    chunk, official = _single_forward_chunk_and_official(
        rollout_policy, observation, goal
    )
    if official.shape != chunk[0].shape:
        raise RuntimeError(
            f"official action shape {official.shape} does not match chunk[0] {chunk[0].shape}"
        )
    error = float(np.max(np.abs(chunk[0] - official)))
    evidence = FutureChunkEvidence(
        chunk=chunk,
        official_first_action=official,
        max_abs_error=error,
    )
    if error > atol:
        raise RuntimeError(
            f"future chunk exposure changed the official action: max abs error {error:.3e}"
        )
    return evidence
