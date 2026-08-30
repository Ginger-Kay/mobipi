from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class DockSettleTimeout(RuntimeError):
    """The dock route failed to produce a stable post-navigation history."""


@dataclass(frozen=True)
class DockSettleEvidence:
    steps: int
    stable_zero_action_frames: int
    final_linear_speed_mps: float
    final_angular_speed_radps: float


def settle_flush_and_reset_policy(
    env: Any,
    rollout_policy: Any,
    *,
    language: str,
    velocity_reader: Callable[[Any], tuple[float, float]],
    action_dim: int = 12,
    history_length: int = 10,
    linear_threshold_mps: float = 0.005,
    angular_threshold_radps: float = 0.02,
    max_steps: int = 200,
    step_callback: Callable[[Any, np.ndarray, tuple[Any, ...]], None] | None = None,
) -> DockSettleEvidence:
    """Flush navigation history without resetting the environment.

    A frame counts only when it follows a zero action and both measured base
    speeds are below threshold. The policy is reset exactly once, after the
    frame stack is entirely stable post-dock context.
    """

    if action_dim <= 0 or history_length <= 0 or max_steps < history_length:
        raise ValueError("invalid settle protocol dimensions")
    if min(linear_threshold_mps, angular_threshold_radps) < 0.0:
        raise ValueError("velocity thresholds must be non-negative")

    stable_frames = 0
    final_linear = float("inf")
    final_angular = float("inf")
    zero_action = np.zeros(action_dim, dtype=np.float64)

    for step in range(1, max_steps + 1):
        result = env.step(zero_action.copy())
        if not isinstance(result, tuple) or len(result) not in (4, 5):
            raise RuntimeError("env.step must return a Gym or Gymnasium tuple")
        if step_callback is not None:
            step_callback(env, zero_action.copy(), result)
        final_linear, final_angular = velocity_reader(env)
        final_linear = float(final_linear)
        final_angular = float(final_angular)
        if not np.isfinite(final_linear) or not np.isfinite(final_angular):
            raise RuntimeError("base velocity reader returned non-finite values")

        if (
            final_linear <= linear_threshold_mps
            and final_angular <= angular_threshold_radps
        ):
            stable_frames += 1
        else:
            stable_frames = 0

        if stable_frames >= history_length:
            rollout_policy.start_episode(lang=language)
            return DockSettleEvidence(
                steps=step,
                stable_zero_action_frames=stable_frames,
                final_linear_speed_mps=final_linear,
                final_angular_speed_radps=final_angular,
            )

    raise DockSettleTimeout(
        f"dock did not produce {history_length} consecutive stable frames in {max_steps} steps"
    )
