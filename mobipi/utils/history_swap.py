"""Fail-closed FrameStack past-history interventions for EC-2.

The intervention deliberately preserves every owner in the physical snapshot
except frames ``0..num_frames-2`` of each FrameStack observation key.  Frame
``num_frames-1`` is the current observation and always remains sourced from the
physical snapshot.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Dict, Tuple

import numpy as np

from mobipi.utils.handoff_state import FrameStackSnapshot, HandoffSnapshot


HISTORY_SWAP_SCHEMA_VERSION = "ec2-history-swap-v1"


class HistorySwapError(RuntimeError):
    """Raised when a history-only intervention cannot be proven well formed."""


def _array_signature(value: Any) -> Tuple[Tuple[int, ...], str]:
    array = np.asarray(value)
    return tuple(array.shape), str(array.dtype)


def _hybrid_frame_stack(
    physical: FrameStackSnapshot, history: FrameStackSnapshot
) -> FrameStackSnapshot:
    if (
        physical.wrapper_name != history.wrapper_name
        or physical.wrapper_index != history.wrapper_index
        or physical.num_frames != history.num_frames
    ):
        raise HistorySwapError("FrameStack wrapper schema differs between sources")
    if physical.num_frames <= 1:
        raise HistorySwapError("history swap requires at least one past frame")
    if set(physical.obs_history) != set(history.obs_history):
        raise HistorySwapError("FrameStack observation keys differ between sources")

    hybrid_history: Dict[str, Tuple[Any, ...]] = {}
    for key in sorted(physical.obs_history):
        physical_frames = physical.obs_history[key]
        history_frames = history.obs_history[key]
        if len(physical_frames) != physical.num_frames:
            raise HistorySwapError(
                f"physical history for {key!r} does not contain num_frames entries"
            )
        if len(history_frames) != history.num_frames:
            raise HistorySwapError(
                f"history source for {key!r} does not contain num_frames entries"
            )
        for index, (physical_frame, history_frame) in enumerate(
            zip(physical_frames, history_frames)
        ):
            if _array_signature(physical_frame) != _array_signature(history_frame):
                raise HistorySwapError(
                    f"frame schema differs for {key!r} at temporal index {index}"
                )

        # Temporal index -1 is current t. It must never come from history_source.
        hybrid_history[key] = tuple(
            copy.deepcopy(list(history_frames[:-1]))
            + [copy.deepcopy(physical_frames[-1])]
        )

    return FrameStackSnapshot(
        wrapper_name=physical.wrapper_name,
        wrapper_index=physical.wrapper_index,
        num_frames=physical.num_frames,
        timestep=copy.deepcopy(physical.timestep),
        obs_history=hybrid_history,
    )


def build_history_intervention(
    physical_snapshot: HandoffSnapshot,
    history_snapshot: HandoffSnapshot,
    *,
    physical_source: str,
    history_source: str,
) -> HandoffSnapshot:
    """Return a physical snapshot with only FrameStack past frames replaced."""

    if not isinstance(physical_snapshot, HandoffSnapshot) or not isinstance(
        history_snapshot, HandoffSnapshot
    ):
        raise HistorySwapError("both sources must be HandoffSnapshot instances")
    if physical_snapshot.schema_version != history_snapshot.schema_version:
        raise HistorySwapError("handoff snapshot schema differs between sources")
    if len(physical_snapshot.frame_stacks) != len(history_snapshot.frame_stacks):
        raise HistorySwapError("FrameStack wrapper count differs between sources")
    if not physical_snapshot.frame_stacks:
        raise HistorySwapError("physical snapshot has no FrameStack state")

    frame_stacks = tuple(
        _hybrid_frame_stack(physical, history)
        for physical, history in zip(
            physical_snapshot.frame_stacks, history_snapshot.frame_stacks
        )
    )
    metadata = copy.deepcopy(physical_snapshot.metadata)
    metadata["history_intervention"] = {
        "schema_version": HISTORY_SWAP_SCHEMA_VERSION,
        "physical_source": str(physical_source),
        "history_source": str(history_source),
        "past_indices": list(range(frame_stacks[0].num_frames - 1)),
        "current_index": frame_stacks[0].num_frames - 1,
        "current_source": str(physical_source),
    }
    return replace(
        copy.deepcopy(physical_snapshot),
        frame_stacks=frame_stacks,
        metadata=metadata,
    )


def validate_history_intervention(
    hybrid: HandoffSnapshot,
    physical_snapshot: HandoffSnapshot,
    history_snapshot: HandoffSnapshot,
) -> None:
    """Prove that only past FrameStack frames changed in ``hybrid``."""

    if len(hybrid.frame_stacks) != len(physical_snapshot.frame_stacks):
        raise HistorySwapError("hybrid FrameStack wrapper count changed")

    for hybrid_stack, physical_stack, history_stack in zip(
        hybrid.frame_stacks,
        physical_snapshot.frame_stacks,
        history_snapshot.frame_stacks,
    ):
        if hybrid_stack.timestep != physical_stack.timestep:
            raise HistorySwapError("hybrid timestep did not remain physical-sourced")
        for key in sorted(physical_stack.obs_history):
            hybrid_frames = hybrid_stack.obs_history[key]
            physical_frames = physical_stack.obs_history[key]
            history_frames = history_stack.obs_history[key]
            if not np.array_equal(hybrid_frames[-1], physical_frames[-1]):
                raise HistorySwapError(
                    f"current frame for {key!r} did not remain physical-sourced"
                )
            for index in range(physical_stack.num_frames - 1):
                if not np.array_equal(hybrid_frames[index], history_frames[index]):
                    raise HistorySwapError(
                        f"past frame for {key!r} at {index} is not history-sourced"
                    )
