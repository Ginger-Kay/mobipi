"""Fail-closed state snapshots for EC-1 handoff experiments.

The snapshot deliberately has no simulator or torch import at module import
time.  Callers must provide explicit adapters for controller state and, when
needed, the torch module.  This prevents a partial snapshot from being
mistaken for a reproducible handoff.

The serialized form is an internal pickle artifact.  It must only be loaded
from artifacts produced by the same trusted experiment workspace.
"""

from __future__ import annotations

import copy
import os
import pickle
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np


HANDOFF_SNAPSHOT_SCHEMA_VERSION = "ec1-handoff-v1"


class HandoffSnapshotError(RuntimeError):
    """Base error for invalid or incomplete handoff snapshots."""


class IncompleteHandoffSnapshot(HandoffSnapshotError):
    """Raised when a required reproducibility component is unavailable."""


@dataclass
class FrameStackSnapshot:
    """A copy of one frame-stack wrapper's mutable rollout state."""

    wrapper_name: str
    wrapper_index: int
    num_frames: int
    timestep: Any
    obs_history: Dict[str, Tuple[Any, ...]]


@dataclass
class RNGSnapshot:
    """Global and explicitly named generator states used by a rollout."""

    python_random_state: Any
    numpy_global_state: Any
    numpy_generator_states: Dict[str, Dict[str, Any]]
    torch_cpu_state: Any = None
    torch_cuda_states: Any = None


@dataclass
class HandoffSnapshot:
    """Scoped EC-1 handoff state; exhaustive coverage is not implied."""

    schema_version: str
    env_state: Any
    frame_stacks: Tuple[FrameStackSnapshot, ...]
    rng_state: RNGSnapshot
    controller_state: Any
    controller_state_captured: bool
    metadata: Dict[str, Any]


def _walk_env_chain(env: Any) -> Iterable[Any]:
    """Yield wrapper and environment objects without invoking proxy fallback."""

    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        attributes = vars(current) if hasattr(current, "__dict__") else {}
        current = attributes.get("env")


def _frame_stack_wrappers(env: Any) -> Tuple[Any, ...]:
    wrappers = []
    for current in _walk_env_chain(env):
        attributes = vars(current) if hasattr(current, "__dict__") else {}
        required = {"obs_history", "num_frames", "timestep"}
        if required.issubset(attributes):
            wrappers.append(current)
    return tuple(wrappers)


def _copy_obs_history(wrapper: Any) -> Dict[str, Tuple[Any, ...]]:
    attributes = vars(wrapper)
    history = attributes.get("obs_history")
    if not isinstance(history, Mapping) or not history:
        raise IncompleteHandoffSnapshot(
            "FrameStackWrapper has no initialized obs_history; call env.reset() "
            "or reach the handoff point before capturing"
        )

    num_frames = attributes.get("num_frames")
    if not isinstance(num_frames, int) or num_frames <= 1:
        raise IncompleteHandoffSnapshot(
            "FrameStackWrapper has an invalid num_frames value"
        )

    cached_history = attributes.get("obs_history_cache", None)
    if cached_history is not None:
        raise IncompleteHandoffSnapshot(
            "FrameStackWrapper has an active obs_history_cache; cached and live "
            "histories are not covered by this snapshot"
        )

    copied = {}
    for key, frames in history.items():
        if not isinstance(frames, (deque, list, tuple)):
            raise IncompleteHandoffSnapshot(
                f"obs_history[{key!r}] is not a deque/list/tuple"
            )
        if len(frames) != num_frames:
            raise IncompleteHandoffSnapshot(
                f"obs_history[{key!r}] has {len(frames)} frames, expected {num_frames}"
            )
        copied[key] = tuple(copy.deepcopy(list(frames)))
    return copied


def capture_frame_stack_state(env: Any) -> Tuple[FrameStackSnapshot, ...]:
    """Capture every initialized frame-stack wrapper in the env chain."""

    wrappers = _frame_stack_wrappers(env)
    if not wrappers:
        raise IncompleteHandoffSnapshot(
            "no frame-stack wrapper with obs_history/num_frames/timestep was found"
        )

    snapshots = []
    for index, wrapper in enumerate(wrappers):
        attributes = vars(wrapper)
        snapshots.append(
            FrameStackSnapshot(
                wrapper_name=type(wrapper).__name__,
                wrapper_index=index,
                num_frames=attributes["num_frames"],
                timestep=copy.deepcopy(attributes["timestep"]),
                obs_history=_copy_obs_history(wrapper),
            )
        )
    return tuple(snapshots)


def restore_frame_stack_state(
    env: Any, frame_stacks: Tuple[FrameStackSnapshot, ...]
) -> None:
    """Restore frame-stack state after the underlying env has been reset_to."""

    wrappers = _frame_stack_wrappers(env)
    if len(wrappers) != len(frame_stacks):
        raise IncompleteHandoffSnapshot(
            f"frame-stack wrapper count changed: snapshot has {len(frame_stacks)}, "
            f"env has {len(wrappers)}"
        )

    for wrapper, snapshot in zip(wrappers, frame_stacks):
        attributes = vars(wrapper)
        if type(wrapper).__name__ != snapshot.wrapper_name:
            raise IncompleteHandoffSnapshot(
                "frame-stack wrapper type changed between capture and restore"
            )
        if attributes.get("num_frames") != snapshot.num_frames:
            raise IncompleteHandoffSnapshot(
                "frame-stack num_frames changed between capture and restore"
            )
        current_keys = set(attributes.get("obs_history", {}) or {})
        snapshot_keys = set(snapshot.obs_history)
        if current_keys and current_keys != snapshot_keys:
            raise IncompleteHandoffSnapshot(
                "frame-stack observation keys changed between capture and restore"
            )

        attributes["obs_history"] = {
            key: deque(
                copy.deepcopy(list(frames)), maxlen=snapshot.num_frames
            )
            for key, frames in snapshot.obs_history.items()
        }
        attributes["timestep"] = copy.deepcopy(snapshot.timestep)
        if "obs_history_cache" in attributes:
            attributes["obs_history_cache"] = None


def _capture_numpy_generator_state(generator: Any) -> Dict[str, Any]:
    if hasattr(generator, "bit_generator"):
        return {
            "kind": "bit_generator",
            "state": copy.deepcopy(generator.bit_generator.state),
        }
    if hasattr(generator, "get_state") and hasattr(generator, "set_state"):
        return {
            "kind": "random_state",
            "state": copy.deepcopy(generator.get_state()),
        }
    raise IncompleteHandoffSnapshot(
        "named NumPy generator must expose bit_generator.state or get_state/set_state"
    )


def _restore_numpy_generator_state(generator: Any, state: Mapping[str, Any]) -> None:
    kind = state.get("kind")
    if kind == "bit_generator" and hasattr(generator, "bit_generator"):
        generator.bit_generator.state = copy.deepcopy(state["state"])
        return
    if kind == "random_state" and hasattr(generator, "set_state"):
        generator.set_state(copy.deepcopy(state["state"]))
        return
    raise IncompleteHandoffSnapshot(
        "named NumPy generator type changed between capture and restore"
    )


def capture_rng_state(
    *,
    torch_module: Any = None,
    numpy_generators: Optional[Mapping[str, Any]] = None,
    required_numpy_generators: Iterable[str] = (),
    require_torch: bool = True,
    require_cuda: bool = True,
) -> RNGSnapshot:
    """Capture RNG state without importing torch unless explicitly requested."""

    numpy_generators = dict(numpy_generators or {})
    missing_generators = set(required_numpy_generators) - set(numpy_generators)
    if missing_generators:
        raise IncompleteHandoffSnapshot(
            "required named NumPy generators were not provided: "
            + ", ".join(sorted(missing_generators))
        )

    if require_torch and torch_module is None:
        raise IncompleteHandoffSnapshot(
            "torch_module is required to capture policy RNG state"
        )
    if require_cuda and torch_module is None:
        raise IncompleteHandoffSnapshot(
            "torch_module is required to capture CUDA RNG state"
        )

    torch_cpu_state = None
    torch_cuda_states = None
    if torch_module is not None:
        get_rng_state = getattr(torch_module, "get_rng_state", None)
        if not callable(get_rng_state):
            raise IncompleteHandoffSnapshot("torch module has no get_rng_state()")
        torch_cpu_state = copy.deepcopy(get_rng_state())

        if require_cuda:
            cuda = getattr(torch_module, "cuda", None)
            if cuda is None or not callable(getattr(cuda, "is_available", None)):
                raise IncompleteHandoffSnapshot(
                    "torch module has no CUDA availability interface"
                )
            if not cuda.is_available():
                raise IncompleteHandoffSnapshot(
                    "CUDA RNG state is required but CUDA is unavailable"
                )
            get_cuda_rng_state_all = getattr(cuda, "get_rng_state_all", None)
            if not callable(get_cuda_rng_state_all):
                raise IncompleteHandoffSnapshot(
                    "torch CUDA module has no get_rng_state_all()"
                )
            torch_cuda_states = copy.deepcopy(get_cuda_rng_state_all())

    return RNGSnapshot(
        python_random_state=copy.deepcopy(random.getstate()),
        numpy_global_state=copy.deepcopy(np.random.get_state()),
        numpy_generator_states={
            name: _capture_numpy_generator_state(generator)
            for name, generator in sorted(numpy_generators.items())
        },
        torch_cpu_state=torch_cpu_state,
        torch_cuda_states=torch_cuda_states,
    )


def restore_rng_state(
    rng_state: RNGSnapshot,
    *,
    torch_module: Any = None,
    numpy_generators: Optional[Mapping[str, Any]] = None,
    require_torch: bool = True,
    require_cuda: bool = True,
) -> None:
    """Restore RNG state, rejecting missing adapters before partial restore."""

    numpy_generators = dict(numpy_generators or {})
    saved_names = set(rng_state.numpy_generator_states)
    provided_names = set(numpy_generators)
    if saved_names != provided_names:
        raise IncompleteHandoffSnapshot(
            "named NumPy generator coverage changed: "
            f"snapshot={sorted(saved_names)}, provided={sorted(provided_names)}"
        )

    has_torch_state = rng_state.torch_cpu_state is not None
    if has_torch_state and torch_module is None:
        raise IncompleteHandoffSnapshot(
            "snapshot contains torch RNG state but no torch module was provided"
        )
    if rng_state.torch_cuda_states is not None and torch_module is None:
        raise IncompleteHandoffSnapshot(
            "snapshot contains CUDA RNG state but no torch module was provided"
        )
    if require_torch and (torch_module is None or not has_torch_state):
        raise IncompleteHandoffSnapshot(
            "torch RNG state is required for restore but is missing"
        )
    if require_cuda and (torch_module is None or rng_state.torch_cuda_states is None):
        raise IncompleteHandoffSnapshot(
            "CUDA RNG state is required for restore but is missing"
        )

    # Validate all torch adapters before mutating Python / NumPy global state.
    if has_torch_state:
        set_rng_state = getattr(torch_module, "set_rng_state", None)
        if not callable(set_rng_state):
            raise IncompleteHandoffSnapshot("torch module has no set_rng_state()")
    if rng_state.torch_cuda_states is not None:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is None or not callable(getattr(cuda, "set_rng_state_all", None)):
            raise IncompleteHandoffSnapshot(
                "torch CUDA module has no set_rng_state_all()"
            )

    random.setstate(copy.deepcopy(rng_state.python_random_state))
    np.random.set_state(copy.deepcopy(rng_state.numpy_global_state))
    for name, generator in sorted(numpy_generators.items()):
        _restore_numpy_generator_state(
            generator, rng_state.numpy_generator_states[name]
        )
    if has_torch_state:
        torch_module.set_rng_state(copy.deepcopy(rng_state.torch_cpu_state))
    if rng_state.torch_cuda_states is not None:
        torch_module.cuda.set_rng_state_all(copy.deepcopy(rng_state.torch_cuda_states))


def _validate_snapshot_requirements(
    snapshot: HandoffSnapshot,
    *,
    require_controller: bool,
    require_torch: bool,
    require_cuda: bool,
) -> None:
    if not isinstance(snapshot, HandoffSnapshot):
        raise IncompleteHandoffSnapshot("object is not a HandoffSnapshot")
    if snapshot.schema_version != HANDOFF_SNAPSHOT_SCHEMA_VERSION:
        raise IncompleteHandoffSnapshot(
            "unsupported handoff snapshot schema: " + str(snapshot.schema_version)
        )
    if snapshot.env_state is None:
        raise IncompleteHandoffSnapshot("snapshot has no simulator state")
    if not snapshot.frame_stacks:
        raise IncompleteHandoffSnapshot("snapshot has no frame-stack state")
    if require_controller and not snapshot.controller_state_captured:
        raise IncompleteHandoffSnapshot(
            "controller state is required but was not captured"
        )
    if require_torch and snapshot.rng_state.torch_cpu_state is None:
        raise IncompleteHandoffSnapshot("torch CPU RNG state is missing")
    if require_cuda and snapshot.rng_state.torch_cuda_states is None:
        raise IncompleteHandoffSnapshot("CUDA RNG state is missing")


def capture_handoff_snapshot(
    env: Any,
    *,
    controller_get_state: Optional[Callable[[], Any]] = None,
    controller_set_state: Optional[Callable[[Any], None]] = None,
    torch_module: Any = None,
    numpy_generators: Optional[Mapping[str, Any]] = None,
    required_numpy_generators: Iterable[str] = (),
    require_controller: bool = True,
    require_torch: bool = True,
    require_cuda: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HandoffSnapshot:
    """Capture a scoped, restorable EC-1 snapshot for covered state owners.

    ``controller_get_state`` and ``controller_set_state`` are intentionally
    caller-provided.  A generic ``__dict__`` copy is unsafe for simulator
    controllers and would make a false reproducibility claim.
    """

    get_env_state = getattr(env, "get_state", None)
    reset_to = getattr(env, "reset_to", None)
    if not callable(get_env_state) or not callable(reset_to):
        raise IncompleteHandoffSnapshot(
            "environment must expose callable get_state() and reset_to()"
        )

    if callable(controller_get_state) != callable(controller_set_state):
        raise IncompleteHandoffSnapshot(
            "controller state adapter must provide both get_state and set_state"
        )
    if require_controller and (
        not callable(controller_get_state) or not callable(controller_set_state)
    ):
        raise IncompleteHandoffSnapshot(
            "controller state adapter requires callable get_state and set_state"
        )

    env_state = copy.deepcopy(get_env_state())
    if env_state is None:
        raise IncompleteHandoffSnapshot("environment get_state() returned None")

    frame_stacks = capture_frame_stack_state(env)
    controller_captured = callable(controller_get_state)
    controller_state = (
        copy.deepcopy(controller_get_state()) if controller_captured else None
    )
    if require_controller and controller_state is None:
        # None can be a valid state, but an explicit state adapter returning
        # None is too ambiguous for a fail-closed artifact.
        raise IncompleteHandoffSnapshot(
            "controller state adapter returned None"
        )

    # Capture RNG last so an adapter that performs bookkeeping cannot leave the
    # snapshot's RNG position one step earlier than the actual handoff.
    rng_state = capture_rng_state(
        torch_module=torch_module,
        numpy_generators=numpy_generators,
        required_numpy_generators=required_numpy_generators,
        require_torch=require_torch,
        require_cuda=require_cuda,
    )

    snapshot_metadata = copy.deepcopy(dict(metadata or {}))
    snapshot_metadata["schema_version"] = HANDOFF_SNAPSHOT_SCHEMA_VERSION
    snapshot_metadata["components"] = {
        "simulator_state": True,
        "frame_stack_state": True,
        "wrapper_timestep": True,
        "python_numpy_rng": True,
        "torch_cpu_rng": rng_state.torch_cpu_state is not None,
        "torch_cuda_rng": rng_state.torch_cuda_states is not None,
        "named_numpy_generators": sorted(rng_state.numpy_generator_states),
        "controller_state": controller_captured,
    }

    snapshot = HandoffSnapshot(
        schema_version=HANDOFF_SNAPSHOT_SCHEMA_VERSION,
        env_state=env_state,
        frame_stacks=frame_stacks,
        rng_state=rng_state,
        controller_state=controller_state,
        controller_state_captured=controller_captured,
        metadata=snapshot_metadata,
    )
    _validate_snapshot_requirements(
        snapshot,
        require_controller=require_controller,
        require_torch=require_torch,
        require_cuda=require_cuda,
    )
    return snapshot


def restore_handoff_snapshot(
    env: Any,
    snapshot: HandoffSnapshot,
    *,
    controller_set_state: Optional[Callable[[Any], None]] = None,
    torch_module: Any = None,
    numpy_generators: Optional[Mapping[str, Any]] = None,
    require_controller: bool = True,
    require_torch: bool = True,
    require_cuda: bool = True,
) -> None:
    """Restore simulator, controller, frame-stack, and RNG state."""

    _validate_snapshot_requirements(
        snapshot,
        require_controller=require_controller,
        require_torch=require_torch,
        require_cuda=require_cuda,
    )
    reset_to = getattr(env, "reset_to", None)
    if not callable(reset_to):
        raise IncompleteHandoffSnapshot("environment has no callable reset_to()")
    if snapshot.controller_state_captured and not callable(controller_set_state):
        raise IncompleteHandoffSnapshot(
            "snapshot contains controller state but no restore adapter was provided"
        )

    # reset_to intentionally runs first because the normal FrameStackWrapper
    # reset path overwrites history. RNG is restored last so reset/controller
    # bookkeeping cannot consume the final handoff RNG position.
    reset_to(copy.deepcopy(snapshot.env_state))
    if snapshot.controller_state_captured:
        controller_set_state(copy.deepcopy(snapshot.controller_state))
    restore_frame_stack_state(env, snapshot.frame_stacks)
    restore_rng_state(
        snapshot.rng_state,
        torch_module=torch_module,
        numpy_generators=numpy_generators,
        require_torch=require_torch,
        require_cuda=require_cuda,
    )


def write_handoff_snapshot(path: os.PathLike, snapshot: HandoffSnapshot) -> None:
    """Write a trusted internal snapshot artifact atomically."""

    _validate_snapshot_requirements(
        snapshot,
        require_controller=True,
        require_torch=True,
        require_cuda=True,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp-{}".format(os.getpid()))
    with open(temporary, "wb") as stream:
        pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def read_handoff_snapshot(path: os.PathLike) -> HandoffSnapshot:
    """Load and schema-check a snapshot produced by write_handoff_snapshot."""

    path = Path(path)
    with open(path, "rb") as stream:
        snapshot = pickle.load(stream)
    _validate_snapshot_requirements(
        snapshot,
        require_controller=True,
        require_torch=True,
        require_cuda=True,
    )
    return snapshot
