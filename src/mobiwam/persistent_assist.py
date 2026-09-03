"""Outcome-blind B0 persistent-assist primitives.

These helpers deliberately operate on deployment-observable state and nominal
policy intent only; privileged route outcomes are rejected by the compiler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ASSIST_CANDIDATES: Mapping[str, tuple[float, float, float]] = {
    "a1": (0.25, 0.00, 0.00),
    "a2": (0.50, 0.00, 0.00),
    "a3": (0.75, 0.00, 0.00),
    "a4": (0.50, 0.25, 0.25),
    "a5": (0.50, -0.25, -0.25),
}
FORBIDDEN_SOURCE_FIELDS = frozenset(
    {"success", "progress_after", "contact_after", "route_cost", "outcome", "terminal_reason"}
)


@dataclass(frozen=True)
class PersistentAssistPlan:
    candidate_id: str
    chunk_poses_world: np.ndarray
    planned_translation_m: float
    planned_yaw_rad: float
    chunk_count: int


def compile_persistent_assist(
    nominal_intents_world: Sequence[np.ndarray],
    actual_base_poses_world: Sequence[np.ndarray],
    candidate_id: str,
    *,
    parallel_cap_m: float = 0.45,
    lateral_cap_m: float = 0.25,
    yaw_cap_rad: float = 0.35,
) -> PersistentAssistPlan:
    """Compile one receding-horizon plan using the actual pose at each chunk."""
    if candidate_id not in ASSIST_CANDIDATES:
        raise ValueError("candidate_id must be one of a1..a5; A(0) is not a candidate")
    if len(nominal_intents_world) != len(actual_base_poses_world) or not nominal_intents_world:
        raise ValueError("one actual base pose is required for every nominal chunk")
    parallel, lateral, yaw_gain = ASSIST_CANDIDATES[candidate_id]
    planned: list[np.ndarray] = []
    total_translation = 0.0
    total_yaw = 0.0
    for intent, actual in zip(nominal_intents_world, actual_base_poses_world):
        intent = np.asarray(intent, dtype=float)
        actual = np.asarray(actual, dtype=float)
        if intent.shape != (4, 4) or actual.shape != (4, 4):
            raise ValueError("nominal intents and actual poses must have shape (4, 4)")
        tangent = intent[:2, 3]
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-9:
            delta = np.zeros(2)
        else:
            tangent = tangent / norm
            lateral_axis = np.array([-tangent[1], tangent[0]])
            delta = (parallel * tangent + lateral * lateral_axis) * min(norm, parallel_cap_m / max(abs(parallel), 1e-12))
        delta = np.clip(delta, -max(parallel_cap_m, lateral_cap_m), max(parallel_cap_m, lateral_cap_m))
        pose = actual.copy()
        pose[:2, 3] = actual[:2, 3] + delta
        yaw = float(np.arctan2(actual[1, 0], actual[0, 0]))
        yaw_delta = float(np.clip(yaw_gain * min(norm, 1.0), -yaw_cap_rad, yaw_cap_rad))
        new_yaw = yaw + yaw_delta
        c, s = np.cos(new_yaw), np.sin(new_yaw)
        pose[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        planned.append(pose)
        total_translation += float(np.linalg.norm(delta))
        total_yaw += yaw_delta
    return PersistentAssistPlan(candidate_id, np.stack(planned), total_translation, total_yaw, len(planned))


def realized_motion_metrics(
    base_poses_world: Sequence[np.ndarray], base_velocities_world: Sequence[np.ndarray],
    *, moving_threshold_mps: float, phase: Sequence[str] | None = None,
) -> dict[str, float | int]:
    poses = np.asarray(base_poses_world, dtype=float)
    velocities = np.asarray(base_velocities_world, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or velocities.ndim != 2 or velocities.shape[0] != poses.shape[0]:
        raise ValueError("poses must be [T,4,4] and velocities [T,V]")
    speed = np.linalg.norm(velocities[:, :2], axis=1)
    moving = speed > float(moving_threshold_mps)
    path = float(np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1).sum()) if len(poses) > 1 else 0.0
    net = float(np.linalg.norm(poses[-1, :2, 3] - poses[0, :2, 3]))
    result: dict[str, float | int] = {
        "actual_net_translation_m": net,
        "actual_path_length_m": path,
        "net_over_path": net / path if path > 1e-12 else 0.0,
        "moving_steps": int(moving.sum()),
        "moving_fraction": float(moving.mean()) if len(moving) else 0.0,
    }
    if phase is not None:
        if len(phase) != len(poses):
            raise ValueError("phase must have one label per pose")
        active = np.asarray([p in {"MANIPULATE", "ASSIST_ACTIVE"} for p in phase])
        result["active_path_fraction"] = float(np.linalg.norm(np.diff(poses[active, :2, 3], axis=0), axis=1).sum() / path) if active.sum() > 1 and path > 1e-12 else 0.0
    return result


def compile_source_without_outcome(fields: Mapping[str, object]) -> dict[str, object]:
    leaked = sorted(FORBIDDEN_SOURCE_FIELDS.intersection(fields))
    if leaked:
        raise ValueError(f"source compiler received forbidden outcome fields: {leaked}")
    return dict(fields)
