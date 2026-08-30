from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


ACTION_DIM = 12
ARM_POSITION = slice(0, 3)
ARM_ROTATION = slice(3, 6)
TORSO = 6
BASE = slice(7, 10)
GRIPPER = 10
CONTROL_MODE = 11

ARM_POSITION_LIMIT_M = 0.05
ARM_ROTATION_LIMIT_RAD = 0.5


@dataclass(frozen=True)
class PandaOmronAction:
    arm_position: np.ndarray
    arm_rotation: np.ndarray
    torso: float
    base: np.ndarray
    gripper: float
    control_mode: float


@dataclass(frozen=True)
class ArmCompensation:
    action: np.ndarray
    desired_eef_pose_world: np.ndarray
    raw_normalized_arm_delta: np.ndarray
    saturated: bool
    transform_closure_pos_error_m: float
    transform_closure_rot_error_rad: float


def as_action(action: Sequence[float] | np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (ACTION_DIM,):
        raise ValueError(f"expected a {ACTION_DIM}-D action, got shape {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("action contains non-finite values")
    return value


def split_action(action: Sequence[float] | np.ndarray) -> PandaOmronAction:
    value = as_action(action)
    return PandaOmronAction(
        arm_position=value[ARM_POSITION].copy(),
        arm_rotation=value[ARM_ROTATION].copy(),
        torso=float(value[TORSO]),
        base=value[BASE].copy(),
        gripper=float(value[GRIPPER]),
        control_mode=float(value[CONTROL_MODE]),
    )


def lock_base(action: Sequence[float] | np.ndarray) -> np.ndarray:
    value = as_action(action).copy()
    value[BASE] = 0.0
    return value


def with_base_command(
    action: Sequence[float] | np.ndarray,
    base_command: Sequence[float] | np.ndarray,
) -> np.ndarray:
    command = np.asarray(base_command, dtype=np.float64)
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("base command must be a finite 3-D vector")
    value = as_action(action).copy()
    value[BASE] = np.clip(command, -1.0, 1.0)
    return value


def scaled_arm_delta(action: Sequence[float] | np.ndarray) -> np.ndarray:
    value = as_action(action)
    return np.concatenate(
        (
            np.clip(value[ARM_POSITION], -1.0, 1.0) * ARM_POSITION_LIMIT_M,
            np.clip(value[ARM_ROTATION], -1.0, 1.0) * ARM_ROTATION_LIMIT_RAD,
        )
    )


def _as_pose(name: str, pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} is not a homogeneous transform")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant is not one")
    return value


def invert_pose(pose: np.ndarray) -> np.ndarray:
    value = _as_pose("pose", pose)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def axis_angle_to_matrix(axis_angle: Sequence[float] | np.ndarray) -> np.ndarray:
    value = np.asarray(axis_angle, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("axis-angle must be a finite 3-D vector")
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = value / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    cosine = float(np.clip((np.trace(value) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return 0.5 * np.array(
            [value[2, 1] - value[1, 2], value[0, 2] - value[2, 0], value[1, 0] - value[0, 1]]
        )
    if np.pi - angle < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eig(value)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array(
        [value[2, 1] - value[1, 2], value[0, 2] - value[2, 0], value[1, 0] - value[0, 1]]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def nominal_world_intent(
    nominal_action: Sequence[float] | np.ndarray,
    nominal_origin_pose_world: np.ndarray,
    nominal_eef_pose_world: np.ndarray,
) -> np.ndarray:
    """Convert a fixed-base OSC delta command into its desired world-frame EE pose."""

    origin = _as_pose("nominal_origin_pose_world", nominal_origin_pose_world)
    eef = _as_pose("nominal_eef_pose_world", nominal_eef_pose_world)
    eef_in_origin = invert_pose(origin) @ eef
    delta = scaled_arm_delta(nominal_action)

    desired_in_origin = eef_in_origin.copy()
    desired_in_origin[:3, 3] = eef_in_origin[:3, 3] + delta[:3]
    desired_in_origin[:3, :3] = axis_angle_to_matrix(delta[3:]) @ eef_in_origin[:3, :3]
    return origin @ desired_in_origin


def compensate_world_intent(
    nominal_action: Sequence[float] | np.ndarray,
    *,
    nominal_origin_pose_world: np.ndarray,
    nominal_eef_pose_world: np.ndarray,
    assist_origin_pose_world_current: np.ndarray,
    assist_origin_pose_world_next: np.ndarray,
    assist_eef_pose_world_current: np.ndarray,
    clip: bool = True,
) -> ArmCompensation:
    """Build A's OSC command for an anticipated next base pose.

    RoboSuite stores the OSC target in the moving controller-origin frame. The
    target is therefore expressed in the *next* origin frame, while the delta
    is measured from the currently achieved EE pose in the current origin
    frame. This is the compensation that a direct action-array subtraction
    misses.
    """

    nominal = as_action(nominal_action)
    current_origin = _as_pose(
        "assist_origin_pose_world_current", assist_origin_pose_world_current
    )
    next_origin = _as_pose("assist_origin_pose_world_next", assist_origin_pose_world_next)
    current_eef = _as_pose("assist_eef_pose_world_current", assist_eef_pose_world_current)

    desired_world = nominal_world_intent(
        nominal,
        nominal_origin_pose_world=nominal_origin_pose_world,
        nominal_eef_pose_world=nominal_eef_pose_world,
    )
    current_eef_in_current_origin = invert_pose(current_origin) @ current_eef
    desired_eef_in_next_origin = invert_pose(next_origin) @ desired_world

    delta_position = (
        desired_eef_in_next_origin[:3, 3] - current_eef_in_current_origin[:3, 3]
    )
    delta_rotation = matrix_to_axis_angle(
        desired_eef_in_next_origin[:3, :3]
        @ current_eef_in_current_origin[:3, :3].T
    )
    raw = np.concatenate(
        (
            delta_position / ARM_POSITION_LIMIT_M,
            delta_rotation / ARM_ROTATION_LIMIT_RAD,
        )
    )
    saturated = bool(np.any(np.abs(raw) > 1.0 + 1e-9))

    result = nominal.copy()
    result[:6] = np.clip(raw, -1.0, 1.0) if clip else raw

    reconstructed_in_next_origin = current_eef_in_current_origin.copy()
    executed_delta = scaled_arm_delta(result)
    reconstructed_in_next_origin[:3, 3] += executed_delta[:3]
    reconstructed_in_next_origin[:3, :3] = (
        axis_angle_to_matrix(executed_delta[3:])
        @ current_eef_in_current_origin[:3, :3]
    )
    reconstructed_world = next_origin @ reconstructed_in_next_origin
    closure_position = float(
        np.linalg.norm(reconstructed_world[:3, 3] - desired_world[:3, 3])
    )
    closure_rotation = desired_world[:3, :3].T @ reconstructed_world[:3, :3]
    closure_angle = float(
        np.arccos(np.clip((np.trace(closure_rotation) - 1.0) / 2.0, -1.0, 1.0))
    )
    return ArmCompensation(
        action=result,
        desired_eef_pose_world=desired_world,
        raw_normalized_arm_delta=raw,
        saturated=saturated,
        transform_closure_pos_error_m=closure_position,
        transform_closure_rot_error_rad=closure_angle,
    )
