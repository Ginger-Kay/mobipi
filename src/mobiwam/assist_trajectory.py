from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarAssistTrajectory:
    poses_world: np.ndarray
    translation_m: float
    yaw_rad: float
    fraction_toward_dock: float


def _yaw(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def build_truncated_assist_trajectory(
    start_pose_world: np.ndarray,
    dock_pose_world: np.ndarray,
    *,
    steps: int = 10,
    fraction_toward_dock: float = 0.25,
    max_translation_m: float = 0.05,
    max_yaw_rad: float = np.deg2rad(3.0),
) -> PlanarAssistTrajectory:
    """Create canonical A by taking a short prefix toward Mobi-pi's D pose."""

    start = np.asarray(start_pose_world, dtype=np.float64)
    dock = np.asarray(dock_pose_world, dtype=np.float64)
    if start.shape != (4, 4) or dock.shape != (4, 4):
        raise ValueError("start and dock poses must have shape (4, 4)")
    if steps <= 0 or not 0.0 <= fraction_toward_dock <= 1.0:
        raise ValueError("invalid assist horizon or dock fraction")
    if min(max_translation_m, max_yaw_rad) < 0.0:
        raise ValueError("assist limits must be non-negative")

    translation = (dock[:2, 3] - start[:2, 3]) * fraction_toward_dock
    translation_norm = float(np.linalg.norm(translation))
    if translation_norm > max_translation_m > 0.0:
        translation *= max_translation_m / translation_norm
    elif max_translation_m == 0.0:
        translation[:] = 0.0

    yaw_delta = _wrap(_yaw(dock[:3, :3]) - _yaw(start[:3, :3]))
    yaw_delta = float(np.clip(yaw_delta * fraction_toward_dock, -max_yaw_rad, max_yaw_rad))
    start_yaw = _yaw(start[:3, :3])

    poses = []
    for index in range(steps + 1):
        alpha = index / steps
        pose = start.copy()
        pose[:2, 3] = start[:2, 3] + alpha * translation
        yaw = start_yaw + alpha * yaw_delta
        cosine, sine = np.cos(yaw), np.sin(yaw)
        pose[:3, :3] = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        poses.append(pose)

    return PlanarAssistTrajectory(
        poses_world=np.stack(poses),
        translation_m=float(np.linalg.norm(translation)),
        yaw_rad=yaw_delta,
        fraction_toward_dock=fraction_toward_dock,
    )
