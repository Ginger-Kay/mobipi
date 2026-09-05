"""Deterministic planner primitives for PLANNER-MIN-v1.5.1.

This module is outcome-blind. It operates on pre-outcome geometry, kinematics,
nominal EEF intent, and policy-view compatibility. Environment execution and
task success/progress are deliberately outside its API.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PlannerLimits:
    grid_resolution_m: float = 0.025
    continuous_spacing_m: float = 0.01
    integration_dt_s: float = 0.05
    horizon_steps: int = 8
    maximum_iterations: int = 80
    damping: float = 1e-3
    position_tolerance_m: float = 0.01
    solver_tolerance: float = 1e-8


@dataclass(frozen=True)
class TaskSpaceRegionChain:
    task: str
    approach_points_world: np.ndarray
    precontact_points_world: np.ndarray
    manipulation_points_world: np.ndarray
    joint_type: str


def task_space_region_chain(
    task: str,
    handle_world: Sequence[float],
    joint_origin_world: Sequence[float],
    joint_axis_world: Sequence[float],
    joint_start: float,
    joint_goal: float,
    *,
    samples: int = 16,
) -> TaskSpaceRegionChain:
    """Build approach, pre-contact, and actual hinge/prismatic manifolds."""
    if task not in ("CloseDrawer", "CloseSingleDoor") or samples < 3:
        raise ValueError("unsupported task or insufficient TSR samples")
    handle = np.asarray(handle_world, float)
    origin = np.asarray(joint_origin_world, float)
    axis = np.asarray(joint_axis_world, float)
    if handle.shape != (3,) or origin.shape != (3,) or axis.shape != (3,):
        raise ValueError("TSR geometry must be three-dimensional")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        raise ValueError("fixture joint axis is degenerate")
    axis /= axis_norm
    values = np.linspace(float(joint_start), float(joint_goal), samples)
    if task == "CloseDrawer":
        manifold = handle[None, :] + (values - values[0])[:, None] * axis[None, :]
        joint_type = "prismatic"
    else:
        radial = handle - origin
        manifold = []
        for value in values - values[0]:
            cross = np.cross(axis, radial)
            rotated = radial * math.cos(value) + cross * math.sin(value) + axis * np.dot(axis, radial) * (1 - math.cos(value))
            manifold.append(origin + rotated)
        manifold = np.asarray(manifold)
        joint_type = "hinge"
    if task == "CloseDrawer":
        # The prismatic axis is the drawer-front normal. An open drawer has a
        # negative joint displacement in RoboCasa, so its free-space approach
        # direction is opposite the closing axis.
        approach_direction = -axis.copy()
    else:
        # For a hinged panel, the handle radial lies in the panel plane and
        # axis x radial is its normal.
        approach_direction = np.cross(axis, manifold[0] - origin)
    norm = float(np.linalg.norm(approach_direction))
    if norm < 1e-9:
        raise ValueError("handle does not define an approach direction")
    approach_direction /= norm
    precontact = manifold[0] + 0.04 * approach_direction
    approach = np.stack((manifold[0] + 0.12 * approach_direction, precontact))
    return TaskSpaceRegionChain(task, approach, precontact[None, :], np.asarray(manifold), joint_type)


def occupancy_lattice_astar(
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    is_free: Callable[[np.ndarray], bool],
    *,
    bounds_xy: Sequence[float],
    resolution_m: float = 0.025,
    maximum_expansions: int = 200000,
) -> np.ndarray:
    """Deterministic 8-connected A* for the live-confirmed holonomic base."""
    if resolution_m <= 0 or len(bounds_xy) != 4:
        raise ValueError("invalid A* lattice specification")
    xmin, xmax, ymin, ymax = map(float, bounds_xy)
    origin = np.array([xmin, ymin])
    shape = np.floor((np.array([xmax, ymax]) - origin) / resolution_m).astype(int) + 1
    def cell(point: Sequence[float]) -> tuple[int, int]:
        value = np.rint((np.asarray(point, float) - origin) / resolution_m).astype(int)
        return int(value[0]), int(value[1])
    def point(node: tuple[int, int]) -> np.ndarray:
        return origin + resolution_m * np.asarray(node, float)
    start, goal = cell(start_xy), cell(goal_xy)
    if any(v < 0 for v in (*start, *goal)) or start[0] >= shape[0] or goal[0] >= shape[0] or start[1] >= shape[1] or goal[1] >= shape[1]:
        raise ValueError("A* endpoints lie outside frozen bounds")
    if not is_free(point(start)) or not is_free(point(goal)):
        raise ValueError("A* endpoint is hard-invalid")
    moves = tuple(sorted(((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0))))
    queue: list[tuple[float, float, int, int]] = [(float(np.linalg.norm(np.subtract(start, goal))), 0.0, *start)]
    cost = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    expanded = 0
    while queue:
        _, distance, x, y = heapq.heappop(queue)
        node = (x, y)
        if distance != cost.get(node):
            continue
        if node == goal:
            chain = [node]
            while chain[-1] != start:
                chain.append(parent[chain[-1]])
            return np.stack([point(item) for item in reversed(chain)])
        expanded += 1
        if expanded > maximum_expansions:
            raise RuntimeError("A* frozen expansion cap exhausted")
        for dx, dy in moves:
            nxt = (x + dx, y + dy)
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] >= shape[0] or nxt[1] >= shape[1] or not is_free(point(nxt)):
                continue
            step = math.hypot(dx, dy)
            candidate = distance + step
            if candidate + 1e-12 < cost.get(nxt, float("inf")):
                cost[nxt] = candidate
                parent[nxt] = node
                heuristic = math.hypot(nxt[0] - goal[0], nxt[1] - goal[1])
                heapq.heappush(queue, (candidate + heuristic, candidate, *nxt))
    raise RuntimeError("A* found no continuous-free base route")


def densify_path(points: Sequence[Sequence[float]], spacing_m: float) -> np.ndarray:
    points = np.asarray(points, float)
    if points.ndim != 2 or len(points) < 1 or spacing_m <= 0:
        raise ValueError("invalid path densification input")
    result = [points[0]]
    for start, end in zip(points, points[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(distance / spacing_m)))
        result.extend(start + (end - start) * (index / count) for index in range(1, count + 1))
    return np.asarray(result)


def continuous_path_metrics(
    knots: Sequence[Sequence[float]],
    clearance: Callable[[np.ndarray], float],
    *,
    spacing_m: float,
    velocity_limit: float,
    acceleration_limit: float,
    dt_s: float,
) -> Mapping[str, Any]:
    dense = densify_path(knots, spacing_m)
    clearances = np.asarray([clearance(point) for point in dense], float)
    velocities = np.diff(dense, axis=0) / dt_s
    accelerations = np.diff(velocities, axis=0) / dt_s
    max_velocity = float(np.max(np.linalg.norm(velocities, axis=1))) if len(velocities) else 0.0
    max_acceleration = float(np.max(np.linalg.norm(accelerations, axis=1))) if len(accelerations) else 0.0
    path = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()) if len(dense) > 1 else 0.0
    return {
        "passed": bool(np.all(clearances >= 0.0) and max_velocity <= velocity_limit and max_acceleration <= acceleration_limit),
        "minimum_continuous_clearance_m": float(clearances.min()),
        "maximum_velocity": max_velocity,
        "maximum_acceleration": max_acceleration,
        "total_path_m": path,
        "sample_count": len(dense),
        "spacing_m": float(spacing_m),
    }


def velocity_level_qp(
    jacobian: np.ndarray,
    desired_twist: Sequence[float],
    lower_velocity: Sequence[float],
    upper_velocity: Sequence[float],
    *,
    base_weight: float,
    damping: float,
    inequalities: tuple[np.ndarray, np.ndarray] | None = None,
    projection_iterations: int = 32,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Solve the frozen convex velocity QP with deterministic projections."""
    jacobian = np.asarray(jacobian, float)
    target = np.asarray(desired_twist, float)
    lower, upper = np.asarray(lower_velocity, float), np.asarray(upper_velocity, float)
    if jacobian.ndim != 2 or target.shape != (jacobian.shape[0],) or lower.shape != upper.shape or lower.shape != (jacobian.shape[1],):
        raise ValueError("velocity QP dimensions are inconsistent")
    weights = np.ones(jacobian.shape[1]); weights[:3] = float(base_weight)
    hessian = jacobian.T @ jacobian + float(damping) * np.diag(weights)
    rhs = jacobian.T @ target
    velocity = np.clip(np.linalg.solve(hessian, rhs), lower, upper)
    if inequalities is not None:
        matrix, bound = map(lambda value: np.asarray(value, float), inequalities)
        for _ in range(projection_iterations):
            changed = False
            for normal, minimum in zip(matrix, bound):
                deficit = float(minimum - np.dot(normal, velocity))
                norm = float(np.dot(normal, normal))
                if deficit > 0 and norm > 1e-12:
                    velocity = np.clip(velocity + deficit * normal / norm, lower, upper)
                    changed = True
            if not changed:
                break
    residual = float(np.linalg.norm(jacobian @ velocity - target))
    feasible = inequalities is None or bool(np.all(inequalities[0] @ velocity >= inequalities[1] - 1e-8))
    return velocity, {"solver": "deterministic_projected_dense_qp", "residual": residual, "feasible": feasible}


def rank_primary(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    hard = [row for row in candidates if bool(row.get("hard_valid"))]
    if not hard:
        raise ValueError("candidate family has no hard-valid primary")
    required = (
        "minimum_continuous_clearance_m", "minimum_manipulability_or_joint_margin",
        "minimum_policy_view_compatibility", "total_planned_base_path_m", "total_planned_time_s", "candidate_id",
    )
    for row in hard:
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"primary candidate lacks frozen ranking fields: {missing}")
    return min(hard, key=lambda row: (
        -float(row["minimum_continuous_clearance_m"]),
        -float(row["minimum_manipulability_or_joint_margin"]),
        -float(row["minimum_policy_view_compatibility"]),
        float(row["total_planned_base_path_m"]),
        float(row["total_planned_time_s"]),
        str(row["candidate_id"]),
    ))
