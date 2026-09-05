"""Low-level, zero-outcome MuJoCo planner adapter for frozen snapshots."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from .planner_min import occupancy_lattice_astar, task_space_region_chain, velocity_level_qp


class SnapshotPlanner:
    """Plan against immutable XML/state without constructing or stepping a task."""

    def __init__(self, snapshot: Path, fixture: Mapping[str, Any]):
        self.snapshot = Path(snapshot)
        self.fixture = dict(fixture)
        self.model = mujoco.MjModel.from_xml_path(str(self.snapshot / "model.xml"))
        self.data = mujoco.MjData(self.model)
        state = np.load(self.snapshot / "sim_state.npy")
        expected = 1 + self.model.nq + self.model.nv
        if state.shape != (expected,):
            raise ValueError(f"snapshot state shape {state.shape} does not match {(expected,)}")
        self.data.time = state[0]
        self.data.qpos[:] = state[1 : 1 + self.model.nq]
        self.data.qvel[:] = state[1 + self.model.nq :]
        mujoco.mj_forward(self.model, self.data)
        self.initial_qpos = self.data.qpos.copy()
        self.eef_site = self._id(mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
        joint_names = [
            "mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw",
            *[f"robot0_joint{index}" for index in range(1, 8)],
        ]
        self.joint_ids = np.asarray([self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names], int)
        self.qpos_indices = np.asarray(self.model.jnt_qposadr[self.joint_ids], int)
        self.dof_indices = np.asarray(self.model.jnt_dofadr[self.joint_ids], int)
        robot_prefixes = ("robot0_", "mobilebase0_", "gripper0_")
        self.robot_geom = np.asarray([
            index for index in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith(robot_prefixes)
            and int(self.model.geom_contype[index]) != 0
        ], int)
        robot_set = set(map(int, self.robot_geom))
        self.world_geom = np.asarray([
            index for index in range(self.model.ngeom)
            if index not in robot_set and int(self.model.geom_contype[index]) != 0
        ], int)
        base_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "robot0_link0_collision")
        # robot0_base is the parent of the planar joints and therefore has a
        # zero Jacobian with respect to them. link0 is the first collision body
        # downstream of all three mobile-base joints and is the correct
        # realized planar reference.
        self.base_body = int(self.model.geom_bodyid[base_geom])
        target_prefix = str(self.fixture["joint_names"][0]).rsplit("_", 1)[0]
        self.allowed_target_prefix = target_prefix
        self.baseline_pairs = self._penetrating_robot_collision_pairs()

    def _id(self, kind: Any, name: str) -> int:
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"snapshot model lacks {name}")
        return int(value)

    def reset_qpos(self) -> None:
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def base_xy(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.base_body, :2], float).copy()

    def eef_xyz(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self.eef_site], float).copy()

    def _jacobian(self, *, site: bool) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
        if site:
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.eef_site)
        else:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.base_body)
        return jacp[:, self.dof_indices]

    def _name(self, geom: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or f"geom{geom}"

    def _penetrating_robot_collision_pairs(self) -> set[tuple[str, str]]:
        robot = set(map(int, self.robot_geom))
        pairs = set()
        for contact in self.data.contact:
            if float(contact.dist) >= -1e-5:
                continue
            one, two = int(contact.geom1), int(contact.geom2)
            if one not in robot and two not in robot:
                continue
            if one in robot and two in robot:
                pairs.add(tuple(sorted((self._name(one), self._name(two)))))
                continue
            r, w = (one, two) if one in robot else (two, one)
            world_name = self._name(w)
            robot_name = self._name(r)
            if world_name.startswith(self.allowed_target_prefix) and ("gripper" in robot_name or "hand" in robot_name):
                continue
            pairs.add((robot_name, world_name))
        return pairs

    def collision_receipt(self) -> Mapping[str, Any]:
        current = self._penetrating_robot_collision_pairs()
        new = sorted(current - self.baseline_pairs)
        return {"passed": not new, "new_penetrating_pairs": new, "baseline_pair_count": len(self.baseline_pairs),
                "minimum_geom_distance_m": self.minimum_robot_world_distance()}

    def minimum_robot_world_distance(self, *, distance_cap_m: float = 0.20) -> float:
        """Minimum exact distance among MuJoCo's active collision pairs.

        MuJoCo contact generation is the authoritative broad/narrow-phase
        query for the model's geom types, masks, margins, and gaps. Direct
        ``mj_geomDistance`` calls return negative sentinels for some unsupported
        mesh pairs, so only generated collision-compatible contacts are used.
        Absence of an active pair certifies clearance beyond the model's
        contact margin and is reported at this capped value.
        """
        minimum = float(distance_cap_m)
        robot = set(map(int, self.robot_geom))
        for contact in self.data.contact:
            one, two = int(contact.geom1), int(contact.geom2)
            if one not in robot and two not in robot:
                continue
            if one in robot and two not in robot:
                robot_name, world_name = self._name(one), self._name(two)
            elif two in robot and one not in robot:
                robot_name, world_name = self._name(two), self._name(one)
            else:
                robot_name, world_name = self._name(one), self._name(two)
            if world_name.startswith(self.allowed_target_prefix) and ("gripper" in robot_name or "hand" in robot_name):
                continue
            minimum = min(minimum, float(contact.dist))
        return minimum

    def _set_base_xy(self, target_xy: Sequence[float]) -> bool:
        self.data.qpos[:] = self.initial_qpos
        mujoco.mj_forward(self.model, self.data)
        error = np.asarray(target_xy, float) - self.base_xy()
        jac = self._jacobian(site=False)[:2, :2]
        delta = np.linalg.solve(jac, error)
        self.data.qpos[self.qpos_indices[:2]] += delta
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return bool(np.linalg.norm(self.base_xy() - target_xy) <= 2e-4 and self.collision_receipt()["passed"])

    def plan_base_astar(self, target_xy: Sequence[float], *, resolution_m: float = 0.05) -> np.ndarray:
        self.reset_qpos()
        start = self.base_xy()
        target = np.asarray(target_xy, float)
        lower = np.minimum(start, target) - 0.8
        upper = np.maximum(start, target) + 0.8
        cache: dict[tuple[float, float], bool] = {}
        def free(point: np.ndarray) -> bool:
            key = tuple(np.round(point, 6))
            if key not in cache:
                cache[key] = self._set_base_xy(point)
            return cache[key]
        path = occupancy_lattice_astar(start, target, free, bounds_xy=[lower[0], upper[0], lower[1], upper[1]],
                                       resolution_m=resolution_m, maximum_expansions=20000)
        self.reset_qpos()
        return path

    def fixture_chain(self, task: str):
        joint_name = str(self.fixture["joint_names"][0])
        joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        body_id = int(self.model.jnt_bodyid[joint_id])
        body_rotation = self.data.xmat[body_id].reshape(3, 3)
        origin = self.data.xpos[body_id] + body_rotation @ self.model.jnt_pos[joint_id]
        axis = body_rotation @ self.model.jnt_axis[joint_id]
        handle = np.asarray(self.fixture["handle_position_world"], float)
        current = float(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])
        goal = float(self.fixture["joint_ranges"][0][1])
        return task_space_region_chain(task, handle, origin, axis, current, goal)

    def plan_ik_path(
        self,
        targets_xyz: Sequence[Sequence[float]],
        *,
        allow_base: bool,
        base_target_xy: Sequence[float] | None = None,
        base_path_xy: Sequence[Sequence[float]] | None = None,
        alpha: float = 1.0,
        dt_s: float = 0.05,
        iterations_per_target: int = 30,
    ) -> Mapping[str, Any]:
        qpos_trace = [self.data.qpos[self.qpos_indices].copy()]
        base_trace = [self.base_xy()]
        eef_trace = [self.eef_xyz()]
        collision_rows = [self.collision_receipt()]
        solver_rows = []
        previous_velocity = np.zeros(10)
        minimum_joint_margin = float("inf")
        minimum_manipulability = float("inf")
        base_path = None if base_path_xy is None else np.asarray(base_path_xy, float)
        if base_path is not None and (base_path.ndim != 2 or base_path.shape[1] != 2 or len(base_path) < 2):
            raise ValueError("base path must be an [N,2] planned trajectory")
        for index, target in enumerate(np.asarray(targets_xyz, float)):
            active_base_target = None
            if allow_base:
                if base_path is not None:
                    path_index = int(round((index + 1) * (len(base_path) - 1) / len(targets_xyz)))
                    active_base_target = base_path[path_index]
                elif base_target_xy is not None:
                    active_base_target = np.asarray(base_target_xy, float)
            for _ in range(iterations_per_target):
                error = target - self.eef_xyz()
                terminal_base_error = (float("inf") if allow_base and active_base_target is not None
                                       else 0.0)
                if allow_base and active_base_target is not None:
                    terminal_base_error = float(np.linalg.norm(active_base_target - self.base_xy()))
                if np.linalg.norm(error) <= 0.01 and terminal_base_error <= 0.01:
                    break
                eef_jac = self._jacobian(site=True)
                rows = [eef_jac]
                desired = [np.clip(error / dt_s, -0.20, 0.20)]
                if allow_base and active_base_target is not None:
                    base_error = active_base_target - self.base_xy()
                    base_jac = self._jacobian(site=False)[:2]
                    rows.append(0.8 * base_jac)
                    desired.append(0.8 * np.clip(base_error / dt_s, -0.15, 0.15))
                jac = np.vstack(rows)
                twist = np.concatenate(desired)
                lower = np.full(10, -0.35); upper = np.full(10, 0.35)
                lower[:3] = -0.20; upper[:3] = 0.20
                if not allow_base:
                    lower[:3] = 0.0; upper[:3] = 0.0
                acceleration_limit = np.r_[np.full(3, 0.50), np.full(7, 1.0)]
                lower = np.maximum(lower, previous_velocity - acceleration_limit * dt_s)
                upper = np.minimum(upper, previous_velocity + acceleration_limit * dt_s)
                velocity, receipt = velocity_level_qp(jac, twist, lower, upper, base_weight=max(alpha, 1e-3), damping=1e-3)
                old_qpos = self.data.qpos[self.qpos_indices].copy()
                target_qpos = old_qpos + dt_s * velocity
                segment_collision = None
                subdivisions = max(1, int(math.ceil(float(np.max(np.abs(target_qpos - old_qpos))) / 0.005)))
                for substep in range(1, subdivisions + 1):
                    self.data.qpos[self.qpos_indices] = old_qpos + (target_qpos - old_qpos) * (substep / subdivisions)
                    self.data.qvel[:] = 0.0
                    mujoco.mj_forward(self.model, self.data)
                    segment_collision = self.collision_receipt()
                    if not segment_collision["passed"]:
                        break
                for joint_id, qpos_index in zip(self.joint_ids, self.qpos_indices):
                    if self.model.jnt_limited[joint_id]:
                        lo, hi = self.model.jnt_range[joint_id]
                        self.data.qpos[qpos_index] = np.clip(self.data.qpos[qpos_index], lo, hi)
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                qpos_trace.append(self.data.qpos[self.qpos_indices].copy())
                base_trace.append(self.base_xy())
                eef_trace.append(self.eef_xyz())
                collision_rows.append(segment_collision or self.collision_receipt())
                solver_rows.append(receipt)
                previous_velocity = velocity
                arm_jac = self._jacobian(site=True)[:, 3:]
                singular = np.linalg.svd(arm_jac, compute_uv=False)
                minimum_manipulability = min(minimum_manipulability, float(np.prod(singular)))
                margins = []
                for joint_id, qpos_index in zip(self.joint_ids[3:], self.qpos_indices[3:]):
                    if self.model.jnt_limited[joint_id]:
                        lo, hi = self.model.jnt_range[joint_id]
                        margins.append(min(self.data.qpos[qpos_index] - lo, hi - self.data.qpos[qpos_index]))
                minimum_joint_margin = min(minimum_joint_margin, min(margins, default=1.0))
                if not collision_rows[-1]["passed"]:
                    break
            if not collision_rows[-1]["passed"]:
                break
        qpos_trace = np.asarray(qpos_trace); base_trace = np.asarray(base_trace); eef_trace = np.asarray(eef_trace)
        final_error = float(np.linalg.norm(np.asarray(targets_xyz, float)[-1] - eef_trace[-1]))
        base_path = float(np.linalg.norm(np.diff(base_trace, axis=0), axis=1).sum())
        base_net = float(np.linalg.norm(base_trace[-1] - base_trace[0]))
        minimum_clearance = min((float(row["minimum_geom_distance_m"]) for row in collision_rows), default=float("-inf"))
        return {
            "hard_valid": bool(final_error <= 0.01 and minimum_clearance >= 0.0 and all(row["passed"] for row in collision_rows)
                               and minimum_joint_margin >= 0.0 and all(row["feasible"] for row in solver_rows)),
            "qpos_trace": qpos_trace,
            "base_trace": base_trace,
            "eef_trace": eef_trace,
            "final_eef_error_m": final_error,
            "planned_base_path_m": base_path,
            "planned_base_net_m": base_net,
            "minimum_joint_margin_rad": float(minimum_joint_margin),
            "minimum_manipulability": float(minimum_manipulability),
            "solver_residual_max": max((float(row["residual"]) for row in solver_rows), default=float("inf")),
            "minimum_continuous_clearance_m": minimum_clearance,
            "new_collision_pairs": sorted({tuple(pair) for row in collision_rows for pair in row["new_penetrating_pairs"]}),
            "steps": len(qpos_trace) - 1,
            "dt_s": dt_s,
        }
