#!/usr/bin/env python3
"""SCENE-004 U2 fixed 30+28 reset compiler schedule.

The scanner never calls ``env.step`` or a task success/progress checker.  One
``env.reset`` is issued per scheduled call; source states are then compiled
from the actual 27-pose lattice using simulator geometry and no-actuation
policy forwards.  Camera selection is fixed once per task-cell from seeds
0..2 and is reused without relocation for expansion seeds.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon

from mobiwam.mobipi_actions import nominal_world_intent
from mobiwam.mobipi_checkpoint import create_env_from_checkpoint_metadata, load_policy_from_checkpoint
from mobiwam.mobipi_policy import sample_verified_future_chunk
from mobiwam.scene004 import (
    CELLS,
    TASKS,
    AssistCandidate,
    FixtureFunctionalRecord,
    LatticePose,
    assist_candidates,
    camera_grid,
    canonical_hash,
    dock_candidates,
    fixture_anchored_lattice,
    independent_reason_vector,
    sampled_segment,
    search_lattice,
    select_cell_camera,
    signed_corridor_clearance,
    validate_functional_fixture,
)


SCENE_NAMES = {"CloseSingleDoor": "close_single_door", "CloseDrawer": "close_drawer"}
CHECKPOINTS = {
    "CloseSingleDoor": Path("/share/jhk/MobiWAM/checkpoints/inherited/chensiyu-20260830/robocasa/bc_xfmr/04-12-CloseSingleDoor/seed_1_CloseSingleDoor_mg-300/20250413055045/models/model_epoch_1000.pth"),
    "CloseDrawer": Path("/share/jhk/MobiWAM/checkpoints/MMWAM-OBC-001/robocasa/bc_xfmr/04-12-CloseDrawer/seed_1_CloseDrawer_mg-300/20250413055056/models/model_epoch_1000.pth"),
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def load_task(task: str):
    from robomimic.config import config_factory

    checkpoint_path = CHECKPOINTS[task]
    config_data = json.loads((checkpoint_path.parent.parent / "config.json").read_text())
    config = config_factory(config_data["algo_name"])
    with config.values_unlocked():
        config.update(config_data)
    model, policy, _, env_meta, shape_meta = load_policy_from_checkpoint(config, checkpoint_path)
    return config, model, policy, env_meta, shape_meta


def create_env(config: Any, env_meta: Mapping[str, Any], shape_meta: Mapping[str, Any], cell: int, seed: int):
    meta = copy.deepcopy(env_meta)
    cameras = list(meta["env_kwargs"].get("camera_names", []))
    if "freeview" not in cameras:
        cameras.append("freeview")
    override = {
        "layout_and_style_ids": [[cell, cell]], "seed": seed, "camera_names": cameras,
        "hard_reset": True, "render_gpu_device_id": 0,
    }
    return create_env_from_checkpoint_metadata(config, meta, shape_meta, override)


def origin_pose(raw: Any) -> np.ndarray:
    position, rotation = raw.robots[0].composite_controller.get_controller_base_pose("right")
    value = np.eye(4); value[:3, :3] = rotation; value[:3, 3] = position
    return value


def eef_pose(raw: Any) -> np.ndarray:
    site_id = int(raw.robots[0].eef_site_id["right"])
    value = np.eye(4); value[:3, 3] = raw.sim.data.site_xpos[site_id]
    value[:3, :3] = raw.sim.data.site_xmat[site_id].reshape(3, 3)
    return value


def set_planar_base_pose_without_step(raw: Any, xy_heading: np.ndarray) -> None:
    """Teleport only the three planar joints, then ``forward`` (never step)."""
    x, y, heading = map(float, xy_heading)
    model, data = raw.sim.model, raw.sim.data
    parent_id = model.body_name2id("robot0_base")
    parent_pos = np.asarray(data.body_xpos[parent_id][:2], float)
    parent_quat = np.asarray(data.body_xquat[parent_id], float)
    parent_rotation = Rotation.from_quat(parent_quat[[1, 2, 3, 0]])
    forward_id = model.joint_name2id("mobilebase0_joint_mobile_forward")
    side_id = model.joint_name2id("mobilebase0_joint_mobile_side")
    yaw_id = model.joint_name2id("mobilebase0_joint_mobile_yaw")
    joint_offset = np.array([model.jnt_pos[side_id][0], model.jnt_pos[forward_id][1]])
    target_offset = Rotation.from_euler("z", heading).apply([*joint_offset, 0.0])[:2]
    source_offset = parent_rotation.apply([*joint_offset, 0.0])[:2]
    relative_world = np.array([x, y]) + target_offset - (parent_pos + source_offset)
    relative = (Rotation.from_euler("z", 90, degrees=True) * parent_rotation).apply(
        [relative_world[0], -relative_world[1], 0.0]
    )[:2]
    parent_yaw = parent_rotation.as_euler("xyz")[2]
    values = ((forward_id, relative[0]), (side_id, relative[1]), (yaw_id, heading - parent_yaw))
    for joint_id, value in values:
        data.qpos[int(model.jnt_qposadr[joint_id])] = value
        data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0
    raw.sim.forward()


def refresh_frame_stack(env: Any) -> Mapping[str, np.ndarray]:
    observation = env.env.get_observation()
    env.update_obs(observation, reset=True)
    env.timestep = 0
    env.obs_history = env._get_initial_obs_history(observation)
    return env._get_stacked_obs_from_history()


def no_actuation_policy_query(raw: Any, env: Any, policy: Any, seed: int) -> dict[str, Any]:
    import torch

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    policy.start_episode(lang=str(env._ep_lang_str))
    evidence = sample_verified_future_chunk(policy, env._get_stacked_obs_from_history(), atol=1e-6)
    origin = origin_pose(raw); start = eef_pose(raw); end = start.copy()
    for action in evidence.chunk:
        end = nominal_world_intent(action, origin, end)
    planar = end[:2, 3] - start[:2, 3]
    return {
        "chunk": evidence.chunk, "chunk_sha256": hashlib.sha256(np.ascontiguousarray(evidence.chunk).tobytes()).hexdigest(),
        "chunk_shape": list(evidence.chunk.shape), "first_action_max_abs_error": evidence.max_abs_error,
        "start_ee_pose_world": start, "end_ee_pose_world": end,
        "ee_planar_displacement_m": float(np.linalg.norm(planar)), "no_actuation": True,
    }


def obstacle_geometry(raw: Any, target: Any):
    from mobipi.utils.env_utils import get_fixture_bounds_2d
    from robocasa.models.fixtures import Counter, Fridge, HousingCabinet, Stove, Stovetop

    obstacles = []
    for name, fixture in raw.fixtures.items():
        if fixture is target:
            continue
        if isinstance(fixture, (Counter, Fridge, HousingCabinet, Stove, Stovetop)):
            obstacles.append((name, Polygon(get_fixture_bounds_2d(raw, name))))
    return obstacles, Polygon(get_fixture_bounds_2d(raw, "floor_room"))


def joint_type_name(value: int) -> str:
    return {2: "slide", 3: "hinge"}.get(int(value), f"other_{int(value)}")


def target_fixture_record(task: str, raw: Any) -> tuple[Any, FixtureFunctionalRecord, dict[str, Any]]:
    target = raw.door_fxtr if task == "CloseSingleDoor" else raw.drawer
    fixture_name = next(name for name, fixture in raw.fixtures.items() if fixture is target)
    joints = getattr(target, "joints", {})
    names = tuple(sorted(str(name) for name in (joints.keys() if hasattr(joints, "keys") else joints)))
    types, ranges, qpos = [], [], []
    for name in names:
        joint_id = raw.sim.model.joint_name2id(name)
        types.append(joint_type_name(raw.sim.model.jnt_type[joint_id]))
        ranges.append(tuple(map(float, raw.sim.model.jnt_range[joint_id])))
        qpos.append(float(raw.sim.data.qpos[int(raw.sim.model.jnt_qposadr[joint_id])]))
    handle_name = str(getattr(target, "handle_name", ""))
    handle = None
    if handle_name:
        try:
            handle = tuple(map(float, raw.sim.data.geom_xpos[raw.sim.model.geom_name2id(handle_name)]))
        except Exception:
            try:
                handle = tuple(map(float, raw.sim.data.body_xpos[raw.sim.model.body_name2id(handle_name)]))
            except Exception:
                handle = None
    target_binding = (task == "CloseSingleDoor" and getattr(raw, "door_fxtr", None) is target) or (
        task == "CloseDrawer" and getattr(raw, "drawer", None) is target
    )
    checker_source = inspect.getsource(raw.__class__._check_success)
    checker_binding = target_binding and ("door_fxtr" in checker_source if task == "CloseSingleDoor" else "drawer" in checker_source)
    record = FixtureFunctionalRecord(
        task, fixture_name, f"{target.__class__.__module__}.{target.__class__.__name__}", target_binding,
        names, tuple(types), tuple(ranges), tuple(qpos), handle, checker_binding,
    )
    extra = {
        "world_pos_m": np.asarray(target.pos, float).tolist(), "bbox_size_m": np.asarray(target.size, float).tolist(),
        "handle_name": handle_name, "success_checker_qualname": f"{raw.__class__.__module__}.{raw.__class__.__qualname__}._check_success",
        "success_checker_source_sha256": hashlib.sha256(checker_source.encode()).hexdigest(),
    }
    return target, record, extra


def robot_joint_margin(raw: Any) -> float:
    names = raw.robots[0].robot_joints
    ids = np.asarray([raw.sim.model.joint_name2id(name) for name in names], int)
    limited = np.asarray(raw.sim.model.jnt_limited, bool)[ids]
    ranges = np.asarray(raw.sim.model.jnt_range, float)[ids]
    qpos = np.asarray(raw.sim.data.qpos, float)[np.asarray(raw.sim.model.jnt_qposadr, int)[ids]]
    return float(np.min(np.minimum(qpos[limited] - ranges[limited, 0], ranges[limited, 1] - qpos[limited]))) if np.any(limited) else 1.0


def fixture_axis(raw: Any, record: FixtureFunctionalRecord) -> np.ndarray:
    joint_id = raw.sim.model.joint_name2id(record.joint_names[0])
    body_id = int(raw.sim.model.jnt_bodyid[joint_id])
    axis = raw.sim.data.body_xmat[body_id].reshape(3, 3) @ raw.sim.model.jnt_axis[joint_id]
    if record.task == "CloseDrawer" and np.linalg.norm(axis[:2]) > 1e-6:
        return np.asarray(axis[:2], float)
    handle = np.asarray(record.handle_position_world, float)
    hinge = raw.sim.data.body_xpos[body_id] + raw.sim.data.body_xmat[body_id].reshape(3, 3) @ raw.sim.model.jnt_pos[joint_id]
    radial = handle[:2] - hinge[:2]
    return np.array([-radial[1], radial[0]])


def angular_alignment(yaw: float, target_xy: np.ndarray, source_xy: np.ndarray) -> float:
    desired = math.atan2(*(target_xy - source_xy)[::-1])
    error = abs((desired - yaw + math.pi) % (2 * math.pi) - math.pi)
    return float(max(0.0, 1.0 - error / math.pi))


def source_clearance(pose: LatticePose, obstacles: list[Any], floor: Any, radius: float) -> dict[str, Any]:
    return signed_corridor_clearance([[pose.x_m, pose.y_m], [pose.x_m, pose.y_m]], obstacles, floor,
                                     base_radius_m=radius, inflation_m=0.05)


def compile_lattice_sources(
    task: str, cell: int, seed: int, raw: Any, fixture: FixtureFunctionalRecord, fixture_extra: Mapping[str, Any],
    obstacles: list[Any], floor: Any, preliminary_policy: Mapping[str, Any], robot_radius: float,
) -> dict[str, Any]:
    fixture_xy = np.asarray(fixture_extra["world_pos_m"][:2], float)
    nominal_xy = np.asarray(raw._init_robot_pos[:2], float)
    nominal_yaw = float(raw._init_robot_ori[-1])
    lattice = fixture_anchored_lattice(fixture_xy, nominal_xy, nominal_yaw)
    try:
        tangent = preliminary_policy["end_ee_pose_world"][:2, 3] - preliminary_policy["start_ee_pose_world"][:2, 3]
        tangent /= np.linalg.norm(tangent)
    except Exception:
        tangent = np.array([math.cos(nominal_yaw), math.sin(nominal_yaw)])
    axis = fixture_axis(raw, fixture)
    joint_margin = max(0.0, robot_joint_margin(raw))

    def evaluator(stratum: str):
        def evaluate(pose: LatticePose) -> Mapping[str, float | bool]:
            xy = np.array([pose.x_m, pose.y_m])
            point = source_clearance(pose, obstacles, floor, robot_radius)
            docks = dock_candidates(fixture_xy, fixture_extra["bbox_size_m"][:2], xy, robot_radius + 0.05)
            d_metrics = [(candidate, signed_corridor_clearance(sampled_segment(candidate.start_xy, candidate.dock_xy), obstacles, floor,
                                                                    base_radius_m=robot_radius, inflation_m=0.05)) for candidate in docks]
            best_d, best_d_metric = max(d_metrics, key=lambda pair: (pair[1]["min_signed_clearance_m"], -pair[0].planned_path_m, pair[0].candidate_id))
            start_ee, end_ee = np.eye(4), np.eye(4); end_ee[0, 3], end_ee[1, 3] = tangent
            assists = assist_candidates(xy, start_ee, end_ee, axis)
            a2 = next(candidate for candidate in assists if candidate.candidate_id == "a2")
            a_metric = signed_corridor_clearance(sampled_segment(a2.start_xy, a2.end_xy), obstacles, floor,
                                                 base_radius_m=robot_radius, inflation_m=0.05)
            distance = float(np.linalg.norm(xy - fixture_xy))
            visibility = angular_alignment(pose.yaw_rad, fixture_xy, xy)
            reachability = float(np.clip(1.0 - max(distance - 0.55, 0.0) / 1.20, 0.0, 1.0))
            hard = bool(point["passed"])
            planned_path = float(np.linalg.norm(xy - nominal_xy))
            if stratum == "E-compatible":
                hard = hard and reachability > 0.20
            elif stratum == "D-required":
                hard = hard and best_d_metric["passed"] and best_d.planned_path_m >= 0.30
                planned_path = best_d.planned_path_m
            else:
                hard = hard and a_metric["passed"] and a2.planned_net_m == 0.40
                planned_path = a2.planned_net_m
            return {"hard_valid": hard, "visibility": visibility, "reachability": reachability,
                    "joint_margin": float(np.clip(joint_margin, 0.0, 1.0)),
                    "intent_error": 1.0 - angular_alignment(pose.yaw_rad, fixture_xy, xy),
                    "planned_path": planned_path, "point_clearance_m": point["min_signed_clearance_m"],
                    "best_d_clearance_m": best_d_metric["min_signed_clearance_m"],
                    "a2_clearance_m": a_metric["min_signed_clearance_m"]}
        return evaluate

    result = {"lattice_size": len(lattice), "lattice_hash": canonical_hash(lattice), "strata": {}}
    for stratum in ("E-compatible", "D-required", "A-required"):
        selected, table = search_lattice(lattice, evaluator(stratum))
        result["strata"][stratum] = {"selected_pose": selected.__dict__, "candidate_search": table,
                                       "selected_candidate_rank": next(i for i, row in enumerate(table) if row["pose"]["pose_id"] == selected.pose_id)}
    return result


def snapshot_in_memory(env: Any, policy_evidence: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    raw = env.unwrapped.env
    generators = {}
    for name in ("rng", "randomized_robot_base_pose_rng", "place_robot_for_nav_rng"):
        generator = getattr(raw, name, None)
        if generator is not None and hasattr(generator, "bit_generator"):
            generators[name] = copy.deepcopy(generator.bit_generator.state)
    return {
        "env_state": copy.deepcopy(env.get_state()), "obs_history": copy.deepcopy(env.obs_history),
        "timestep": int(env.timestep), "python_rng": copy.deepcopy(random.getstate()),
        "numpy_rng": copy.deepcopy(np.random.get_state()), "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
        "env_generators": generators,
        "policy_evidence": {name: value for name, value in policy_evidence.items() if name not in {"chunk", "start_ee_pose_world", "end_ee_pose_world"}},
    }


def save_snapshot(directory: Path, env: Any, snapshot: Mapping[str, Any], camera: Mapping[str, Any]) -> dict[str, Any]:
    raw = env.unwrapped.env
    raw.sim.set_state_from_flattened(np.asarray(snapshot["env_state"]["states"]))
    raw.sim.forward()
    configure_camera(raw, camera)
    final_state = copy.deepcopy(env.get_state())
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.xml").write_text(str(final_state["model"]))
    (directory / "ep_meta.json").write_text(str(final_state.get("ep_meta", "{}")))
    np.save(directory / "sim_state.npy", np.asarray(final_state["states"]))
    history = {key: np.concatenate(list(values), axis=0) for key, values in snapshot["obs_history"].items()}
    np.savez_compressed(directory / "frame_history.npz", **history)
    with (directory / "rng_state.pkl").open("wb") as handle:
        pickle.dump({name: value for name, value in snapshot.items() if name not in {"env_state", "obs_history", "policy_evidence"}}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "model_xml_sha256": sha256(directory / "model.xml"), "sim_state_sha256": sha256(directory / "sim_state.npy"),
        "frame_history_sha256": sha256(directory / "frame_history.npz"), "rng_state_sha256": sha256(directory / "rng_state.pkl"),
        "camera_hash": camera["camera_hash"], "policy_evidence": snapshot["policy_evidence"],
        "restore_config": {"method": "env.reset_to + exact obs/RNG restore", "environment_seed": None},
    }
    metadata["snapshot_hash"] = canonical_hash(metadata)
    write_json(directory / "snapshot-meta.json", metadata)
    return metadata


def configure_camera(raw: Any, camera: Mapping[str, Any]) -> None:
    selected = camera["selected"]["pose"]
    anchor = np.asarray(camera["anchor_xy"], float)
    offset = np.asarray(selected["center_offset_xy"], float)
    camera_id = raw.sim.model.camera_name2id("freeview")
    if int(raw.sim.model.cam_bodyid[camera_id]) != 0:
        raise RuntimeError("freeview camera is not world-frame fixed")
    raw.sim.model.cam_pos[camera_id] = [*(anchor + offset), float(selected["height_m"])]
    raw.sim.model.cam_quat[camera_id] = [1.0, 0.0, 0.0, 0.0]
    raw.sim.model.cam_fovy[camera_id] = float(selected["fov_deg"])
    raw.sim.forward()


def evaluate_camera(raw: Any, points_xyz: np.ndarray, anchor_xy: np.ndarray, robot_radius: float, pose: Any) -> dict[str, Any]:
    from mobiwam.b0_scene_compiler import camera_projection_metrics
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    camera_id = raw.sim.model.camera_name2id("freeview")
    raw.sim.model.cam_pos[camera_id] = [*(anchor_xy + np.asarray(pose.center_offset_xy)), pose.height_m]
    raw.sim.model.cam_quat[camera_id] = [1.0, 0.0, 0.0, 0.0]
    raw.sim.model.cam_fovy[camera_id] = pose.fov_deg
    raw.sim.forward()
    transform = get_camera_transform_matrix(raw.sim, "freeview", 1080, 1920)
    metrics = camera_projection_metrics(points_xyz, transform, width=1920, height=1080, border_fraction=0.05)
    focal = 0.5 * 1080 / math.tan(math.radians(pose.fov_deg) / 2.0)
    diameter = 2.0 * robot_radius * focal / pose.height_m
    metrics["base_projected_diameter_px"] = float(diameter)
    metrics["passed"] = bool(metrics["passed"] and diameter >= 120.0)
    return metrics


def envelope_points(sources: list[Mapping[str, Any]], fixture_extra: Mapping[str, Any]) -> np.ndarray:
    points = []
    for source in sources:
        points.extend(source["geometry"]["envelope_xy"])
    fixture_xy = np.asarray(fixture_extra["world_pos_m"][:2], float)
    points.append(fixture_xy.tolist())
    xy = np.asarray(points, float)
    min_xy, max_xy = xy.min(axis=0) - 0.25, xy.max(axis=0) + 0.25
    corners = np.array([[min_xy[0], min_xy[1]], [min_xy[0], max_xy[1]], [max_xy[0], min_xy[1]], [max_xy[0], max_xy[1]]])
    xy = np.vstack([xy, corners])
    low = np.column_stack([xy, np.full(len(xy), 0.15)])
    high = np.column_stack([xy, np.full(len(xy), 2.10)])
    return np.vstack([low, high])


def compile_source(
    task: str, cell: int, seed: int, stratum: str, selected: Mapping[str, Any], env: Any, policy: Any,
    fixture: FixtureFunctionalRecord, fixture_result: Mapping[str, Any], fixture_extra: Mapping[str, Any],
    obstacles: list[Any], floor: Any, robot_radius: float, scene_pass: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = env.unwrapped.env
    pose = selected["selected_pose"]
    set_planar_base_pose_without_step(raw, np.array([pose["x_m"], pose["y_m"], pose["yaw_rad"]]))
    refresh_frame_stack(env)
    query_seed = int(hashlib.sha256(f"{task}|{cell}|{seed}|{stratum}".encode()).hexdigest()[:8], 16)
    policy_result = no_actuation_policy_query(raw, env, policy, query_seed)
    current_origin = origin_pose(raw); current_eef = eef_pose(raw)
    source_xy = current_origin[:2, 3]
    point = signed_corridor_clearance([source_xy, source_xy], obstacles, floor, base_radius_m=robot_radius, inflation_m=0.05)
    docks = dock_candidates(fixture_extra["world_pos_m"][:2], fixture_extra["bbox_size_m"][:2], source_xy, robot_radius + 0.05)
    d_rows = []
    for candidate in docks:
        corridor = signed_corridor_clearance(sampled_segment(candidate.start_xy, candidate.dock_xy), obstacles, floor,
                                             base_radius_m=robot_radius, inflation_m=0.05)
        d_rows.append({**candidate.__dict__, "corridor": corridor, "hard_valid": bool(corridor["passed"] and candidate.planned_path_m >= 0.30)})
    d_primary = max(d_rows, key=lambda row: (row["corridor"]["min_signed_clearance_m"], -row["planned_path_m"], row["candidate_id"]))
    a_rows = []
    try:
        assists = assist_candidates(source_xy, policy_result["start_ee_pose_world"], policy_result["end_ee_pose_world"], fixture_axis(raw, fixture))
        for candidate in assists:
            corridor = signed_corridor_clearance(sampled_segment(candidate.start_xy, candidate.end_xy), obstacles, floor,
                                                 base_radius_m=robot_radius, inflation_m=0.05)
            a_rows.append({**candidate.__dict__, "corridor": corridor,
                           "hard_valid": bool(corridor["passed"] and (candidate.candidate_id != "a2" or (0.38 <= candidate.planned_net_m <= 0.45 and candidate.chunks >= 3)))})
        a_primary = next(row for row in a_rows if row["candidate_id"] == "a2")
        intent_pass = True
    except ValueError as error:
        a_primary = {"candidate_id": "a2", "planned_net_m": 0.40, "chunks": 4, "hard_valid": False,
                     "corridor": {"passed": False, "min_signed_clearance_m": -float("inf"), "nearest_obstacle": "not_compiled"},
                     "error": str(error)}
        intent_pass = False
    handle = np.asarray(fixture.handle_position_world, float)
    reach_distance = float(np.linalg.norm(current_eef[:3, 3] - handle))
    joint_margin = robot_joint_margin(raw)
    e_pass = bool(point["passed"] and reach_distance <= 1.50)
    predicates = {
        "fixture": bool(fixture_result["passed"]), "source_lattice": True, "source_point_collision": bool(point["passed"]),
        "E_fixed_base_reachability": e_pass, "D_path": bool(d_primary["hard_valid"]), "A_path": bool(a_primary["hard_valid"]),
        "policy_forward": bool(policy_result["first_action_max_abs_error"] <= 1e-6), "policy_EE_intent": intent_pass,
        "joint_margin": bool(joint_margin > 1e-4), "scene_model": bool(scene_pass),
    }
    reason = independent_reason_vector(predicates)
    paths = [source_xy.tolist()]
    paths.extend(sampled_segment(d_primary["start_xy"], d_primary["dock_xy"]).tolist())
    if a_rows:
        for candidate in a_rows:
            paths.extend(sampled_segment(candidate["start_xy"], candidate["end_xy"]).tolist())
    record = {
        "source_id": f"{task}-l{cell}-seed{seed:02d}-{stratum[0].lower()}", "stratum": stratum,
        "source_pose": pose, "fixture_subtype": fixture.fixture_class.rsplit(".", 1)[-1],
        "policy_forward": {name: value for name, value in policy_result.items() if name not in {"chunk", "start_ee_pose_world", "end_ee_pose_world"}},
        "geometry": {"frame": "world", "source_xy": source_xy.tolist(), "E_handle_reach_distance_m": reach_distance,
                     "source_clearance": point, "D_candidates": d_rows, "D_primary": d_primary,
                     "A_candidates": a_rows, "A_primary": a_primary, "joint_margin": joint_margin,
                     "inflation_m": 0.05, "clearance_acceptance_threshold_m": 0.0, "sample_spacing_m": 0.02,
                     "envelope_xy": paths},
        "reason_vector": reason, "camera": {"passed": None, "camera_hash": None},
        "route_outcome_read": False, "env_step_calls": 0,
    }
    return record, snapshot_in_memory(env, policy_result)


def compile_reset_call(task: str, cell: int, seed: int, env: Any, policy: Any, scene_root: Path, raw_dir: Path):
    started = utcnow(); reset_invoked = True
    env.reset()
    raw = env.unwrapped.env
    raw_state = copy.deepcopy(env.get_state())
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / "sim_state.npy", np.asarray(raw_state["states"]))
    (raw_dir / "model.xml").write_text(str(raw_state["model"]))
    raw_meta = {"task": task, "cell": cell, "environment_seed": seed, "sim_state_sha256": sha256(raw_dir / "sim_state.npy"),
                "model_xml_sha256": sha256(raw_dir / "model.xml"), "complete_raw_snapshot": True,
                "route_outcome_read": False, "env_step_calls": 0}
    write_json(raw_dir / "raw-snapshot-meta.json", raw_meta)
    target, fixture, fixture_extra = target_fixture_record(task, raw)
    fixture_result = validate_functional_fixture(fixture)
    obstacles, floor = obstacle_geometry(raw, target)
    radius = float(raw.robots[0].robot_model.base.horizontal_radius)
    preliminary = no_actuation_policy_query(raw, env, policy, seed + 20260905)
    lattice = compile_lattice_sources(task, cell, seed, raw, fixture, fixture_extra, obstacles, floor, preliminary, radius)
    scene = scene_root / SCENE_NAMES[task] / f"layout{cell}_style{cell}"
    scene_files = {
        "point_cloud": list(scene.glob("pc.ply")), "transforms": list(scene.glob("transforms.json")),
        "config": list(scene.glob("model/splatfacto/*/config.yml")),
        "dataparser": list(scene.glob("model/splatfacto/*/dataparser_transforms.json")),
        "checkpoint": list(scene.glob("model/splatfacto/*/nerfstudio_models/step-*.ckpt")),
    }
    scene_pass = all(len(paths) == 1 for paths in scene_files.values())
    sources, snapshots = [], {}
    for stratum in ("E-compatible", "D-required", "A-required"):
        source, snapshot = compile_source(task, cell, seed, stratum, lattice["strata"][stratum], env, policy, fixture,
                                          fixture_result, fixture_extra, obstacles, floor, radius, scene_pass)
        sources.append(source); snapshots[stratum] = snapshot
    group = {
        "task": task, "cell": cell, "layout": cell, "style": cell, "environment_seed": seed,
        "started_at": started, "ended_at": utcnow(), "complete_raw_snapshot": True,
        "fixture": {**fixture.__dict__, **fixture_extra, **fixture_result}, "lattice": lattice,
        "scene_model": {"path": str(scene), "files": {name: [str(path) for path in paths] for name, paths in scene_files.items()}, "passed": scene_pass},
        "robot_radius_m": radius, "sources": sources, "camera": {"passed": None},
        "route_outcome_read": False, "env_step_calls": 0, "reset_invoked": reset_invoked,
    }
    group["envelope_points_xyz"] = envelope_points(sources, fixture_extra).tolist()
    return group, snapshots


def freeze_camera_for_groups(task: str, cell: int, held: list[dict[str, Any]]) -> dict[str, Any]:
    all_points = np.vstack([np.asarray(item["group"]["envelope_points_xyz"], float) for item in held])
    anchor = all_points[:, :2].mean(axis=0)
    evaluations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pose in camera_grid():
        for item in held:
            raw = item["env"].unwrapped.env
            points = np.asarray(item["group"]["envelope_points_xyz"], float)
            evaluations[pose.camera_id].append(evaluate_camera(raw, points, anchor, item["group"]["robot_radius_m"], pose))
    camera = select_cell_camera(f"{task}-l{cell}", evaluations)
    camera["anchor_xy"] = anchor.tolist()
    camera["camera_hash"] = canonical_hash({"cell_key": camera["cell_key"], "anchor_xy": camera["anchor_xy"],
                                             "selected": camera["selected"]["pose"]})
    return camera


def render_and_save_group(root: Path, item: dict[str, Any], camera: Mapping[str, Any]) -> dict[str, Any]:
    env, group, snapshots = item["env"], item["group"], item["snapshots"]
    raw = env.unwrapped.env; configure_camera(raw, camera)
    frame_dir = root / "native-frames"; frame_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root = root / "snapshots"
    for source in group["sources"]:
        stratum = source["stratum"]; snapshot = snapshots[stratum]
        raw.sim.set_state_from_flattened(np.asarray(snapshot["env_state"]["states"])); raw.sim.forward(); configure_camera(raw, camera)
        frame = np.asarray(raw.sim.render(camera_name="freeview", width=1920, height=1080))[::-1]
        if frame.shape != (1080, 1920, 3): raise RuntimeError(f"native render shape mismatch: {frame.shape}")
        frame_path = frame_dir / f"{source['source_id']}.png"; Image.fromarray(frame).save(frame_path)
        snapshot_meta = save_snapshot(snapshot_root / source["source_id"], env, snapshot, camera)
        snapshot_meta["restore_config"]["environment_seed"] = group["environment_seed"]
        write_json(snapshot_root / source["source_id"] / "snapshot-meta.json", snapshot_meta)
        source["snapshot_path"] = str(snapshot_root / source["source_id"])
        source["snapshot_hash"] = snapshot_meta["snapshot_hash"]
        source["camera"] = {"passed": bool(camera["selected"]["passed"]), "camera_hash": camera["camera_hash"],
                            "frame_path": str(frame_path), "frame_sha256": sha256(frame_path), "native_width": 1920,
                            "native_height": 1080, "upscale_ratio": 1.0}
        source["reason_vector"]["predicates"]["camera"] = bool(camera["selected"]["passed"])
        source["reason_vector"] = independent_reason_vector(source["reason_vector"]["predicates"])
    group["camera"] = {"passed": bool(camera["selected"]["passed"]), "camera_hash": camera["camera_hash"],
                       "selected": camera["selected"]["pose"], "anchor_xy": camera["anchor_xy"],
                       "minimum_border_fraction": camera["selected"]["minimum_border_fraction"],
                       "minimum_base_projected_diameter_px": camera["selected"]["minimum_base_projected_diameter_px"]}
    group["complete_seed_group"] = all(source["reason_vector"]["passed"] for source in group["sources"])
    group["independent_failure_reasons"] = sorted({reason for source in group["sources"] for reason in source["reason_vector"]["failure_reasons"]})
    group.pop("envelope_points_xyz", None)
    overlay = topdown_overlay(group)
    overlay_dir = root / "topdown-overlays"; overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / f"{group['task']}-l{group['cell']}-seed{group['environment_seed']:02d}.png"
    overlay.save(overlay_path); group["topdown_overlay"] = {"path": str(overlay_path), "sha256": sha256(overlay_path)}
    return group


def topdown_overlay(group: Mapping[str, Any]) -> Image.Image:
    xy = []
    for source in group["sources"]: xy.extend(source["geometry"]["envelope_xy"])
    points = np.asarray(xy, float); minimum, maximum = points.min(axis=0) - .3, points.max(axis=0) + .3
    image = Image.new("RGB", (1000, 1000), "white"); draw = ImageDraw.Draw(image)
    def pixel(p):
        value = (np.asarray(p) - minimum) / np.maximum(maximum - minimum, 1e-9)
        return int(60 + value[0] * 880), int(940 - value[1] * 880)
    for tick in np.arange(np.floor(minimum[0] * 10) / 10, maximum[0] + .1, .1):
        x = pixel([tick, minimum[1]])[0]; draw.line((x, 60, x, 940), fill=(225, 225, 225), width=1)
    for tick in np.arange(np.floor(minimum[1] * 10) / 10, maximum[1] + .1, .1):
        y = pixel([minimum[0], tick])[1]; draw.line((60, y, 940, y), fill=(225, 225, 225), width=1)
    colours = {"E-compatible": (60, 120, 255), "D-required": (30, 190, 70), "A-required": (255, 130, 20)}
    for source in group["sources"]:
        geometry = source["geometry"]; colour = colours[source["stratum"]]
        d = geometry["D_primary"]; draw.line([pixel(p) for p in sampled_segment(d["start_xy"], d["dock_xy"])], fill=(30, 190, 70), width=7)
        if geometry["A_candidates"]:
            a = geometry["A_primary"]; draw.line([pixel(p) for p in sampled_segment(a["start_xy"], a["end_xy"])], fill=(255, 130, 20), width=7)
        center = pixel(geometry["source_xy"]); draw.ellipse((center[0]-10,center[1]-10,center[0]+10,center[1]+10),fill=colour)
    draw.text((30, 20), f"{group['task']} cell {group['cell']} seed {group['environment_seed']} | 10cm grid", fill="black")
    draw.line((70, 970, pixel([minimum[0]+.5, minimum[1]])[0], 970), fill="black", width=8); draw.text((70, 945), "0.5 m", fill="black")
    return image


def cell_rank(groups: list[Mapping[str, Any]], task: str, cell: int) -> tuple[Any, ...]:
    selected = [group for group in groups if group["task"] == task and group["cell"] == cell]
    complete = sum(bool(group["complete_seed_group"]) for group in selected)
    clearances = []
    borders = []
    margins = []
    for group in selected:
        borders.append(float(group["camera"]["minimum_border_fraction"]))
        for source in group["sources"]:
            geometry = source["geometry"]
            clearances.extend([float(geometry["source_clearance"]["min_signed_clearance_m"]),
                               float(geometry["D_primary"]["corridor"]["min_signed_clearance_m"]),
                               float(geometry["A_primary"]["corridor"]["min_signed_clearance_m"])])
            margins.append(float(geometry["joint_margin"]))
    return (-complete, -min(clearances, default=-float("inf")), -min(borders, default=-float("inf")),
            -min(margins, default=-float("inf")), cell)


def update_status(root: Path, **payload: Any) -> None:
    write_json(root / "status.json", {**payload, "updated_at": utcnow()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args(); root = args.artifact_root; root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    observed = git(repo, "rev-parse", "HEAD")
    if observed != args.code_commit or git(repo, "status", "--porcelain"):
        raise RuntimeError("U2 requires the exact clean pushed U1 code commit")
    reset_calls = 0; reserve_used = 0; all_groups: list[dict[str, Any]] = []
    camera_freezes: dict[tuple[str, int], dict[str, Any]] = {}
    call_ledger = root / "all-cell-reset-ledger.jsonl"
    checkpoint_meta = {}
    for task in TASKS:
        config, model, policy, env_meta, shape_meta = load_task(task)
        checkpoint_meta[task] = {"path": str(CHECKPOINTS[task]), "bytes": CHECKPOINTS[task].stat().st_size, "sha256": sha256(CHECKPOINTS[task])}
        for cell in CELLS:
            held = []
            for seed in range(3):
                attempt = 0
                while True:
                    attempt += 1; env = None; reset_calls += 1; call_started = utcnow()
                    try:
                        env = create_env(config, env_meta, shape_meta, cell, seed)
                        group, snapshots = compile_reset_call(task, cell, seed, env, policy, args.scene_root,
                                                              root / "raw-snapshots" / f"{task}-l{cell}-seed{seed:02d}-call{reset_calls:02d}")
                        held.append({"env": env, "group": group, "snapshots": snapshots})
                        append_jsonl(call_ledger, {"call_index": reset_calls, "phase": "screening", "task": task, "cell": cell,
                                                   "environment_seed": seed, "attempt": attempt, "started_at": call_started,
                                                   "ended_at": utcnow(), "status": "complete_raw_snapshot", "env_step_calls": 0,
                                                   "route_outcome_read": False})
                        break
                    except Exception as error:
                        append_jsonl(call_ledger, {"call_index": reset_calls, "phase": "screening", "task": task, "cell": cell,
                                                   "environment_seed": seed, "attempt": attempt, "started_at": call_started,
                                                   "ended_at": utcnow(), "status": "mechanical_failure", "error_type": type(error).__name__,
                                                   "error": str(error), "traceback": traceback.format_exc(), "env_step_calls": 0,
                                                   "route_outcome_read": False})
                        if env is not None:
                            try: env.close()
                            except Exception: pass
                        if reserve_used >= 6 or attempt >= 5: raise
                        reserve_used += 1
                update_status(root, status="running", unit="U2-screening", reset_calls=reset_calls, reserve_used=reserve_used,
                              last_unit=[task, cell, seed], route_rollouts=0, env_step_calls=0, route_outcome_reads=0)
            camera = freeze_camera_for_groups(task, cell, held); camera_freezes[(task, cell)] = camera
            write_json(root / f"cell-camera-{task}-l{cell}.json", camera)
            for item in held:
                group = render_and_save_group(root, item, camera); all_groups.append(group)
                append_jsonl(root / "screening-records.jsonl", group)
                try: item["env"].close()
                except Exception: pass
            del held; gc.collect()
        del policy, model; gc.collect()

    selected_cells = {task: sorted(CELLS, key=lambda cell: cell_rank(all_groups, task, cell))[:2] for task in TASKS}
    write_json(root / "screening-cell-ranking.json", {
        "ranking_rule": ["complete_seed_groups_desc", "worst_signed_clearance_desc", "fixed_camera_min_border_desc", "joint_margin_desc", "layout_id_asc"],
        "tasks": {task: {"selected_cells": selected_cells[task], "ranked": [{"cell": cell, "rank_key": cell_rank(all_groups, task, cell)}
                                                                       for cell in sorted(CELLS, key=lambda c: cell_rank(all_groups, task, c))]}
                  for task in TASKS}, "route_outcome_reads": 0,
    })

    for task in TASKS:
        config, model, policy, env_meta, shape_meta = load_task(task)
        for cell in selected_cells[task]:
            camera = camera_freezes[(task, cell)]
            for seed in range(3, 10):
                attempt = 0
                while True:
                    attempt += 1; env = None; reset_calls += 1; call_started = utcnow()
                    if reset_calls > 64: raise RuntimeError("64-reset cap exceeded")
                    try:
                        env = create_env(config, env_meta, shape_meta, cell, seed)
                        group, snapshots = compile_reset_call(task, cell, seed, env, policy, args.scene_root,
                                                              root / "raw-snapshots" / f"{task}-l{cell}-seed{seed:02d}-call{reset_calls:02d}")
                        item = {"env": env, "group": group, "snapshots": snapshots}
                        # Expansion is evaluated against the frozen screening camera; never relocate.
                        frozen_evaluations = {}
                        points = np.asarray(group["envelope_points_xyz"], float)
                        selected_pose = next(pose for pose in camera_grid() if pose.camera_id == camera["selected"]["pose"]["camera_id"])
                        metric = evaluate_camera(env.unwrapped.env, points, np.asarray(camera["anchor_xy"]), group["robot_radius_m"], selected_pose)
                        expansion_camera = copy.deepcopy(camera)
                        expansion_camera["selected"]["passed"] = bool(metric["passed"])
                        expansion_camera["selected"]["minimum_border_fraction"] = metric["min_border_fraction"]
                        expansion_camera["selected"]["minimum_base_projected_diameter_px"] = metric["base_projected_diameter_px"]
                        group = render_and_save_group(root, item, expansion_camera); all_groups.append(group)
                        append_jsonl(root / "expansion-records.jsonl", group)
                        append_jsonl(call_ledger, {"call_index": reset_calls, "phase": "expansion", "task": task, "cell": cell,
                                                   "environment_seed": seed, "attempt": attempt, "started_at": call_started,
                                                   "ended_at": utcnow(), "status": "complete_raw_snapshot", "env_step_calls": 0,
                                                   "route_outcome_read": False})
                        env.close(); break
                    except Exception as error:
                        append_jsonl(call_ledger, {"call_index": reset_calls, "phase": "expansion", "task": task, "cell": cell,
                                                   "environment_seed": seed, "attempt": attempt, "started_at": call_started,
                                                   "ended_at": utcnow(), "status": "mechanical_failure", "error_type": type(error).__name__,
                                                   "error": str(error), "traceback": traceback.format_exc(), "env_step_calls": 0,
                                                   "route_outcome_read": False})
                        if env is not None:
                            try: env.close()
                            except Exception: pass
                        if reserve_used >= 6 or attempt >= 5: raise
                        reserve_used += 1
                update_status(root, status="running", unit="U2-expansion", reset_calls=reset_calls, reserve_used=reserve_used,
                              last_unit=[task, cell, seed], route_rollouts=0, env_step_calls=0, route_outcome_reads=0)
        del policy, model; gc.collect()

    selected_summary = {}
    insufficient = False
    for task in TASKS:
        selected_summary[task] = {}
        for cell in selected_cells[task]:
            groups = [group for group in all_groups if group["task"] == task and group["cell"] == cell]
            complete = [group for group in groups if group["complete_seed_group"]]
            def quality(group: Mapping[str, Any]):
                clearances = []
                margins = []
                for source in group["sources"]:
                    g = source["geometry"]; clearances.extend([g["source_clearance"]["min_signed_clearance_m"],
                                                               g["D_primary"]["corridor"]["min_signed_clearance_m"],
                                                               g["A_primary"]["corridor"]["min_signed_clearance_m"]]); margins.append(g["joint_margin"])
                return (-min(clearances), -group["camera"]["minimum_border_fraction"], -min(margins), group["environment_seed"])
            complete.sort(key=quality)
            insufficient |= len(complete) < 6
            selected_summary[task][str(cell)] = {"complete_count": len(complete),
                                                  "ranked_environment_seeds": [group["environment_seed"] for group in complete],
                                                  "development_seed": complete[0]["environment_seed"] if complete else None,
                                                  "integrity_reserve_seed": complete[1]["environment_seed"] if len(complete) > 1 else None}
    if insufficient:
        verdict = "scene_compiler_insufficient"; k = 0; source_freeze = None
    else:
        k = min(8, min(data["complete_count"] - 2 for task in selected_summary.values() for data in task.values()))
        verdict = "u2_source_freeze_pass"
        source_freeze = {"K_per_cell": k, "selected_cells": selected_cells, "tasks": selected_summary,
                         "B0_source_count": 12, "quantitative_source_count": 12 * k}
        write_json(root / "source-freeze-v1.1.json", source_freeze)
    decision = {"machine_verdict": verdict, "reset_calls": reset_calls, "baseline_schedule_calls": 58,
                "mechanical_retry_reserve_used": reserve_used, "selected_cells": selected_cells,
                "selected_cell_summary": selected_summary, "K_per_cell": k, "source_freeze": source_freeze,
                "route_rollouts": 0, "route_outcome_reads": 0, "env_step_calls": 0,
                "later_units": "not_run_gate" if insufficient else "authorized_by_u2_gate"}
    write_json(root / "u2-decision.json", decision); write_json(root / "checkpoint-compatibility.json", checkpoint_meta)
    update_status(root, status="completed", unit="U2", reset_calls=reset_calls, reserve_used=reserve_used,
                  verdict=verdict, route_rollouts=0, env_step_calls=0, route_outcome_reads=0)
    return 0 if not insufficient else 3


if __name__ == "__main__":
    raise SystemExit(main())
