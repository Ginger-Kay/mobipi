"""Outcome-blind SCENE-003 Stage A/B reset scanner.

The scanner never calls ``env.step`` or task success/progress code.  It uses
actual reset fixture, floor, obstacle, camera, checkpoint, and official scene
asset state to compile deterministic E/D/A source geometry.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, Polygon

from mobiwam.b0_scene_compiler import (
    camera_projection_metrics,
    fixture_record,
    sample_segment,
    source_lattice,
    stable_seed,
    validate_fixture,
    validate_native_frame,
)
from mobiwam.mobipi_checkpoint import create_env_from_checkpoint_metadata, load_policy_from_checkpoint
from mobiwam.mobipi_policy import sample_verified_future_chunk


TASKS = ("CloseSingleDoor", "CloseDrawer")
CELLS = (1, 4, 7, 8, 9)
SCENE_NAMES = {"CloseSingleDoor": "close_single_door", "CloseDrawer": "close_drawer"}
CHECKPOINTS = {
    "CloseSingleDoor": Path("/share/jhk/MobiWAM/checkpoints/inherited/chensiyu-20260830/robocasa/bc_xfmr/04-12-CloseSingleDoor/seed_1_CloseSingleDoor_mg-300/20250413055045/models/model_epoch_1000.pth"),
    "CloseDrawer": Path("/share/jhk/MobiWAM/checkpoints/MMWAM-OBC-001/robocasa/bc_xfmr/04-12-CloseDrawer/seed_1_CloseDrawer_mg-300/20250413055056/models/model_epoch_1000.pth"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_task(task: str):
    from robomimic.config import config_factory

    checkpoint_path = CHECKPOINTS[task]
    config_data = json.loads((checkpoint_path.parent.parent / "config.json").read_text())
    config = config_factory(config_data["algo_name"])
    with config.values_unlocked():
        config.update(config_data)
    model, policy, _, env_meta, shape_meta = load_policy_from_checkpoint(config, checkpoint_path)
    return config, model, policy, env_meta, shape_meta


def create_env(config, env_meta: dict, shape_meta: dict, cell: int, seed: int):
    meta = copy.deepcopy(env_meta)
    cameras = list(meta["env_kwargs"].get("camera_names", []))
    if "freeview" not in cameras:
        cameras.append("freeview")
    meta["env_kwargs"].update({
        "layout_and_style_ids": [[cell, cell]],
        "seed": seed,
        "camera_names": cameras,
        "hard_reset": True,
        "render_gpu_device_id": 0,
    })
    return create_env_from_checkpoint_metadata(
        config, meta, shape_meta, {
            "layout_and_style_ids": [[cell, cell]],
            "seed": seed,
            "camera_names": cameras,
            "hard_reset": True,
            "render_gpu_device_id": 0,
        },
    )


def obstacle_geometry(raw):
    from mobipi.utils.env_utils import get_fixture_bounds_2d
    from robocasa.models.fixtures import Counter, Floor, Fridge, HousingCabinet, Stove, Stovetop

    obstacle_types = (Counter, Fridge, HousingCabinet, Stove, Stovetop)
    rows = []
    for name, fixture in raw.fixtures.items():
        if isinstance(fixture, obstacle_types):
            rows.append((name, Polygon(get_fixture_bounds_2d(raw, name))))
    floor = Polygon(get_fixture_bounds_2d(raw, "floor_room"))
    return rows, floor


def clearance(points: np.ndarray, obstacles, floor: Polygon, inflated_radius: float):
    values, nearest = [], []
    safe_floor = floor.buffer(-inflated_radius)
    for xy in points:
        point = Point(float(xy[0]), float(xy[1]))
        floor_clearance = point.distance(safe_floor.boundary) if safe_floor.contains(point) else -point.distance(safe_floor)
        candidates = [(name, point.distance(poly) - inflated_radius) for name, poly in obstacles]
        name, obstacle_clearance = min(candidates, key=lambda row: row[1])
        values.append(min(floor_clearance, obstacle_clearance))
        nearest.append(name if obstacle_clearance <= floor_clearance else "floor_boundary")
    return np.asarray(values), nearest


def configure_camera(raw, points_xy: np.ndarray, robot_radius: float):
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    camera_name = "freeview"
    camera_id = raw.sim.model.camera_name2id(camera_name)
    if int(raw.sim.model.cam_bodyid[camera_id]) != 0:
        raise RuntimeError("freeview is not world-frame attached")
    center = np.mean(points_xy, axis=0)
    raw.sim.model.cam_pos[camera_id] = np.array([center[0], center[1], 3.2])
    raw.sim.model.cam_quat[camera_id] = np.array([1.0, 0.0, 0.0, 0.0])
    raw.sim.model.cam_fovy[camera_id] = 48.0
    raw.sim.forward()
    points = np.column_stack([points_xy, np.full(len(points_xy), 0.20)])
    transform = get_camera_transform_matrix(raw.sim, camera_name, 1080, 1920)
    metrics = camera_projection_metrics(points, transform)
    focal = 0.5 * 1080 / math.tan(math.radians(48.0) / 2.0)
    base_diameter_px = 2.0 * robot_radius * focal / 3.0
    metrics.update({
        "name": camera_name,
        "world_frame": True,
        "pos": raw.sim.data.cam_xpos[camera_id].tolist(),
        "xmat": raw.sim.data.cam_xmat[camera_id].reshape(3, 3).tolist(),
        "fovy_deg": 48.0,
        "native_width": 1920,
        "native_height": 1080,
        "base_projected_diameter_px": float(base_diameter_px),
    })
    if base_diameter_px < 120.0:
        metrics["passed"] = False
    return metrics


def compile_reset(task: str, cell: int, seed: int, env, policy, scene_root: Path, frame_dir: Path):
    env.reset()
    raw = env.unwrapped.env
    policy.start_episode(lang=str(env._ep_lang_str))
    policy_evidence = sample_verified_future_chunk(policy, env._get_stacked_obs_from_history(), atol=1e-6)
    fixture = raw.door_fxtr if task == "CloseSingleDoor" else raw.drawer
    fixture_name = next(name for name, value in raw.fixtures.items() if value is fixture)
    fixture_data = fixture_record(fixture_name, fixture, raw.sim)
    fixture_data["parent_or_housing"] = str(getattr(fixture, "parent", None))
    fixture_data["model_xml_sha256"] = hashlib.sha256(raw.model.get_xml().encode()).hexdigest()
    joint_ranges = {}
    for name in fixture_data["joint_names"]:
        joint_id = raw.sim.model.joint_name2id(name)
        joint_ranges[name] = raw.sim.model.jnt_range[joint_id].tolist()
    fixture_data["joint_ranges"] = joint_ranges
    fixture_pass, fixture_reason = True, None
    try:
        validate_fixture(task, fixture_data)
        if task == "CloseDrawer" and "fridge" in (fixture_name + fixture_data["class"]).lower():
            raise ValueError("fridge drawer is forbidden")
    except ValueError as error:
        fixture_pass, fixture_reason = False, str(error)
    fixture_data.update({"passed": fixture_pass, "rejection_reason": fixture_reason})

    robot_radius = float(raw.robots[0].robot_model.base.horizontal_radius)
    policy_base = np.asarray(raw._init_robot_pos, float)[:2]
    fixture_xy = np.asarray(fixture.pos, float)[:2]
    obstacles, floor = obstacle_geometry(raw)
    inflated_radius = robot_radius + 0.05
    base_outward = policy_base - fixture_xy
    base_outward /= np.linalg.norm(base_outward)
    angle_offsets = (0, 15, -15, 30, -30, 45, -45, 60, -60)
    d_options = []
    fixture_size = np.asarray(fixture.size, float)[:2]
    for angle_deg in angle_offsets:
        outward = Rotation.from_euler("z", angle_deg, degrees=True).apply([*base_outward, 0.0])[:2]
        half_extent = 0.5 * float(np.sum(np.abs(outward) * fixture_size))
        dock = fixture_xy + outward * (half_extent + robot_radius + 0.12)
        d_start = dock + 0.50 * outward
        d_points = sample_segment(d_start, dock)
        d_clearance, d_nearest = clearance(d_points, obstacles, floor, inflated_radius)
        d_options.append((float(np.min(d_clearance)), -abs(angle_deg), angle_deg, dock, d_start, d_clearance, d_nearest))
    _, _, angle_deg, dock, d_start, d_clearance, d_nearest = max(d_options, key=lambda option: option[:2])
    nominal_planar = np.asarray(policy_evidence.chunk[:, :2], float).sum(axis=0)
    nominal_norm = float(np.linalg.norm(nominal_planar))
    intent_valid = nominal_norm >= 1e-4
    if intent_valid:
        local_tangent = nominal_planar / nominal_norm
        heading = float(np.asarray(raw._init_robot_ori).reshape(-1)[-1])
        world_tangent = Rotation.from_euler("z", heading).apply([*local_tangent, 0.0])[:2]
    else:
        world_tangent = np.zeros(2)
    a_start = policy_base
    a_end = a_start + 0.40 * world_tangent
    a_envelope_end = a_start + 0.50 * world_tangent
    a_clearance, a_nearest = clearance(sample_segment(a_start, a_envelope_end), obstacles, floor, inflated_radius)
    robot_joint_ids = np.asarray([raw.sim.model.joint_name2id(name) for name in raw.robots[0].robot_joints], int)
    limited = np.asarray(raw.sim.model.jnt_limited, bool)[robot_joint_ids]
    ranges = np.asarray(raw.sim.model.jnt_range, float)[robot_joint_ids]
    qpos_addresses = np.asarray(raw.sim.model.jnt_qposadr, int)[robot_joint_ids]
    joint_values = np.asarray(raw.sim.data.qpos, float)[qpos_addresses]
    joint_margin = float(np.min(np.minimum(joint_values[limited] - ranges[limited, 0], ranges[limited, 1] - joint_values[limited]))) if np.any(limited) else float("inf")
    kinematic_pass = bool(intent_valid and joint_margin > 1e-4 and np.all(np.isfinite(policy_evidence.chunk)))
    geometry_pass = bool(np.min(d_clearance) >= 0.05 and np.min(a_clearance) >= 0.05 and kinematic_pass)

    context = np.array([[.25, .25], [.25, -.25], [-.25, .25], [-.25, -.25]])
    key_xy = np.vstack([policy_base, d_start, dock, a_start, a_end, fixture_xy, dock + context, d_start + context, a_end + context])
    camera = configure_camera(raw, key_xy, robot_radius)
    frame = np.asarray(raw.sim.render(camera_name="freeview", width=1920, height=1080))[::-1]
    validate_native_frame(frame)
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_dir / f"{task}-l{cell}-s{seed:02d}.png"
    Image.fromarray(frame).save(frame_path)
    camera["frame_path"] = str(frame_path)
    camera["frame_bytes"] = frame_path.stat().st_size
    camera["frame_sha256"] = sha256(frame_path)

    scene = scene_root / SCENE_NAMES[task] / f"layout{cell}_style{cell}"
    scene_files = {
        "point_cloud": list(scene.glob("pc.ply")),
        "transforms": list(scene.glob("transforms.json")),
        "config": list(scene.glob("model/splatfacto/*/config.yml")),
        "dataparser": list(scene.glob("model/splatfacto/*/dataparser_transforms.json")),
        "checkpoint": list(scene.glob("model/splatfacto/*/nerfstudio_models/step-*.ckpt")),
    }
    scene_pass = all(len(paths) == 1 for paths in scene_files.values())
    source_rows = []
    for stratum, xy, margin in (("E-compatible", policy_base, joint_margin), ("D-required", d_start, float(np.min(d_clearance))), ("A-required", a_start, float(np.min(a_clearance)))):
        source_rows.append({
            "source_pose_id": f"{task}-l{cell}-seed{seed:02d}-{stratum[0].lower()}",
            "stratum": stratum,
            "world_xy_m": np.asarray(xy).tolist(),
            "geometry_margin_m": margin,
            "rank_key": [-round(margin, 8), stable_seed(20260903, task, cell, seed, stratum)],
        })
    lattice = source_lattice((cell, cell), stable_seed(20260903, task, cell, seed))
    passed = bool(fixture_pass and geometry_pass and camera["passed"] and scene_pass)
    return {
        "task": task,
        "layout": cell,
        "style": cell,
        "environment_seed": seed,
        "route_outcome_read": False,
        "env_step_calls": 0,
        "fixture": fixture_data,
        "robot_radius_m": robot_radius,
        "geometry": {
            "frame": "world",
            "units": "m/rad",
            "policy_base_xy": policy_base.tolist(),
            "dock_xy": dock.tolist(),
            "d_start_xy": d_start.tolist(),
            "d_distance_m": 0.50,
            "a_primary": "a2",
            "a_start_xy": a_start.tolist(),
            "a_endpoint_xy": a_end.tolist(),
            "a_planned_net_m": 0.40,
            "a_chunks": 4,
            "front_direction_offset_deg": angle_deg,
            "nominal_planar_action_sum": nominal_planar.tolist(),
            "nominal_intent_norm": nominal_norm,
            "nominal_world_tangent": world_tangent.tolist(),
            "joint_limit_margin": joint_margin,
            "kinematic_dry_run_passed": kinematic_pass,
            "inflation_m": 0.05,
            "sample_spacing_m": 0.02,
            "d_min_clearance_m": float(np.min(d_clearance)),
            "d_nearest_obstacle": d_nearest[int(np.argmin(d_clearance))],
            "a_min_clearance_m": float(np.min(a_clearance)),
            "a_nearest_obstacle": a_nearest[int(np.argmin(a_clearance))],
            "passed": geometry_pass,
        },
        "camera": camera,
        "scene_model": {"path": str(scene), "files": {key: [str(p) for p in value] for key, value in scene_files.items()}, "passed": scene_pass},
        "controlled_lattice_size": len(lattice),
        "policy_forward": {
            "future_chunk_shape": list(policy_evidence.chunk.shape),
            "future_chunk_sha256": hashlib.sha256(np.ascontiguousarray(policy_evidence.chunk).tobytes()).hexdigest(),
            "first_action_max_abs_error": policy_evidence.max_abs_error,
            "no_actuation": True,
        },
        "sources": source_rows,
        "status": "eligible" if passed else ("ineligible_fixture" if not fixture_pass else "ineligible_hard_geometry_or_camera"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--stage-a-start", type=int, default=0)
    parser.add_argument("--stage-a-end", type=int, default=31)
    parser.add_argument("--tasks", default=",".join(TASKS))
    args = parser.parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    frame_dir = args.artifact_root / "native-frames"
    records_path = args.artifact_root / "scene-capacity-records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    reset_count = len(rows)
    checkpoint_path = args.artifact_root / "checkpoint-compatibility.json"
    checkpoint_meta = json.loads(checkpoint_path.read_text()).get("checkpoints", {}) if checkpoint_path.exists() else {}
    selected_tasks = tuple(part.strip() for part in args.tasks.split(",") if part.strip())
    if not selected_tasks or any(task not in TASKS for task in selected_tasks):
        raise ValueError(f"unsupported --tasks selection: {selected_tasks}")
    for task in selected_tasks:
        config, model, policy, env_meta, shape_meta = load_task(task)
        checkpoint_meta[task] = {"path": str(CHECKPOINTS[task]), "bytes": CHECKPOINTS[task].stat().st_size, "sha256": sha256(CHECKPOINTS[task])}
        for cell in CELLS:
            for seed in range(args.stage_a_start, args.stage_a_end + 1):
                started = datetime.now(timezone.utc).isoformat()
                env = None
                try:
                    env = create_env(config, env_meta, shape_meta, cell, seed)
                    row = compile_reset(task, cell, seed, env, policy, args.scene_root, frame_dir)
                    reset_count += 1
                    row.update({"started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(), "attempt": 1})
                except Exception as error:
                    reset_count += 1
                    row = {
                        "task": task, "layout": cell, "style": cell, "environment_seed": seed,
                        "status": "rejected_or_unknown", "error_type": type(error).__name__, "error": str(error),
                        "traceback": traceback.format_exc(), "route_outcome_read": False, "env_step_calls": 0,
                        "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(), "attempt": 1,
                    }
                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    del env
                    gc.collect()
                rows.append(row)
                with records_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                write_json(args.artifact_root / "status.json", {
                    "status": "running", "reset_count": reset_count, "route_rollouts": 0,
                    "last_unit": [task, cell, seed], "updated_at": datetime.now(timezone.utc).isoformat(),
                })
        del policy, model
        gc.collect()

    summary = []
    for task in TASKS:
        for cell in CELLS:
            selected = [row for row in rows if row["task"] == task and row["layout"] == cell]
            eligible = [row for row in selected if row["status"] == "eligible"]
            summary.append({"task": task, "layout": cell, "style": cell, "eligible_reset_sources": len(eligible), "scheduled_resets": len(selected), "cell_qualified": len(eligible) >= 2})
    qualified = {task: sum(row["cell_qualified"] for row in summary if row["task"] == task) for task in TASKS}
    unknown = sum(row["status"] == "rejected_or_unknown" and row.get("error_type") not in {"ValueError"} for row in rows)
    stage_b_complete = {
        task: all(any(row["task"] == task and row["layout"] == cell and row["environment_seed"] == 63 for row in rows) for cell in CELLS)
        for task in TASKS
    }
    if unknown:
        verdict = "implementation_incomplete"
    elif all(value >= 2 for value in qualified.values()):
        common = sum(all(any(row["task"] == task and row["layout"] == cell and row["cell_qualified"] for row in summary) for task in TASKS) for cell in CELLS)
        verdict = "scene_scout_pass_common_layouts" if common >= 2 else "scene_scout_pass_task_specific_layouts"
    elif all(qualified[task] >= 2 or stage_b_complete[task] for task in TASKS):
        verdict = "insufficient_official_test_pool"
    else:
        verdict = "stage_b_required"
    write_json(args.artifact_root / "scene-capacity-inventory.json", {"cells": summary, "qualified_cells": qualified, "reset_count": reset_count})
    write_json(args.artifact_root / "scene-capacity-decision.json", {"machine_verdict": verdict, "reset_count": reset_count, "route_rollouts": 0, "unknown_units": unknown, "qualified_cells": qualified, "stage_b_complete": stage_b_complete})
    write_json(checkpoint_path, {"checkpoints": checkpoint_meta, "status": "load_metadata_pass_policy_forward_separate"})
    write_json(args.artifact_root / "completion.json", {"status": "completed_stage_a", "ended_at": datetime.now(timezone.utc).isoformat(), "reset_count": reset_count, "route_rollouts": 0, "machine_verdict": verdict})
    return 0 if not unknown else 2


if __name__ == "__main__":
    raise SystemExit(main())
