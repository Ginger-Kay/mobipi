"""Utilities for reproducible paired candidate rollouts.

This module intentionally keeps configuration, fingerprinting, and manifest logic
independent from GPU execution so it can be tested on CPU-only hosts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


REQUIRED_MANIFEST_PATHS = (
    "experiment_id",
    "run_id",
    "scene_group_id",
    "candidate.candidate_id",
    "research.commit",
    "code.commit",
    "environment.python_executable",
    "protocol.config_uri",
    "protocol.config_sha256",
    "seeds.checkpoint_seed",
    "seeds.environment_seed",
    "seeds.candidate_seed",
    "seeds.evaluation_seed",
    "execution.command",
    "execution.output_root",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_candidate_config(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with open(path, "r") as stream:
        config = json.load(stream)
    if config.get("schema_version") != "1.0":
        raise ValueError("candidate config schema_version must be '1.0'")
    if config.get("review_status") not in {"proposed", "frozen"}:
        raise ValueError("candidate config review_status must be proposed or frozen")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate config must contain a non-empty candidates list")
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("every candidate requires candidate_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id values must be unique")
    return config


def verify_config_sha256(path: os.PathLike[str] | str, expected: Optional[str]) -> str:
    actual = sha256_file(path)
    if expected and actual.lower() != expected.lower():
        raise ValueError(f"candidate config SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def resolve_seeds(
    checkpoint_seed: Optional[int],
    legacy_seed: Optional[int],
    environment_seed: int,
    candidate_seed: int,
    evaluation_seed: int,
) -> Tuple[Dict[str, int], Optional[str]]:
    warning = None
    if checkpoint_seed is None:
        if legacy_seed is None:
            raise ValueError("--checkpoint_seed is required when legacy --seed is absent")
        checkpoint_seed = legacy_seed
        warning = (
            "legacy --seed maps only to checkpoint_seed; environment_seed, "
            "candidate_seed, and evaluation_seed remain independent"
        )
    elif legacy_seed is not None and legacy_seed != checkpoint_seed:
        raise ValueError("legacy --seed conflicts with --checkpoint_seed")
    return {
        "checkpoint_seed": int(checkpoint_seed),
        "environment_seed": int(environment_seed),
        "candidate_seed": int(candidate_seed),
        "evaluation_seed": int(evaluation_seed),
    }, warning


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return vector


def validate_candidate(candidate: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    transform = _finite_vector(candidate.get("requested_transform"), 3, "requested_transform")
    frame = candidate.get("coordinate_frame", config.get("coordinate_frame"))
    units = candidate.get("units", config.get("units"))
    if frame not in {"world", "nominal_base"}:
        raise ValueError(f"unsupported coordinate_frame: {frame}")
    if units != {"translation": "m", "yaw": "rad"}:
        raise ValueError("units must be {'translation': 'm', 'yaw': 'rad'}")
    bounds = config.get("bounds", {})
    maxima = np.array(
        [
            float(bounds.get("max_abs_dx_m", 0.0)),
            float(bounds.get("max_abs_dy_m", 0.0)),
            float(bounds.get("max_abs_dyaw_rad", 0.0)),
        ]
    )
    if np.any(maxima <= 0):
        raise ValueError("candidate config bounds must be positive")
    if np.any(np.abs(transform) > maxima + 1e-12):
        raise ValueError(
            f"candidate {candidate.get('candidate_id')} exceeds bounds: "
            f"abs({transform.tolist()}) > {maxima.tolist()}"
        )


def requested_pose(
    nominal_pose: Sequence[float],
    transform: Sequence[float],
    coordinate_frame: str,
) -> np.ndarray:
    nominal = _finite_vector(nominal_pose, 3, "nominal_pose")
    delta = _finite_vector(transform, 3, "requested_transform")
    if coordinate_frame == "world":
        translation = delta[:2]
    elif coordinate_frame == "nominal_base":
        yaw = nominal[2]
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        translation = rotation @ delta[:2]
    else:
        raise ValueError(f"unsupported coordinate_frame: {coordinate_frame}")
    result = nominal.copy()
    result[:2] += translation
    result[2] = wrap_angle(result[2] + delta[2])
    return result


def wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)


def pose_error(requested: Sequence[float], actual: Sequence[float]) -> Dict[str, Any]:
    requested_array = _finite_vector(requested, 3, "requested_pose")
    actual_array = _finite_vector(actual, 3, "actual_pose")
    delta = actual_array - requested_array
    delta[2] = wrap_angle(delta[2])
    return {
        "delta": delta.tolist(),
        "translation_l2_m": float(np.linalg.norm(delta[:2])),
        "abs_yaw_rad": float(abs(delta[2])),
    }


def _normalized(value: Any, decimals: int) -> Any:
    if isinstance(value, np.ndarray):
        return _normalized(value.tolist(), decimals)
    if isinstance(value, np.generic):
        return _normalized(value.item(), decimals)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return round(value, decimals)
    if isinstance(value, Mapping):
        return {str(key): _normalized(value[key], decimals) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalized(item, decimals) for item in value]
    return value


def stable_hash(payload: Any, tolerance: float = 1e-6) -> Tuple[str, Any]:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    decimals = max(0, int(math.ceil(-math.log10(tolerance))))
    normalized = _normalized(payload, decimals)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), normalized


def compare_payloads(first: Any, second: Any, tolerance: float) -> Dict[str, Any]:
    """Compare nested reset state with explicit numeric tolerance and exact schema checks."""
    mismatches = []
    numeric_max_abs_diff = 0.0

    def compare(left: Any, right: Any, path: str) -> None:
        nonlocal numeric_max_abs_diff
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if set(left) != set(right):
                mismatches.append(
                    {"path": path, "reason": "keys", "left": sorted(left), "right": sorted(right)}
                )
                return
            for key in sorted(left):
                compare(left[key], right[key], f"{path}/{key}")
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                mismatches.append(
                    {"path": path, "reason": "length", "left": len(left), "right": len(right)}
                )
                return
            try:
                left_array = np.asarray(left, dtype=np.float64)
                right_array = np.asarray(right, dtype=np.float64)
            except (TypeError, ValueError):
                for index, (left_item, right_item) in enumerate(zip(left, right)):
                    compare(left_item, right_item, f"{path}/{index}")
                return
            if left_array.shape != right_array.shape:
                mismatches.append(
                    {
                        "path": path,
                        "reason": "shape",
                        "left": list(left_array.shape),
                        "right": list(right_array.shape),
                    }
                )
                return
            difference = float(np.max(np.abs(left_array - right_array))) if left_array.size else 0.0
            numeric_max_abs_diff = max(numeric_max_abs_diff, difference)
            if difference > tolerance:
                mismatches.append(
                    {"path": path, "reason": "numeric_tolerance", "max_abs_diff": difference}
                )
            return
        if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
            difference = abs(float(left) - float(right))
            numeric_max_abs_diff = max(numeric_max_abs_diff, difference)
            if difference > tolerance:
                mismatches.append(
                    {"path": path, "reason": "numeric_tolerance", "max_abs_diff": difference}
                )
            return
        if left != right:
            mismatches.append({"path": path, "reason": "value", "left": left, "right": right})

    compare(first, second, "")
    return {
        "matched": not mismatches,
        "tolerance": tolerance,
        "numeric_max_abs_diff": numeric_max_abs_diff,
        "mismatches": mismatches[:100],
    }


def array_summary(value: Any) -> Dict[str, Any]:
    array = np.asarray(value)
    summary: Dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size and np.issubdtype(array.dtype, np.number):
        numeric = array.astype(np.float64)
        summary.update(
            min=float(np.min(numeric)),
            max=float(np.max(numeric)),
            mean=float(np.mean(numeric)),
            std=float(np.std(numeric)),
            l2=float(np.linalg.norm(numeric.ravel())),
        )
    return summary


def unwrap_env(env: Any) -> Any:
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "unwrapped"):
            unwrapped = current.unwrapped
            if unwrapped is not current:
                current = unwrapped
                continue
        if hasattr(current, "env"):
            nested = current.env
            if nested is not current:
                current = nested
                continue
        break
    return current


def _fixture_pose(fixture: Any) -> Dict[str, Any]:
    pose: Dict[str, Any] = {"position": np.asarray(fixture.pos).tolist()}
    if hasattr(fixture, "get_quat"):
        pose["quaternion_wxyz"] = np.asarray(fixture.get_quat()).tolist()
    elif hasattr(fixture, "rot"):
        pose["yaw_rad"] = float(fixture.rot)
    return pose


def _fixture_joint_state(raw_env: Any, fixture: Any) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for joint_name in getattr(fixture, "joints", []) or []:
        try:
            address = raw_env.sim.model.get_joint_qpos_addr(joint_name)
            value = raw_env.sim.data.qpos[address]
            if np.asarray(value).size == 1:
                result[joint_name] = float(value)
        except Exception:
            continue
    return result


def collect_reset_fingerprint(
    env: Any,
    observation: Mapping[str, Any],
    task: str,
    layout_id: int,
    style_id: int,
    environment_seed: int,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    from robocasa.utils.env_utils import get_base_pose

    raw_env = unwrap_env(env)
    objects: Dict[str, Any] = {}
    for name, model in sorted(raw_env.objects.items()):
        body_id = raw_env.sim.model.body_name2id(model.root_body)
        objects[name] = {
            "position": np.asarray(raw_env.sim.data.body_xpos[body_id]).tolist(),
            "quaternion_wxyz": np.asarray(raw_env.sim.data.body_xquat[body_id]).tolist(),
            "model_class": type(model).__name__,
        }
    fixtures: Dict[str, Any] = {}
    for name, fixture in sorted(raw_env.fixtures.items()):
        fixtures[name] = {
            **_fixture_pose(fixture),
            "joint_qpos": _fixture_joint_state(raw_env, fixture),
            "model_class": type(fixture).__name__,
        }
    robot = raw_env.robots[0]
    arm_qpos = np.asarray(raw_env.sim.data.qpos[robot._ref_joint_pos_indexes]).tolist()
    observation_summaries = {key: array_summary(value) for key, value in sorted(observation.items())}
    invariant_payload = {
        "task": task,
        "layout_id": int(layout_id),
        "style_id": int(style_id),
        "environment_seed": int(environment_seed),
        "objects": objects,
        "fixtures": fixtures,
        "robot_arm_qpos": arm_qpos,
        "observation_schema": {
            key: {"shape": value["shape"], "dtype": value["dtype"]}
            for key, value in observation_summaries.items()
        },
    }
    full_payload = {
        **deepcopy(invariant_payload),
        "base_pose_world_xy_yaw": get_base_pose(raw_env, unwrapped=True).tolist(),
        "observation_summaries": observation_summaries,
    }
    full_hash, normalized_full = stable_hash(full_payload, tolerance=tolerance)
    invariant_hash, normalized_invariant = stable_hash(invariant_payload, tolerance=tolerance)
    return {
        "schema_version": "1.0",
        "float_tolerance": tolerance,
        "full_hash": full_hash,
        "scene_invariant_hash": invariant_hash,
        "raw": full_payload,
        "raw_scene_invariant": invariant_payload,
        "normalized": normalized_full,
        "normalized_scene_invariant": normalized_invariant,
    }


def robot_collision_pairs(env: Any) -> list[Dict[str, Any]]:
    raw_env = unwrap_env(env)
    pairs = []
    for contact in raw_env.sim.data.contact:
        if contact.dist >= 0:
            continue
        first = raw_env.sim.model.geom_id2name(contact.geom1)
        second = raw_env.sim.model.geom_id2name(contact.geom2)
        if not first or not second:
            continue
        first_robot = first.startswith(("robot0", "mobilebase0"))
        second_robot = second.startswith(("robot0", "mobilebase0"))
        if first_robot == second_robot:
            continue
        other = second if first_robot else first
        if "floor" in other.lower():
            continue
        pairs.append({"geom1": first, "geom2": second, "distance": float(contact.dist)})
    return pairs


def workspace_boundary_check(env: Any, base_pose: Sequence[float]) -> Dict[str, Any]:
    """Check the mobile-base footprint against the floor and fixed obstacles."""
    from robocasa.utils.env_utils import get_robot_xy_corners
    from shapely.geometry import Polygon

    raw_env = unwrap_env(env)
    occupied_bounds, floor_bounds = raw_env._get_env_map()
    robot_size = np.asarray(
        raw_env.robots[0].robot_model.base.horizontal_radius, dtype=np.float64
    ).reshape(-1)
    if robot_size.size == 1:
        robot_size = np.repeat(robot_size, 2)
    robot_size = robot_size[:2]
    robot_polygon = Polygon(get_robot_xy_corners(np.asarray(base_pose), robot_size))
    floor_polygon = Polygon(floor_bounds)
    overlap_areas = [
        float(robot_polygon.intersection(Polygon(bounds)).area)
        for bounds in occupied_bounds
    ]
    return {
        "inside_floor": bool(floor_polygon.covers(robot_polygon)),
        "max_fixture_overlap_area_m2": max(overlap_areas, default=0.0),
        "robot_size_m": robot_size.tolist(),
        "robot_footprint_area_m2": float(robot_polygon.area),
    }


def target_relative_pose(env: Any, base_pose: Sequence[float]) -> Dict[str, Any]:
    """Express the requested/actual robot base pose in the task fixture frame."""
    raw_env = unwrap_env(env)
    fixture = raw_env.door_fxtr
    fixture_position = np.asarray(fixture.pos, dtype=np.float64)
    if hasattr(fixture, "get_quat"):
        w, x, y, z = np.asarray(fixture.get_quat(), dtype=np.float64)
        fixture_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    else:
        fixture_yaw = float(fixture.rot)
    pose = _finite_vector(base_pose, 3, "base_pose")
    world_delta = pose[:2] - fixture_position[:2]
    c, s = math.cos(fixture_yaw), math.sin(fixture_yaw)
    fixture_delta = np.array([[c, s], [-s, c]]) @ world_delta
    return {
        "target_fixture_name": getattr(fixture, "name", "door_fxtr"),
        "target_fixture_class": type(fixture).__name__,
        "coordinate_frame": "target_fixture",
        "units": {"translation": "m", "yaw": "rad"},
        "xy_yaw": [
            float(fixture_delta[0]),
            float(fixture_delta[1]),
            wrap_angle(pose[2] - fixture_yaw),
        ],
    }


def repo_state(repo: os.PathLike[str] | str) -> Dict[str, Any]:
    repo = str(repo)

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", repo, *args], text=True).strip()

    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "remote": git("remote", "get-url", "origin"),
    }


def sanitized_command(argv: Sequence[str]) -> Tuple[str, bool]:
    command = shlex.join(argv)
    secret_pattern = re.compile(
        r"(?i)(authorization|api[_-]?key|token|password|cookie)(=|\s+)([^\s]+)"
    )
    checked = secret_pattern.search(command) is None
    if not checked:
        command = secret_pattern.sub(r"\1\2<REDACTED>", command)
    return command, True


def candidate_run_id(run_id: str, environment_seed: int, candidate_id: str) -> str:
    safe_candidate = re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate_id)
    return f"{run_id}-env{environment_seed:03d}-{safe_candidate}"


def scene_group_id(
    run_id: str,
    task: str,
    layout_id: int,
    style_id: int,
    checkpoint_seed: int,
    environment_seed: int,
) -> str:
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "-", task)
    return (
        f"{run_id}-{safe_task}-l{layout_id}-s{style_id}-"
        f"ckpt{checkpoint_seed}-env{environment_seed:03d}"
    )


def nested_value(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for component in dotted_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = [path for path in REQUIRED_MANIFEST_PATHS if nested_value(manifest, path) in {None, ""}]
    if missing:
        raise ValueError(f"manifest is missing required fields: {missing}")


def artifact_record(path: os.PathLike[str] | str, fmt: str, validation: str) -> Dict[str, Any]:
    path = Path(path)
    return {
        "uri": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": fmt,
        "validation": validation,
    }


def environment_record(torch_module: Any) -> Dict[str, Any]:
    conda_prefix = sys.prefix
    return {
        "conda_name": os.environ.get("CONDA_DEFAULT_ENV", Path(conda_prefix).name),
        "conda_prefix": conda_prefix,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "cuda_version": getattr(torch_module.version, "cuda", None),
        "pytorch_version": torch_module.__version__,
        "key_module_sources": {},
    }


def base_manifest(
    *,
    experiment_id: str,
    run_id: str,
    scene_id: str,
    candidate: Mapping[str, Any],
    research: Mapping[str, Any],
    code: Mapping[str, Any],
    environment: Mapping[str, Any],
    protocol: Mapping[str, Any],
    seeds: Mapping[str, Any],
    command: str,
    output_root: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "scene_group_id": scene_id,
        "status": "prepared",
        "owner": os.environ.get("USER", ""),
        "created_at": utc_now(),
        "started_at": "TBD",
        "ended_at": "TBD",
        "host": {
            "hostname": socket.gethostname(),
            "os": sys.platform,
            "working_root": "/data/worldmodel_jhk_2",
        },
        "research": dict(research),
        "code": dict(code),
        "environment": dict(environment),
        "protocol": dict(protocol),
        "seeds": dict(seeds),
        "candidate": dict(candidate),
        "resources": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_model": "TBD",
            "pid": os.getpid(),
            "tmux_session": os.environ.get("TMUX_SESSION_NAME", ""),
            "tmux_windows": ["rollout", "status", "logs"],
        },
        "execution": {
            "command": command,
            "secrets_redacted": True,
            "output_root": output_root,
            "log_uri": "",
            "exit_code": None,
            "failure_reason": None,
        },
        "artifacts": [],
        "events": [],
        "notes": {
            "protocol_deviations": [],
            "unverified_items": [],
            "secret_redaction_checked": True,
        },
    }
