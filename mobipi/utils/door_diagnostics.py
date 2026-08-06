"""CPU-testable instrumentation for the microwave door failure audit.

The recorder only reads simulator state after the existing reset/step calls.  It
does not sample a policy, touch a random-number generator, or issue an extra
environment step.  Simulator-derived quantities are intentionally marked as
privileged audit labels by the manifest schema.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


DIAGNOSTIC_SCHEMA_VERSION = "1.0"
DOOR_SUCCESS_THRESHOLD_NORMALIZED = 0.05
DOOR_JOINT_MIN_RAD = 0.0
DOOR_JOINT_MAX_RAD = math.pi / 2.0
DOOR_JOINT_SIGN = -1.0


# The values below are frozen in the diagnostic config before the rerun.  They
# are deliberately expressed in the trajectory's native units.
DEFAULT_STALL_CONFIG: Dict[str, float] = {
    "window_steps": 25,
    "door_progress_range_threshold": 0.005,
    "eef_displacement_threshold_m": 0.02,
    "action_norm_min": 1.0,
    "action_delta_norm_max": 0.05,
    "contact_fraction_threshold": 0.5,
    "plane_tolerance_m": 0.002,
    "plane_hysteresis_m": 0.005,
    "control_frequency_hz": 20.0,
}


STATE_FIELDS = (
    "door_joint_raw",
    "door_joint_initial_raw",
    "door_joint_success_threshold_raw",
    "door_joint_open_fraction",
    "door_closing_progress",
    "eef_world_pos",
    "door_plane_origin",
    "door_plane_normal",
    "eef_signed_distance_m",
    "eef_on_exterior_side",
    "door_plane_crossed",
    "eef_plane_projection_uv_m",
    "door_site_span_uv_m",
    "robot_door_contact_active",
    "robot_door_contact_pair_count",
    "robot_door_contact_min_distance_m",
    "robot_door_contact_position",
    "robot_door_contact_normal",
    "door_stall",
    "eef_stall",
    "action_stall",
    "contact_stall",
    "stall",
    "eef_window_displacement_m",
    "door_window_progress_range",
)

ACTION_FIELDS = (
    "action_norm",
    "action_delta_norm",
    "eef_step_displacement_m",
)


def diagnostic_field_schema() -> Dict[str, Dict[str, Any]]:
    """Return the machine-readable field contract used in each manifest."""

    privileged = "simulator_privileged_audit_label"
    return {
        "door_joint_raw": {"shape": ["T+1"], "dtype": "float64", "units": "rad", "alignment": "state", "kind": "raw", "provenance": privileged},
        "door_joint_initial_raw": {"shape": ["T+1"], "dtype": "float64", "units": "rad", "alignment": "state", "kind": "raw", "provenance": privileged},
        "door_joint_success_threshold_raw": {"shape": ["T+1"], "dtype": "float64", "units": "rad", "alignment": "state", "kind": "derived_from_task_checker", "provenance": privileged},
        "door_joint_open_fraction": {"shape": ["T+1"], "dtype": "float64", "units": "fraction", "alignment": "state", "kind": "derived", "provenance": privileged},
        "door_closing_progress": {"shape": ["T+1"], "dtype": "float64", "units": "fraction", "alignment": "state", "kind": "derived", "provenance": privileged},
        "eef_world_pos": {"shape": ["T+1", 3], "dtype": "float64", "units": "m", "alignment": "state", "kind": "raw", "provenance": "robot_state_plus_simulator_frame"},
        "door_plane_origin": {"shape": ["T+1", 3], "dtype": "float64", "units": "m", "alignment": "state", "kind": "raw_geometry", "provenance": privileged},
        "door_plane_normal": {"shape": ["T+1", 3], "dtype": "float64", "units": "unitless", "alignment": "state", "kind": "derived_geometry", "provenance": privileged},
        "eef_signed_distance_m": {"shape": ["T+1"], "dtype": "float64", "units": "m", "alignment": "state", "kind": "derived", "provenance": privileged},
        "eef_on_exterior_side": {"shape": ["T+1"], "dtype": "bool", "units": "boolean", "alignment": "state", "kind": "derived", "provenance": privileged},
        "door_plane_crossed": {"shape": ["T+1"], "dtype": "bool", "units": "event", "alignment": "state", "kind": "derived_event", "provenance": privileged},
        "eef_plane_projection_uv_m": {"shape": ["T+1", 2], "dtype": "float64", "units": "m", "alignment": "state", "kind": "derived", "provenance": privileged},
        "door_site_span_uv_m": {"shape": ["T+1", 2], "dtype": "float64", "units": "m", "alignment": "state", "kind": "raw_geometry", "provenance": privileged},
        "robot_door_contact_active": {"shape": ["T+1"], "dtype": "bool", "units": "boolean", "alignment": "state", "kind": "raw_contact", "provenance": privileged},
        "robot_door_contact_pair_count": {"shape": ["T+1"], "dtype": "int64", "units": "count", "alignment": "state", "kind": "raw_contact", "provenance": privileged},
        "robot_door_contact_min_distance_m": {"shape": ["T+1"], "dtype": "float64", "units": "m", "alignment": "state", "kind": "raw_contact", "provenance": privileged},
        "robot_door_contact_position": {"shape": ["T+1", 3], "dtype": "float64", "units": "m", "alignment": "state", "kind": "raw_contact", "provenance": privileged},
        "robot_door_contact_normal": {"shape": ["T+1", 3], "dtype": "float64", "units": "unitless", "alignment": "state", "kind": "raw_contact", "provenance": privileged},
        "door_stall": {"shape": ["T+1"], "dtype": "bool", "units": "boolean", "alignment": "state", "kind": "derived", "provenance": "audit_heuristic"},
        "eef_stall": {"shape": ["T+1"], "dtype": "bool", "units": "boolean", "alignment": "state", "kind": "derived", "provenance": "audit_heuristic"},
        "action_stall": {"shape": ["T+1"], "dtype": "bool", "units": "boolean", "alignment": "state", "kind": "derived", "provenance": "action_log"},
        "contact_stall": {"shape": ["T+1"], "dtype": "bool", "units": "boolean", "alignment": "state", "kind": "derived", "provenance": privileged},
        "stall": {"shape": ["T+1"], "dtype": "bool", "units": "boolean", "alignment": "state", "kind": "derived", "provenance": "audit_heuristic"},
        "eef_window_displacement_m": {"shape": ["T+1"], "dtype": "float64", "units": "m", "alignment": "state", "kind": "derived", "provenance": "robot_state"},
        "door_window_progress_range": {"shape": ["T+1"], "dtype": "float64", "units": "fraction", "alignment": "state", "kind": "derived", "provenance": "audit_heuristic"},
        "action_norm": {"shape": ["T"], "dtype": "float64", "units": "action_units", "alignment": "action", "kind": "raw_derived", "provenance": "policy_output"},
        "action_delta_norm": {"shape": ["T"], "dtype": "float64", "units": "action_units", "alignment": "action", "kind": "derived", "provenance": "policy_output"},
        "eef_step_displacement_m": {"shape": ["T"], "dtype": "float64", "units": "m", "alignment": "action_transition", "kind": "derived", "provenance": "robot_state"},
    }


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def raw_threshold_from_normalized(
    threshold_normalized: float = DOOR_SUCCESS_THRESHOLD_NORMALIZED,
    *,
    joint_min_rad: float = DOOR_JOINT_MIN_RAD,
    joint_max_rad: float = DOOR_JOINT_MAX_RAD,
    joint_sign: float = DOOR_JOINT_SIGN,
) -> float:
    """Convert RoboCasa's normalized open fraction to the raw hinge qpos."""

    if not 0.0 <= threshold_normalized <= 1.0:
        raise ValueError("normalized threshold must be in [0, 1]")
    if joint_max_rad <= joint_min_rad or joint_sign == 0:
        raise ValueError("invalid joint range/sign")
    return float(joint_sign * (joint_min_rad + threshold_normalized * (joint_max_rad - joint_min_rad)))


def door_open_fraction_from_raw(
    raw_value: Any,
    *,
    joint_min_rad: float = DOOR_JOINT_MIN_RAD,
    joint_max_rad: float = DOOR_JOINT_MAX_RAD,
    joint_sign: float = DOOR_JOINT_SIGN,
) -> np.ndarray:
    raw = _finite_array(raw_value, "door_joint_raw")
    hinge = joint_sign * raw
    return np.clip((hinge - joint_min_rad) / (joint_max_rad - joint_min_rad), 0.0, 1.0)


def normalize_closing_progress(
    initial_raw: float,
    current_raw: Any,
    success_threshold_raw: float,
) -> np.ndarray:
    """Return 0 at the reset state and 1 at the task success threshold."""

    initial = float(initial_raw)
    threshold = float(success_threshold_raw)
    if not math.isfinite(initial) or not math.isfinite(threshold):
        raise ValueError("door progress endpoints must be finite")
    denominator = threshold - initial
    if abs(denominator) <= 1e-12:
        raise ValueError("door progress endpoints must differ")
    current = _finite_array(current_raw, "door_joint_raw")
    return np.clip((current - initial) / denominator, 0.0, 1.0)


def plane_from_points(points: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return origin, unit axes (u/v), and unit normal from three door sites."""

    array = _finite_array(points, "door_plane_points")
    if array.shape != (3, 3):
        raise ValueError("door_plane_points must have shape (3, 3)")
    u = array[1] - array[0]
    v = array[2] - array[0]
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm <= 1e-12 or v_norm <= 1e-12:
        raise ValueError("door plane sites must be distinct")
    u = u / u_norm
    v = v / v_norm
    normal = np.cross(u, v)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-12:
        raise ValueError("door plane sites must not be collinear")
    return array[0].copy(), u, v, normal / normal_norm


def signed_distance_to_plane(point: Any, origin: Any, normal: Any) -> float:
    point_array = _finite_array(point, "point")
    origin_array = _finite_array(origin, "plane_origin")
    normal_array = _finite_array(normal, "plane_normal")
    if point_array.shape != (3,) or origin_array.shape != (3,) or normal_array.shape != (3,):
        raise ValueError("point, origin and normal must have shape (3,)")
    norm = np.linalg.norm(normal_array)
    if norm <= 1e-12:
        raise ValueError("plane normal must be nonzero")
    return float(np.dot(point_array - origin_array, normal_array / norm))


def crossing_event(
    previous_distance: float,
    current_distance: float,
    *,
    already_crossed: bool,
    hysteresis_m: float,
) -> bool:
    """Detect one exterior(+)->interior(-) crossing with hysteresis."""

    if hysteresis_m <= 0:
        raise ValueError("hysteresis_m must be positive")
    if already_crossed:
        return False
    if not math.isfinite(previous_distance) or not math.isfinite(current_distance):
        raise ValueError("plane distances must be finite")
    return bool(previous_distance >= hysteresis_m and current_distance <= -hysteresis_m)


def aggregate_robot_door_contacts(contacts: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate already-filtered physical MuJoCo contacts for one state."""

    pairs = []
    min_distance = None
    representative_position = np.zeros(3, dtype=np.float64)
    representative_normal = np.zeros(3, dtype=np.float64)
    for contact in contacts:
        distance = float(contact["distance"])
        if not math.isfinite(distance) or distance > 0.0:
            continue
        position = np.asarray(contact.get("position", np.zeros(3)), dtype=np.float64)
        normal = np.asarray(contact.get("normal", np.zeros(3)), dtype=np.float64)
        if position.shape != (3,) or normal.shape != (3,):
            raise ValueError("contact position/normal must have shape (3,)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(normal)):
            raise ValueError("contact position/normal must be finite")
        pairs.append(
            {
                "robot_geom": str(contact["robot_geom"]),
                "robot_body": str(contact.get("robot_body", "")),
                "door_geom": str(contact["door_geom"]),
                "door_body": str(contact.get("door_body", "")),
                "distance_m": distance,
                "position_m": position.tolist(),
                "normal": normal.tolist(),
            }
        )
        if min_distance is None or distance < min_distance:
            min_distance = distance
            representative_position = position
            representative_normal = normal
    return {
        "active": bool(pairs),
        "pair_count": len(pairs),
        "min_distance_m": float(min_distance if min_distance is not None else 0.0),
        "position_m": representative_position.tolist(),
        "normal": representative_normal.tolist(),
        "pairs": pairs,
    }


def compute_stall_flags(
    door_progress: Any,
    eef_world_pos: Any,
    action_norm: Any,
    action_delta_norm: Any,
    contact_active: Any,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Compute frozen trailing-window stall labels on state-aligned arrays."""

    cfg = dict(DEFAULT_STALL_CONFIG)
    if config:
        cfg.update({key: float(value) for key, value in config.items()})
    progress = _finite_array(door_progress, "door_progress")
    positions = _finite_array(eef_world_pos, "eef_world_pos")
    norms = _finite_array(action_norm, "action_norm")
    deltas = _finite_array(action_delta_norm, "action_delta_norm")
    contacts = np.asarray(contact_active, dtype=bool)
    if progress.ndim != 1 or positions.shape != (len(progress), 3):
        raise ValueError("progress/EEF arrays have incompatible shapes")
    if norms.shape != progress.shape or deltas.shape != progress.shape or contacts.shape != progress.shape:
        raise ValueError("stall inputs must be state-aligned")
    window = int(cfg["window_steps"])
    if window < 1:
        raise ValueError("window_steps must be positive")

    door_stall = np.zeros(len(progress), dtype=bool)
    eef_stall = np.zeros(len(progress), dtype=bool)
    action_stall = np.zeros(len(progress), dtype=bool)
    contact_stall = np.zeros(len(progress), dtype=bool)
    stall = np.zeros(len(progress), dtype=bool)
    eef_displacement = np.zeros(len(progress), dtype=np.float64)
    progress_range = np.zeros(len(progress), dtype=np.float64)
    for index in range(window, len(progress)):
        start = index - window
        progress_range[index] = float(np.ptp(progress[start : index + 1]))
        eef_displacement[index] = float(np.linalg.norm(positions[index] - positions[start]))
        door_stall[index] = progress_range[index] <= cfg["door_progress_range_threshold"]
        eef_stall[index] = eef_displacement[index] <= cfg["eef_displacement_threshold_m"]
        mean_norm = float(np.mean(norms[start : index + 1]))
        mean_delta = float(np.mean(deltas[start : index + 1]))
        action_stall[index] = (
            mean_norm >= cfg["action_norm_min"]
            and mean_delta <= cfg["action_delta_norm_max"]
        )
        contact_fraction = float(np.mean(contacts[start : index + 1]))
        contact_stall[index] = eef_stall[index] and contact_fraction >= cfg["contact_fraction_threshold"]
        stall[index] = door_stall[index] and eef_stall[index] and action_stall[index]
    return {
        "door_stall": door_stall,
        "eef_stall": eef_stall,
        "action_stall": action_stall,
        "contact_stall": contact_stall,
        "stall": stall,
        "eef_window_displacement_m": eef_displacement,
        "door_window_progress_range": progress_range,
    }


def _first_true(values: np.ndarray) -> int:
    indices = np.flatnonzero(values)
    return int(indices[0]) if len(indices) else -1


def validate_diagnostic_trajectory(
    trajectory: Mapping[str, Any],
    episode_steps: int,
    *,
    summary: Optional[Mapping[str, Any]] = None,
    atol: float = 1e-5,
) -> None:
    """Validate the diagnostic NPZ payload without importing simulator code."""

    if episode_steps < 0:
        raise ValueError("episode_steps must be nonnegative")
    state_length = episode_steps + 1
    action_length = episode_steps
    arrays: Dict[str, np.ndarray] = {}
    for name in STATE_FIELDS + ACTION_FIELDS:
        key = f"diagnostic__{name}"
        if key not in trajectory:
            raise ValueError(f"diagnostic field missing: {key}")
        arrays[name] = np.asarray(trajectory[key])
    for name in STATE_FIELDS:
        array = arrays[name]
        expected = state_length
        if array.shape[0] != expected:
            raise ValueError(f"{name} must have first dimension T+1={expected}, got {array.shape}")
        if name in {"eef_world_pos", "door_plane_origin", "door_plane_normal", "robot_door_contact_position", "robot_door_contact_normal"}:
            if array.shape != (expected, 3):
                raise ValueError(f"{name} must have shape {(expected, 3)}, got {array.shape}")
        elif name in {"eef_plane_projection_uv_m", "door_site_span_uv_m"}:
            if array.shape != (expected, 2):
                raise ValueError(f"{name} must have shape {(expected, 2)}, got {array.shape}")
        elif array.shape != (expected,):
            raise ValueError(f"{name} must have shape {(expected,)}, got {array.shape}")
    for name in ACTION_FIELDS:
        if arrays[name].shape != (action_length,):
            raise ValueError(f"{name} must have shape {(action_length,)}")

    for name, array in arrays.items():
        if array.dtype.kind in "fiu" and not np.all(np.isfinite(array.astype(np.float64))):
            raise ValueError(f"{name} contains NaN or Inf")
    raw = arrays["door_joint_raw"].astype(np.float64)
    initial = arrays["door_joint_initial_raw"].astype(np.float64)
    threshold = arrays["door_joint_success_threshold_raw"].astype(np.float64)
    expected_progress = normalize_closing_progress(initial[0], raw, threshold[0])
    if not np.allclose(arrays["door_closing_progress"], expected_progress, atol=atol, rtol=0):
        raise ValueError("door_closing_progress is inconsistent with raw joint values")
    expected_open = door_open_fraction_from_raw(raw)
    if not np.allclose(arrays["door_joint_open_fraction"], expected_open, atol=atol, rtol=0):
        raise ValueError("door_joint_open_fraction is inconsistent with raw joint values")
    normal = arrays["door_plane_normal"].astype(np.float64)
    if not np.allclose(np.linalg.norm(normal, axis=1), 1.0, atol=atol, rtol=0):
        raise ValueError("door_plane_normal must be unit length")
    expected_distance = np.sum(
        (arrays["eef_world_pos"] - arrays["door_plane_origin"]) * normal, axis=1
    )
    if not np.allclose(arrays["eef_signed_distance_m"], expected_distance, atol=atol, rtol=0):
        raise ValueError("eef_signed_distance_m is inconsistent with plane geometry")
    crossing_indices = np.flatnonzero(arrays["door_plane_crossed"])
    if len(crossing_indices) > 1:
        raise ValueError("door_plane_crossed may contain at most one first-crossing event")
    if len(crossing_indices) and crossing_indices[0] == 0:
        raise ValueError("door_plane_crossed cannot fire at the reset state")
    if np.any(arrays["robot_door_contact_pair_count"] < 0):
        raise ValueError("contact pair counts must be nonnegative")
    if not np.array_equal(
        arrays["robot_door_contact_active"], arrays["robot_door_contact_pair_count"] > 0
    ):
        raise ValueError("contact active flag disagrees with pair count")
    if summary is not None:
        for name in ("first_crossing_step", "first_contact_step", "last_contact_step", "first_stall_step"):
            value = int(summary.get(name, -1))
            if value < -1 or value >= state_length:
                raise ValueError(f"{name} is outside the valid step range")
        contact_indices = np.flatnonzero(arrays["robot_door_contact_active"])
        first_contact = int(contact_indices[0]) if len(contact_indices) else -1
        last_contact = int(contact_indices[-1]) if len(contact_indices) else -1
        if int(summary.get("first_contact_step", -1)) != first_contact:
            raise ValueError("first_contact_step disagrees with contact array")
        if int(summary.get("last_contact_step", -1)) != last_contact:
            raise ValueError("last_contact_step disagrees with contact array")
        if int(summary.get("contact_duration_steps", len(contact_indices))) != len(contact_indices):
            raise ValueError("contact_duration_steps disagrees with contact array")
        first_crossing = int(crossing_indices[0]) if len(crossing_indices) else -1
        if int(summary.get("first_crossing_step", -1)) != first_crossing:
            raise ValueError("first_crossing_step disagrees with crossing array")
    stall = compute_stall_flags(
        arrays["door_closing_progress"],
        arrays["eef_world_pos"],
        np.concatenate(([0.0], arrays["action_norm"].astype(np.float64))),
        np.concatenate(([0.0], arrays["action_delta_norm"].astype(np.float64))),
        arrays["robot_door_contact_active"],
    )
    for name in ("door_stall", "eef_stall", "action_stall", "contact_stall", "stall"):
        if not np.array_equal(arrays[name], stall[name]):
            raise ValueError(f"{name} is inconsistent with frozen stall config")
    if summary is not None and int(summary.get("first_stall_step", -1)) != _first_true(stall["stall"]):
        raise ValueError("first_stall_step disagrees with stall array")


def validate_case_list(
    payload: Mapping[str, Any],
    *,
    candidate_ids: Iterable[str],
    candidate_config_sha256: str,
) -> Dict[str, Any]:
    """Validate the exact eight-case, failure/control diagnostic contract."""

    if payload.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("diagnostic case-list schema_version must be '1.0'")
    if int(payload.get("case_count", -1)) != 8:
        raise ValueError("diagnostic case-list case_count must be 8")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 8:
        raise ValueError("diagnostic case list must contain exactly 8 cases")
    if str(payload.get("candidate_config_sha256", "")).lower() != candidate_config_sha256.lower():
        raise ValueError("diagnostic case list candidate config checksum mismatch")
    known = set(candidate_ids)
    seen = set()
    pair_groups: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    failures = controls = 0
    for case in cases:
        required = ("case_id", "environment_seed", "evaluation_seed", "candidate_id", "expected_historical_outcome", "matched_pair_id", "candidate_config_sha256")
        missing = [key for key in required if key not in case]
        if missing:
            raise ValueError(f"diagnostic case missing fields: {missing}")
        case_id = str(case["case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate diagnostic case_id: {case_id}")
        seen.add(case_id)
        if str(case["candidate_id"]) not in known:
            raise ValueError(f"unknown diagnostic candidate: {case['candidate_id']}")
        if str(case["candidate_config_sha256"]).lower() != candidate_config_sha256.lower():
            raise ValueError(f"case {case_id} candidate checksum mismatch")
        outcome = str(case["expected_historical_outcome"]).lower()
        if outcome not in {"failure", "control_success"}:
            raise ValueError(f"case {case_id} has invalid expected outcome {outcome}")
        failures += outcome == "failure"
        controls += outcome == "control_success"
        if not str(case["matched_pair_id"]):
            raise ValueError(f"case {case_id} must have matched_pair_id")
        pair_groups[str(case["matched_pair_id"])].append(case)
    if failures != 4 or controls != 4:
        raise ValueError(f"diagnostic cases must contain 4 failures and 4 controls, got {failures}/{controls}")
    if len(pair_groups) != 4 or any(len(group) != 2 for group in pair_groups.values()):
        raise ValueError("diagnostic cases must form four two-case matched pairs")
    for pair_id, group in pair_groups.items():
        outcomes = {str(item["expected_historical_outcome"]).lower() for item in group}
        if outcomes != {"failure", "control_success"}:
            raise ValueError(f"matched pair {pair_id} must contain one failure and one control")
        if len({int(item["environment_seed"]) for item in group}) != 1:
            raise ValueError(f"matched pair {pair_id} must share environment_seed")
        if len({int(item["evaluation_seed"]) for item in group}) != 1:
            raise ValueError(f"matched pair {pair_id} must share evaluation_seed")
    return {"schema_version": DIAGNOSTIC_SCHEMA_VERSION, **dict(payload), "cases": cases}


def _unwrap_env(env: Any) -> Any:
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "unwrapped") and current.unwrapped is not current:
            current = current.unwrapped
            continue
        if hasattr(current, "env") and current.env is not current:
            current = current.env
            continue
        break
    return current


class DoorDiagnosticRecorder:
    """Read-only per-state recorder for one already-reset microwave episode."""

    def __init__(self, env: Any, *, stall_config: Optional[Mapping[str, Any]] = None):
        self.env = env
        self.raw_env = _unwrap_env(env)
        self.sim = self.raw_env.sim
        self.fixture = self.raw_env.door_fxtr
        self.fixture_name = str(getattr(self.fixture, "name", ""))
        if not self.fixture_name:
            raise ValueError("door fixture has no stable name")
        self.door_joint_name = f"{self.fixture_name}_microjoint"
        qpos_address = self.sim.model.get_joint_qpos_addr(self.door_joint_name)
        if isinstance(qpos_address, (tuple, list, np.ndarray)):
            if len(qpos_address) != 1:
                raise ValueError("microwave microjoint must have one qpos address")
            qpos_address = qpos_address[0]
        self.door_joint_id = int(qpos_address)
        prefix = str(getattr(self.fixture, "naming_prefix", f"{self.fixture_name}_"))
        self.site_names = tuple(prefix + suffix for suffix in ("door_p1", "door_p2", "door_p3"))
        self.site_ids = tuple(int(self.sim.model.site_name2id(name)) for name in self.site_names)
        self.door_body_name = f"{self.fixture_name}_door"
        self.door_body_id = int(self.sim.model.body_name2id(self.door_body_name))
        robot = self.raw_env.robots[0]
        self.eef_site_id = int(robot.eef_site_id["right"])
        self.eef_site_name = str(self.sim.model.site_id2name(self.eef_site_id))
        self.stall_config = dict(DEFAULT_STALL_CONFIG)
        if stall_config:
            self.stall_config.update({key: float(value) for key, value in stall_config.items()})
        self.success_threshold_normalized = DOOR_SUCCESS_THRESHOLD_NORMALIZED
        self.success_threshold_raw = raw_threshold_from_normalized(self.success_threshold_normalized)
        self._states = []
        self._step_pairs = []
        self._previous_distance = None
        self._crossed = False
        self._normal_orientation_sign = None
        self._previous_action = None
        self.record_state()

    def _contact_rows(self) -> list[Dict[str, Any]]:
        rows = []
        model = self.sim.model
        for contact in self.sim.data.contact:
            distance = float(contact.dist)
            if not math.isfinite(distance) or distance > 0.0:
                continue
            geom1 = model.geom_id2name(int(contact.geom1))
            geom2 = model.geom_id2name(int(contact.geom2))
            if not geom1 or not geom2:
                continue
            body1_id = int(model.geom_bodyid[int(contact.geom1)])
            body2_id = int(model.geom_bodyid[int(contact.geom2)])
            body1 = model.body_id2name(body1_id) or ""
            body2 = model.body_id2name(body2_id) or ""
            robot1 = str(geom1).startswith(("robot0", "mobilebase0")) or str(body1).startswith(("robot0", "mobilebase0"))
            robot2 = str(geom2).startswith(("robot0", "mobilebase0")) or str(body2).startswith(("robot0", "mobilebase0"))
            door1 = body1_id == self.door_body_id
            door2 = body2_id == self.door_body_id
            if robot1 == robot2 or door1 == door2:
                continue
            if robot1 and door2:
                robot_geom, robot_body, door_geom, door_body = geom1, body1, geom2, body2
            elif robot2 and door1:
                robot_geom, robot_body, door_geom, door_body = geom2, body2, geom1, body1
            else:
                continue
            frame = np.asarray(getattr(contact, "frame", np.zeros(9)), dtype=np.float64).reshape(-1)
            normal = frame[:3] if frame.size >= 3 else np.zeros(3)
            rows.append(
                {
                    "robot_geom": str(robot_geom),
                    "robot_body": str(robot_body),
                    "door_geom": str(door_geom),
                    "door_body": str(door_body),
                    "distance": distance,
                    "position": np.asarray(contact.pos, dtype=np.float64).tolist(),
                    "normal": normal.tolist(),
                }
            )
        return rows

    def _read_geometry(self) -> Dict[str, Any]:
        points = np.stack([np.asarray(self.sim.data.site_xpos[index], dtype=np.float64) for index in self.site_ids])
        origin, u, v, raw_normal = plane_from_points(points)
        eef = np.asarray(self.sim.data.site_xpos[self.eef_site_id], dtype=np.float64).copy()
        if self._normal_orientation_sign is None:
            initial_distance = signed_distance_to_plane(eef, origin, raw_normal)
            self._normal_orientation_sign = 1.0 if initial_distance >= 0.0 else -1.0
        normal = raw_normal * float(self._normal_orientation_sign)
        distance = signed_distance_to_plane(eef, origin, normal)
        projection = np.array([np.dot(eef - origin, u), np.dot(eef - origin, v)], dtype=np.float64)
        return {
            "eef_world_pos": eef,
            "door_plane_origin": origin,
            "door_plane_normal": normal,
            "eef_signed_distance_m": distance,
            "eef_on_exterior_side": bool(distance >= self.stall_config["plane_tolerance_m"]),
            "eef_plane_projection_uv_m": projection,
            "door_site_span_uv_m": np.array([np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[0])], dtype=np.float64),
        }

    def record_state(self, action: Optional[Any] = None) -> None:
        geometry = self._read_geometry()
        raw_joint = float(self.sim.data.qpos[self.door_joint_id])
        if self._states:
            event = crossing_event(
                self._previous_distance,
                geometry["eef_signed_distance_m"],
                already_crossed=self._crossed,
                hysteresis_m=self.stall_config["plane_hysteresis_m"],
            )
            if event:
                self._crossed = True
        else:
            event = False
        contacts = aggregate_robot_door_contacts(self._contact_rows())
        action_array = None if action is None else np.asarray(action, dtype=np.float64).reshape(-1)
        action_norm = 0.0 if action_array is None else float(np.linalg.norm(action_array))
        if action_array is None or self._previous_action is None:
            action_delta_norm = 0.0
        else:
            action_delta_norm = float(np.linalg.norm(action_array - self._previous_action))
        if self._states:
            eef_step_displacement = float(np.linalg.norm(geometry["eef_world_pos"] - self._states[-1]["eef_world_pos"]))
        else:
            eef_step_displacement = 0.0
        state = {
            "door_joint_raw": raw_joint,
            "door_joint_initial_raw": raw_joint if not self._states else self._states[0]["door_joint_initial_raw"],
            "door_joint_success_threshold_raw": self.success_threshold_raw,
            "door_joint_open_fraction": float(door_open_fraction_from_raw(raw_joint)),
            "door_closing_progress": 0.0,  # filled after initial endpoint is known
            **geometry,
            "door_plane_crossed": bool(event),
            "robot_door_contact_active": contacts["active"],
            "robot_door_contact_pair_count": contacts["pair_count"],
            "robot_door_contact_min_distance_m": contacts["min_distance_m"],
            "robot_door_contact_position": np.asarray(contacts["position_m"], dtype=np.float64),
            "robot_door_contact_normal": np.asarray(contacts["normal"], dtype=np.float64),
            "action_norm": action_norm,
            "action_delta_norm": action_delta_norm,
            "eef_step_displacement_m": eef_step_displacement,
        }
        self._states.append(state)
        self._step_pairs.append(contacts["pairs"])
        self._previous_distance = geometry["eef_signed_distance_m"]
        self._crossed = self._crossed or bool(event)
        self._previous_action = None if action_array is None else action_array.copy()
        progress = normalize_closing_progress(
            self._states[0]["door_joint_raw"],
            np.asarray([item["door_joint_raw"] for item in self._states]),
            self.success_threshold_raw,
        )
        for item, value in zip(self._states, progress):
            item["door_closing_progress"] = float(value)

    def finalize(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
        if not self._states:
            raise ValueError("recorder has no states")
        state_progress = np.asarray([item["door_closing_progress"] for item in self._states], dtype=np.float64)
        state_positions = np.stack([item["eef_world_pos"] for item in self._states])
        state_norms = np.asarray([item["action_norm"] for item in self._states], dtype=np.float64)
        state_deltas = np.asarray([item["action_delta_norm"] for item in self._states], dtype=np.float64)
        state_contacts = np.asarray([item["robot_door_contact_active"] for item in self._states], dtype=bool)
        stalls = compute_stall_flags(
            state_progress,
            state_positions,
            state_norms,
            state_deltas,
            state_contacts,
            self.stall_config,
        )
        trajectory: Dict[str, np.ndarray] = {}
        for name in STATE_FIELDS:
            if name in stalls:
                value = stalls[name]
            elif name in {"door_joint_raw", "door_joint_initial_raw", "door_joint_success_threshold_raw", "door_joint_open_fraction", "door_closing_progress", "eef_signed_distance_m", "robot_door_contact_min_distance_m"}:
                value = np.asarray([item[name] for item in self._states], dtype=np.float64)
            elif name in {"eef_world_pos", "door_plane_origin", "door_plane_normal", "eef_plane_projection_uv_m", "door_site_span_uv_m", "robot_door_contact_position", "robot_door_contact_normal"}:
                value = np.stack([item[name] for item in self._states]).astype(np.float64)
            elif name in {"eef_on_exterior_side", "door_plane_crossed", "robot_door_contact_active"}:
                value = np.asarray([item[name] for item in self._states], dtype=bool)
            elif name == "robot_door_contact_pair_count":
                value = np.asarray([item[name] for item in self._states], dtype=np.int64)
            else:
                raise AssertionError(f"unhandled diagnostic field {name}")
            trajectory[f"diagnostic__{name}"] = value
        trajectory["diagnostic__action_norm"] = state_norms[1:].astype(np.float64)
        trajectory["diagnostic__action_delta_norm"] = state_deltas[1:].astype(np.float64)
        trajectory["diagnostic__eef_step_displacement_m"] = np.asarray(
            [item["eef_step_displacement_m"] for item in self._states[1:]], dtype=np.float64
        )
        crossing = np.flatnonzero(trajectory["diagnostic__door_plane_crossed"])
        contact = np.flatnonzero(trajectory["diagnostic__robot_door_contact_active"])
        summary: Dict[str, Any] = {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "episode_steps": len(self._states) - 1,
            "door_fixture_name": self.fixture_name,
            "door_fixture_class": type(self.fixture).__name__,
            "door_joint_name": self.door_joint_name,
            "door_body_name": self.door_body_name,
            "door_site_names": list(self.site_names),
            "eef_site_name": self.eef_site_name,
            "plane_normal_orientation_sign": float(self._normal_orientation_sign),
            "plane_crossing_semantics": "infinite dynamic plane through door_p1/p2/p3; positive exterior to negative interior",
            "plane_crossing_is_edge_approximation": True,
            "first_crossing_step": int(crossing[0]) if len(crossing) else -1,
            "first_contact_step": int(contact[0]) if len(contact) else -1,
            "last_contact_step": int(contact[-1]) if len(contact) else -1,
            "contact_duration_steps": int(len(contact)),
            "contact_duration_s": float(len(contact) / self.stall_config["control_frequency_hz"]),
            "first_door_stall_step": _first_true(trajectory["diagnostic__door_stall"]),
            "first_eef_stall_step": _first_true(trajectory["diagnostic__eef_stall"]),
            "first_action_stall_step": _first_true(trajectory["diagnostic__action_stall"]),
            "first_contact_stall_step": _first_true(trajectory["diagnostic__contact_stall"]),
            "first_stall_step": _first_true(trajectory["diagnostic__stall"]),
            "stall_config": dict(self.stall_config),
            "field_schema": diagnostic_field_schema(),
            "contact_pairs": self._pair_summary(),
        }
        validate_diagnostic_trajectory(trajectory, len(self._states) - 1, summary=summary)
        contacts_sidecar = {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "door_fixture_name": self.fixture_name,
            "door_body_name": self.door_body_name,
            "step_alignment": "T+1 state samples; state 0 is reset before the first action",
            "step_pairs": self._step_pairs,
        }
        return trajectory, summary, contacts_sidecar

    def _pair_summary(self) -> list[Dict[str, Any]]:
        occurrences: Dict[Tuple[str, str, str, str], list[int]] = defaultdict(list)
        for step, pairs in enumerate(self._step_pairs):
            for pair in pairs:
                key = (pair["robot_geom"], pair["robot_body"], pair["door_geom"], pair["door_body"])
                occurrences[key].append(step)
        result = []
        for (robot_geom, robot_body, door_geom, door_body), steps in sorted(occurrences.items()):
            result.append(
                {
                    "robot_geom": robot_geom,
                    "robot_body": robot_body,
                    "door_geom": door_geom,
                    "door_body": door_body,
                    "first_step": int(min(steps)),
                    "last_step": int(max(steps)),
                    "active_steps": int(len(steps)),
                }
            )
        return result


def load_case_list(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as stream:
        payload = json.load(stream)
    return payload
