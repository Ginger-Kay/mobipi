"""Frozen SCENE-004 compiler, selector, and minimal OBC primitives.

All source/candidate compiler inputs are pre-outcome.  Simulator adapters may
populate the typed records below, but this module deliberately cannot query a
task outcome or advance an environment.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


TASKS = ("CloseSingleDoor", "CloseDrawer")
CELLS = (1, 4, 7, 8, 9)
ROUTE_ORDER = {"E": 0, "D": 1, "A": 2, "X": 3}
OUTCOME_DENYLIST = frozenset(
    {
        "success",
        "task_success",
        "progress",
        "progress_after",
        "task_progress_after",
        "irreversible_failure",
        "collision_after",
        "contact_after",
        "outcome",
        "terminal_reason",
        "reward",
        "done",
        "route_cost",
    }
)

CANDIDATE_FEATURE_FIELDS = (
    "route_E",
    "route_D",
    "route_A",
    "task_CloseDrawer",
    "task_CloseSingleDoor",
    "stage_precontact",
    "hard_valid",
    "minimum_continuous_clearance_m",
    "minimum_manipulability",
    "minimum_joint_margin_rad",
    "minimum_policy_view_compatibility",
    "total_planned_base_path_m",
    "planned_time_normalized",
    "planned_base_net_m",
    "planned_eef_path_m",
    "maximum_eef_tracking_error_m",
    "minimum_velocity_margin",
    "minimum_acceleration_margin",
    "solver_residual",
    "slot_index_normalized",
    "simulator_oracle_pre_outcome",
)


def canonical_hash(value: Any) -> str:
    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if hasattr(item, "__dataclass_fields__"):
            return asdict(item)
        if isinstance(item, Mapping):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(val) for val in item]
        return item

    payload = json.dumps(convert(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reject_outcome_fields(value: Any, *, path: str = "root") -> None:
    """Fail closed on privileged/outcome fields at any nesting depth."""

    if isinstance(value, Mapping):
        leaked = sorted(str(key) for key in value if str(key).lower() in OUTCOME_DENYLIST)
        if leaked:
            raise ValueError(f"outcome/privileged fields forbidden at {path}: {leaked}")
        for key, child in value.items():
            reject_outcome_fields(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            reject_outcome_fields(child, path=f"{path}[{index}]")


@dataclass(frozen=True)
class FixtureFunctionalRecord:
    task: str
    fixture_name: str
    fixture_class: str
    target_binding: bool
    joint_names: tuple[str, ...]
    joint_types: tuple[str, ...]
    joint_ranges: tuple[tuple[float, float], ...]
    joint_qpos: tuple[float, ...]
    handle_position_world: tuple[float, float, float] | None
    task_checker_binding: bool


def functional_fixture_predicates(record: FixtureFunctionalRecord) -> dict[str, bool]:
    if record.task not in TASKS:
        raise ValueError(f"unsupported task: {record.task}")
    finite_ranges = bool(record.joint_ranges) and all(
        len(pair) == 2 and np.all(np.isfinite(pair)) and pair[1] > pair[0]
        for pair in record.joint_ranges
    )
    qpos_readable = len(record.joint_qpos) == len(record.joint_names) and bool(record.joint_qpos) and bool(
        np.all(np.isfinite(record.joint_qpos))
    )
    handle_readable = record.handle_position_world is not None and bool(
        np.all(np.isfinite(record.handle_position_world))
    )
    if record.task == "CloseSingleDoor":
        joint_function = len(record.joint_names) == 1 and tuple(record.joint_types) == ("hinge",)
    else:
        joint_function = bool(record.joint_names) and all(kind == "slide" for kind in record.joint_types)
    return {
        "task_target_binding": bool(record.target_binding),
        "joint_function": bool(joint_function),
        "joint_range_readable": bool(finite_ranges),
        "joint_qpos_readable": bool(qpos_readable),
        "handle_geometry_readable": bool(handle_readable),
        "task_checker_binding": bool(record.task_checker_binding),
    }


def validate_functional_fixture(record: FixtureFunctionalRecord) -> dict[str, Any]:
    predicates = functional_fixture_predicates(record)
    reasons = [name for name, passed in predicates.items() if not passed]
    return {
        "passed": not reasons,
        "predicates": predicates,
        "failure_reasons": reasons,
        "fixture_subtype": record.fixture_class.rsplit(".", 1)[-1],
        "class_name_used_as_acceptance_predicate": False,
    }


@dataclass(frozen=True)
class LatticePose:
    pose_id: str
    x_m: float
    y_m: float
    yaw_rad: float


def fixture_anchored_lattice(
    fixture_xy: Sequence[float], nominal_base_xy: Sequence[float], nominal_yaw_rad: float
) -> tuple[LatticePose, ...]:
    """Instantiate the fixed 3x3x3 source lattice around task placement.

    The translation axes are fixture radial/lateral axes, not world-axis
    metadata offsets.  Every returned pose is a concrete search candidate.
    """

    fixture = np.asarray(fixture_xy, dtype=float)
    nominal = np.asarray(nominal_base_xy, dtype=float)
    if fixture.shape != (2,) or nominal.shape != (2,) or not np.all(np.isfinite([*fixture, *nominal, nominal_yaw_rad])):
        raise ValueError("lattice anchor must be finite planar geometry")
    radial = nominal - fixture
    norm = float(np.linalg.norm(radial))
    if norm < 1e-6:
        raise ValueError("fixture and nominal base centers coincide")
    radial /= norm
    lateral = np.array([-radial[1], radial[0]])
    offsets = (-0.40, 0.0, 0.40)
    yaw_offsets = (-0.35, 0.0, 0.35)
    result = []
    for radial_offset in offsets:
        for lateral_offset in offsets:
            for yaw_offset in yaw_offsets:
                xy = nominal + radial_offset * radial + lateral_offset * lateral
                index = len(result)
                result.append(LatticePose(f"p{index:02d}", float(xy[0]), float(xy[1]), float(nominal_yaw_rad + yaw_offset)))
    return tuple(result)


def search_lattice(
    poses: Sequence[LatticePose], evaluator: Callable[[LatticePose], Mapping[str, float | bool]]
) -> tuple[LatticePose, tuple[dict[str, Any], ...]]:
    """Evaluate every lattice pose and deterministically select the best."""

    if len(poses) != 27:
        raise ValueError("SCENE-004 search requires the actual 27-pose lattice")
    rows = []
    for pose in poses:
        metric = dict(evaluator(pose))
        required = ("hard_valid", "visibility", "reachability", "joint_margin", "intent_error", "planned_path")
        if any(name not in metric for name in required):
            raise ValueError(f"lattice evaluator omitted fields for {pose.pose_id}")
        reject_outcome_fields(metric)
        hard_valid = bool(metric["hard_valid"])
        score = (
            0 if hard_valid else 1,
            -min(float(metric["visibility"]), float(metric["reachability"]), float(metric["joint_margin"])),
            float(metric["intent_error"]),
            float(metric["planned_path"]),
            pose.pose_id,
        )
        rows.append({"pose": asdict(pose), "metrics": metric, "rank_key": list(score)})
    rows.sort(key=lambda row: tuple(row["rank_key"]))
    selected_id = rows[0]["pose"]["pose_id"]
    selected = next(pose for pose in poses if pose.pose_id == selected_id)
    return selected, tuple(rows)


@dataclass(frozen=True)
class DockCandidate:
    candidate_id: str
    start_xy: tuple[float, float]
    dock_xy: tuple[float, float]
    planned_path_m: float


def dock_candidates(
    fixture_xy: Sequence[float], fixture_size_xy: Sequence[float], source_xy: Sequence[float],
    inflated_base_radius_m: float,
) -> tuple[DockCandidate, ...]:
    fixture = np.asarray(fixture_xy, float)
    size = np.asarray(fixture_size_xy, float)
    source = np.asarray(source_xy, float)
    if any(array.shape != (2,) for array in (fixture, size, source)) or inflated_base_radius_m <= 0:
        raise ValueError("invalid D candidate geometry")
    outward = source - fixture
    norm = float(np.linalg.norm(outward))
    if norm < 1e-6:
        raise ValueError("D source coincides with fixture")
    outward /= norm
    result = []
    for index, degrees in enumerate((0.0, 15.0, -15.0, 30.0, -30.0), start=1):
        angle = math.radians(degrees)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        direction = rotation @ outward
        half_extent = 0.5 * float(np.sum(np.abs(direction) * size))
        dock = fixture + direction * (half_extent + inflated_base_radius_m + 0.02)
        result.append(DockCandidate(f"d{index}", tuple(source), tuple(dock), float(np.linalg.norm(source - dock))))
    return tuple(result)


@dataclass(frozen=True)
class AssistCandidate:
    candidate_id: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    planned_net_m: float
    chunks: int
    ee_tangent_xy: tuple[float, float]
    fixture_axis_xy: tuple[float, float]


def ee_intent_tangent(start_pose_world: np.ndarray, end_pose_world: np.ndarray) -> np.ndarray:
    start = np.asarray(start_pose_world, float)
    end = np.asarray(end_pose_world, float)
    if start.shape != (4, 4) or end.shape != (4, 4):
        raise ValueError("EE intent requires explicit T_WE_start and T_WE_end")
    tangent = end[:2, 3] - start[:2, 3]
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        raise ValueError("hard_invalid_no_assist_direction")
    return tangent / norm


def assist_candidates(
    start_xy: Sequence[float], start_ee_pose_world: np.ndarray, end_ee_pose_world: np.ndarray,
    fixture_axis_xy: Sequence[float],
) -> tuple[AssistCandidate, ...]:
    start = np.asarray(start_xy, float)
    tangent = ee_intent_tangent(start_ee_pose_world, end_ee_pose_world)
    fixture_axis = np.asarray(fixture_axis_xy, float)
    axis_norm = float(np.linalg.norm(fixture_axis))
    if start.shape != (2,) or fixture_axis.shape != (2,) or axis_norm < 1e-6:
        raise ValueError("invalid A source/fixture axis")
    fixture_axis /= axis_norm
    # Preserve EE intent direction while using actual articulated geometry to
    # orient the lateral variants consistently.
    if float(np.dot(tangent, fixture_axis)) < 0.0:
        fixture_axis = -fixture_axis
    lateral = np.array([-tangent[1], tangent[0]])
    specs = (
        ("a1", 0.30, 0.00),
        ("a2", 0.40, 0.00),
        ("a3", 0.45, 0.00),
        ("a4", 0.40, +0.25),
        ("a5", 0.40, -0.25),
    )
    result = []
    for candidate_id, distance, lateral_gain in specs:
        direction = tangent + lateral_gain * lateral + 0.05 * fixture_axis
        direction /= np.linalg.norm(direction)
        end = start + distance * direction
        result.append(
            AssistCandidate(candidate_id, tuple(start), tuple(end), distance, 4, tuple(tangent), tuple(fixture_axis))
        )
    return tuple(result)


def sampled_segment(start_xy: Sequence[float], end_xy: Sequence[float], spacing_m: float = 0.02) -> np.ndarray:
    if spacing_m <= 0 or spacing_m > 0.02:
        raise ValueError("corridor spacing must be in (0, 0.02]")
    start, end = np.asarray(start_xy, float), np.asarray(end_xy, float)
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(math.ceil(distance / spacing_m)) + 1)
    return np.linspace(start, end, count)


def signed_corridor_clearance(
    points_xy: Iterable[Sequence[float]], obstacles: Sequence[Any], floor: Any,
    *, base_radius_m: float, inflation_m: float = 0.05,
) -> dict[str, Any]:
    """Apply footprint+5 cm once; non-negative signed clearance passes."""

    from shapely.geometry import Point

    if base_radius_m <= 0 or not np.isclose(inflation_m, 0.05):
        raise ValueError("SCENE-004 requires positive footprint and exactly 0.05m inflation")
    radius = base_radius_m + inflation_m
    safe_floor = floor.buffer(-radius)
    inflated_obstacles = [(name, polygon.buffer(radius)) for name, polygon in obstacles]
    values, nearest = [], []
    for xy in points_xy:
        point = Point(np.asarray(xy, float))
        floor_value = point.distance(safe_floor.boundary) if safe_floor.covers(point) else -point.distance(safe_floor)
        obstacle_rows = [(name, point.distance(polygon) if not polygon.covers(point) else -point.distance(polygon.boundary))
                         for name, polygon in inflated_obstacles]
        if obstacle_rows:
            obstacle_name, obstacle_value = min(obstacle_rows, key=lambda row: (row[1], row[0]))
        else:
            obstacle_name, obstacle_value = "none", float("inf")
        if obstacle_value <= floor_value:
            values.append(float(obstacle_value)); nearest.append(obstacle_name)
        else:
            values.append(float(floor_value)); nearest.append("floor_boundary")
    if not values:
        raise ValueError("corridor must contain samples")
    minimum_index = int(np.argmin(values))
    return {
        "passed": bool(min(values) >= 0.0),
        "min_signed_clearance_m": float(min(values)),
        "nearest_obstacle": nearest[minimum_index],
        "signed_clearance_m": values,
        "base_radius_m": float(base_radius_m),
        "inflation_m": 0.05,
        "acceptance_threshold_m": 0.0,
    }


@dataclass(frozen=True)
class CameraGridPose:
    camera_id: str
    center_offset_xy: tuple[float, float]
    height_m: float
    fov_deg: float


def camera_grid() -> tuple[CameraGridPose, ...]:
    offsets = ((0.0, 0.0), (0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25))
    rows = []
    for height in (3.2, 3.6, 4.0):
        for fov in (48.0, 55.0, 62.0):
            for offset in offsets:
                rows.append(CameraGridPose(f"h{height:.1f}-f{int(fov)}-x{offset[0]:+.2f}-y{offset[1]:+.2f}", offset, height, fov))
    return tuple(rows)


def select_cell_camera(
    cell_key: str, evaluations: Mapping[str, Sequence[Mapping[str, float | bool]]]
) -> dict[str, Any]:
    candidates = camera_grid()
    rows = []
    for candidate in candidates:
        per_seed = list(evaluations.get(candidate.camera_id, ()))
        if not per_seed:
            raise ValueError(f"camera {candidate.camera_id} lacks bounded-envelope evaluations")
        passed = all(bool(row["passed"]) for row in per_seed)
        min_border = min(float(row["min_border_fraction"]) for row in per_seed)
        base_pixels = min(float(row["base_projected_diameter_px"]) for row in per_seed)
        rank_key = (0 if passed else 1, -min_border, -base_pixels, candidate.fov_deg, candidate.height_m, candidate.camera_id)
        rows.append({"pose": asdict(candidate), "passed": passed, "minimum_border_fraction": min_border,
                     "minimum_base_projected_diameter_px": base_pixels, "rank_key": list(rank_key)})
    rows.sort(key=lambda row: tuple(row["rank_key"]))
    selected = rows[0]
    payload = {"cell_key": cell_key, "selected": selected, "grid_size": len(rows), "evaluations": rows}
    payload["camera_hash"] = canonical_hash({"cell_key": cell_key, "selected": selected["pose"]})
    return payload


def independent_reason_vector(predicates: Mapping[str, bool]) -> dict[str, Any]:
    ordered = {str(name): bool(predicates[name]) for name in sorted(predicates)}
    return {"predicates": ordered, "failure_reasons": [name for name, passed in ordered.items() if not passed],
            "passed": all(ordered.values())}


def a0_command(nominal_action: Sequence[float]) -> np.ndarray:
    action = np.asarray(nominal_action, dtype=float).copy()
    if action.shape != (12,) or not np.all(np.isfinite(action)):
        raise ValueError("A(0) requires one finite 12-D nominal action")
    action[7:10] = 0.0
    return action


def bounded_x_fallback(action_dim: int = 12, duration_s: float = 0.2) -> dict[str, Any]:
    if action_dim != 12 or duration_s <= 0 or duration_s > 1.0:
        raise ValueError("invalid bounded X fallback")
    return {"route": "X", "command": [0.0] * action_dim, "duration_s": float(duration_s),
            "option_boundaries": 1, "terminates_episode": True, "safe_claim": False}


def candidate_feature_vector(record: Mapping[str, Any]) -> np.ndarray:
    """Materialize the frozen 21-D planner-derived candidate schema.

    The caller must provide already-derived scalar features. Raw simulator
    state, controlled-stratum tags, and outcome fields are rejected here.
    """
    forbidden = set(OUTCOME_DENYLIST) | {"stratum", "source_stratum", "qpos", "qvel", "raw_state"}
    leaked = sorted(forbidden.intersection(str(key) for key in record))
    if leaked:
        raise ValueError(f"candidate feature record contains denied fields: {leaked}")
    missing = [name for name in CANDIDATE_FEATURE_FIELDS if name not in record]
    if missing:
        raise ValueError(f"candidate feature record is incomplete: {missing}")
    result = np.asarray([record[name] for name in CANDIDATE_FEATURE_FIELDS], dtype=np.float32)
    if result.shape != (21,) or not np.all(np.isfinite(result)):
        raise ValueError("candidate feature vector must be finite and 21-D")
    return result


def build_minimal_input(context_tokens: np.ndarray, candidate_encoding: Sequence[float]) -> np.ndarray:
    context = np.asarray(context_tokens, dtype=np.float32)
    candidate = np.asarray(candidate_encoding, dtype=np.float32)
    if context.shape != (4, 1024) or candidate.shape != (21,):
        raise ValueError("minimal input requires [4,1024] context and frozen 21-D candidate features")
    result = np.concatenate([context.mean(axis=0), candidate]).astype(np.float32)
    if result.shape != (1045,) or not np.all(np.isfinite(result)):
        raise ValueError("minimal observable/candidate input must be finite and 1045-D")
    return result


def transformed_outputs(raw: Any, *, timeout_s: float) -> dict[str, Any]:
    import torch

    if raw.shape[-1] != 5 or timeout_s <= 0:
        raise ValueError("minimal OBC raw output must end in 5 dimensions")
    return {
        "success": torch.sigmoid(raw[..., 0]),
        "failure": torch.sigmoid(raw[..., 1]),
        "progress": torch.sigmoid(raw[..., 2]),
        "base_path_m": torch.nn.functional.softplus(raw[..., 3]) * 2.0,
        "completion_time_s": torch.nn.functional.softplus(raw[..., 4]) * float(timeout_s),
    }


def minimal_obc_loss(raw: Any, targets: Mapping[str, Any], *, timeout_s: float) -> dict[str, Any]:
    import torch

    names = ("success", "progress", "failure", "base_path_m", "completion_time_s")
    if any(name not in targets for name in names):
        raise ValueError("minimal OBC targets incomplete")
    success = torch.nn.functional.binary_cross_entropy_with_logits(raw[..., 0], targets["success"].float())
    failure = torch.nn.functional.binary_cross_entropy_with_logits(raw[..., 1], targets["failure"].float())
    progress = torch.nn.functional.smooth_l1_loss(torch.sigmoid(raw[..., 2]), targets["progress"].float())
    path = torch.nn.functional.smooth_l1_loss(
        torch.nn.functional.softplus(raw[..., 3]), targets["base_path_m"].float() / 2.0
    )
    timing = torch.nn.functional.smooth_l1_loss(
        torch.nn.functional.softplus(raw[..., 4]), targets["completion_time_s"].float() / float(timeout_s)
    )
    total = success + progress + failure + path + timing
    return {"loss": total, "success": success, "progress": progress, "failure": failure, "path": path, "time": timing}


def make_minimal_obc() -> Any:
    import torch

    return torch.nn.Linear(1045, 5)


def trainable_parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)


def prediction_first_select(candidates: Sequence[Mapping[str, Any]]) -> str:
    hard = [dict(row) for row in candidates if bool(row.get("stage_eligible")) and bool(row.get("hard_valid"))]
    if not hard:
        return "X"
    best_success = max(float(row["predicted_success"]) for row in hard)
    pool = [row for row in hard if best_success - float(row["predicted_success"]) <= 0.05 + 1e-12]
    best_failure = min(float(row["predicted_failure"]) for row in pool)
    pool = [row for row in pool if float(row["predicted_failure"]) - best_failure <= 0.05 + 1e-12]
    best_progress = max(float(row["predicted_progress"]) for row in pool)
    pool = [row for row in pool if best_progress - float(row["predicted_progress"]) <= 0.05 + 1e-12]
    best_path = min(float(row["predicted_base_path_m"]) for row in pool)
    pool = [row for row in pool if float(row["predicted_base_path_m"]) - best_path <= 0.02 + 1e-12]
    best_time = min(float(row["predicted_completion_time_s"]) for row in pool)
    pool = [row for row in pool if float(row["predicted_completion_time_s"]) - best_time <= 1.0 + 1e-12]
    return min(pool, key=lambda row: (ROUTE_ORDER[str(row["route_family"])], str(row["candidate_id"]))) ["candidate_id"]


def geometry_rule_select(candidates: Sequence[Mapping[str, Any]]) -> str:
    hard = [dict(row) for row in candidates if bool(row.get("stage_eligible")) and bool(row.get("hard_valid"))]
    if not hard:
        return "X"
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        required = (
            "minimum_continuous_clearance_m", "minimum_manipulability_or_joint_margin",
            "minimum_policy_view_compatibility", "total_planned_base_path_m", "total_planned_time_s",
        )
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"geometry rule candidate lacks route-wide fields: {missing}")
        return (-float(row["minimum_continuous_clearance_m"]),
                -float(row["minimum_manipulability_or_joint_margin"]),
                -float(row["minimum_policy_view_compatibility"]),
                float(row["total_planned_base_path_m"]), float(row["total_planned_time_s"]),
                ROUTE_ORDER[str(row["route_family"])], str(row["candidate_id"]))
    return min(hard, key=key)["candidate_id"]


def route_oracle(records: Sequence[Mapping[str, Any]]) -> str:
    """Frozen source-level oracle over hard-valid E/D/A route outcomes."""

    valid = [dict(row) for row in records if bool(row.get("hard_valid")) and str(row.get("route_family")) in "EDA"]
    if not valid:
        return "X"
    return min(
        valid,
        key=lambda row: (
            -int(bool(row["success"])),
            int(bool(row["irreversible_or_collision"])),
            -float(row["progress"]),
            float(row["actual_base_path_m"]),
            float(row["completion_time_s"]),
            ROUTE_ORDER[str(row["route_family"])],
        ),
    )["candidate_id"]


def best_fixed_route(train_records: Sequence[Mapping[str, Any]]) -> str:
    """Choose one E/D/A family from train sources only using frozen ties."""

    grouped: dict[str, list[Mapping[str, Any]]] = {route: [] for route in "EDA"}
    for row in train_records:
        route = str(row.get("route_family"))
        if route in grouped:
            grouped[route].append(row)
    if any(not grouped[route] for route in "EDA"):
        raise ValueError("best-fixed selection requires train records for E, D, and A")
    def aggregate(route: str) -> tuple[Any, ...]:
        rows = grouped[route]
        return (
            -sum(bool(row["success"]) for row in rows) / len(rows),
            sum(bool(row["irreversible_or_collision"]) for row in rows),
            -sum(float(row["progress"]) for row in rows) / len(rows),
            sum(float(row["actual_base_path_m"]) for row in rows) / len(rows),
            sum(float(row["completion_time_s"]) for row in rows) / len(rows),
            ROUTE_ORDER[route],
        )
    return min("EDA", key=aggregate)


def validation_route_diversity_gate(
    records: Sequence[Mapping[str, Any]], *, train_selected_best_fixed: str
) -> dict[str, Any]:
    """Compute the predeclared validation-only crossover and oracle gap Gate."""

    if train_selected_best_fixed not in "EDA":
        raise ValueError("train-selected best fixed must be E, D, or A")
    sources: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        sources.setdefault(str(row["source_id"]), []).append(row)
    winning_families: set[str] = set()
    oracle_successes = fixed_successes = 0
    for rows in sources.values():
        hard = [row for row in rows if bool(row.get("hard_valid"))]
        for row in hard:
            competitors = [other for other in hard if other is not row]
            strictly_better = any(
                (bool(row["success"]) and not bool(other["success"]))
                or (
                    bool(row["success"]) == bool(other["success"])
                    and float(row["progress"]) > float(other["progress"])
                    and not bool(row["irreversible_or_collision"])
                    and bool(row["irreversible_or_collision"]) <= bool(other["irreversible_or_collision"])
                )
                for other in competitors
            )
            if strictly_better:
                winning_families.add(str(row["route_family"]))
        oracle_id = route_oracle(rows)
        oracle = next((row for row in rows if row["candidate_id"] == oracle_id), None)
        fixed = next((row for row in rows if row["route_family"] == train_selected_best_fixed), None)
        oracle_successes += int(bool(oracle and oracle["success"]))
        fixed_successes += int(bool(fixed and fixed["success"]))
    gap = oracle_successes - fixed_successes
    passed = len(winning_families) >= 2 and gap >= 2
    return {"passed": passed, "winning_route_families": sorted(winning_families),
            "route_oracle_successes": oracle_successes, "train_best_fixed_successes": fixed_successes,
            "oracle_success_gap": gap, "source_count": len(sources)}


def summarize_test_row(selected_records: Sequence[Mapping[str, Any]], *, source_count: int) -> dict[str, Any]:
    """Frozen all-source-denominator aggregate used by the sole test table."""

    if len(selected_records) != source_count or source_count <= 0:
        raise ValueError("test table rows require exactly one final record per test source")
    for row in selected_records:
        required = ("task", "success", "irreversible_or_collision", "actual_base_path_m", "completion_time_s")
        if any(name not in row for name in required):
            raise ValueError("test record lacks frozen metric fields")
    door = [row for row in selected_records if row["task"] == "CloseSingleDoor"]
    drawer = [row for row in selected_records if row["task"] == "CloseDrawer"]
    if not door or not drawer:
        raise ValueError("test table must include both tasks")
    return {
        "door_success": sum(bool(row["success"]) for row in door),
        "drawer_success": sum(bool(row["success"]) for row in drawer),
        "overall_success": sum(bool(row["success"]) for row in selected_records),
        "irreversible_or_collision_count": sum(bool(row["irreversible_or_collision"]) for row in selected_records),
        "mean_actual_base_path_m": float(np.mean([float(row["actual_base_path_m"]) for row in selected_records])),
        "mean_completion_time_s": float(np.mean([float(row["completion_time_s"]) for row in selected_records])),
        "denominator_all_test_sources": source_count,
        "hard_invalid_or_x_count": sum(not bool(row.get("hard_valid", False)) or row.get("route_family") == "X" for row in selected_records),
    }
