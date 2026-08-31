from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mobiwam.assist_trajectory import build_truncated_assist_trajectory
from mobiwam.collector import RestoreEvidence, SourceSnapshot
from mobiwam.dataset import assign_group_split
from mobiwam.dock_protocol import DockSettleTimeout, settle_flush_and_reset_policy
from mobiwam.events import compile_option_events
from mobiwam.mobipi_actions import (
    compensate_world_intent,
    invert_pose,
    lock_base,
    nominal_world_intent,
    with_base_command,
)
from mobiwam.mobipi_checkpoint import (
    create_env_from_checkpoint_metadata,
    load_policy_from_checkpoint,
)
from mobiwam.mobipi_policy import FutureChunkEvidence, sample_verified_future_chunk
from mobiwam.records import (
    DataSplit,
    RouteRolloutRecord,
    RouteType,
    SourceStateRecord,
    Stage,
)


ACTION_SEMANTICS_ID = "pandaomron-hybrid-mobile-v1"
HISTORY_PROTOCOL_ID = "postdock-zero-settle-flush10-policy-reset-v1"
VOLATILE_EP_META_KEYS = frozenset({"object_cfgs"})
PLANAR_BASE_JOINT_NAMES = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
)


class SourceStateIneligibleError(RuntimeError):
    """Raised when a simulator state cannot be labeled as precontact."""


# Required by deterministic CUDA BLAS kernels used in paired policy forwards.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class SourceStratum:
    layout_id: int
    style_id: int
    base_noise_sigma: float
    replicate: int


@dataclass
class NominalMacro:
    evidence: FutureChunkEvidence
    policy_seed: int
    e_origin_poses_world: list[np.ndarray] = field(default_factory=list)
    e_eef_poses_world: list[np.ndarray] = field(default_factory=list)
    e_actions: list[np.ndarray] = field(default_factory=list)
    e_desired_eef_poses_world: list[np.ndarray] = field(default_factory=list)

    @property
    def chunk(self) -> np.ndarray:
        return self.evidence.chunk


@dataclass(frozen=True)
class SnapshotPayload:
    env_state: Mapping[str, Any]
    obs_history: Mapping[str, deque[np.ndarray]]
    timestep: int
    python_rng: object
    numpy_rng: tuple[Any, ...]
    torch_rng: Any
    cuda_rng: Any
    env_rng_states: Mapping[str, object]
    controller_state: Mapping[str, Mapping[str, Any]]
    controller_hash: str
    contact_hash: str
    snapshot_hash: str
    observation_hash: str
    progress_before: float


@dataclass
class RolloutTrace:
    actions: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    origin_poses: list[np.ndarray] = field(default_factory=list)
    eef_poses: list[np.ndarray] = field(default_factory=list)
    desired_eef_poses: list[np.ndarray] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)
    base_positions: list[np.ndarray] = field(default_factory=list)
    manipulation_contacts: list[bool] = field(default_factory=list)
    base_reference_xy: np.ndarray | None = None
    intent_pos_errors: list[float] = field(default_factory=list)
    intent_rot_errors: list[float] = field(default_factory=list)
    transform_pos_errors: list[float] = field(default_factory=list)
    transform_rot_errors: list[float] = field(default_factory=list)
    collision: bool = False
    invalid_reason: str | None = None


@dataclass(frozen=True)
class PlanarBaseLock:
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    qpos_values: np.ndarray


def _joint_scalar_address(model: Any, getter_name: str, joint_name: str) -> int:
    getter = getattr(model, getter_name, None)
    if not callable(getter):
        raise RuntimeError(f"MuJoCo model does not expose {getter_name}")
    address = getter(joint_name)
    if isinstance(address, (tuple, list, np.ndarray)):
        values = np.asarray(address).reshape(-1)
        if values.size != 1:
            raise RuntimeError(f"{joint_name} is not a scalar joint")
        address = values[0]
    index = int(address)
    if index < 0:
        raise RuntimeError(f"{joint_name} has an invalid {getter_name} address")
    return index


def _capture_planar_base_lock(raw_env: Any) -> PlanarBaseLock:
    sim = raw_env.sim
    qpos_indices = np.asarray(
        [
            _joint_scalar_address(sim.model, "get_joint_qpos_addr", name)
            for name in PLANAR_BASE_JOINT_NAMES
        ],
        dtype=np.int64,
    )
    qvel_indices = np.asarray(
        [
            _joint_scalar_address(sim.model, "get_joint_qvel_addr", name)
            for name in PLANAR_BASE_JOINT_NAMES
        ],
        dtype=np.int64,
    )
    return PlanarBaseLock(
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        qpos_values=np.asarray(sim.data.qpos[qpos_indices], dtype=np.float64).copy(),
    )


def _apply_planar_base_lock(raw_env: Any, base_lock: PlanarBaseLock) -> None:
    if base_lock.qpos_indices.shape != (3,) or base_lock.qvel_indices.shape != (3,):
        raise ValueError("planar base lock must contain exactly three joint addresses")
    if base_lock.qpos_values.shape != (3,) or not np.all(
        np.isfinite(base_lock.qpos_values)
    ):
        raise ValueError("planar base lock contains invalid qpos values")
    sim = raw_env.sim
    sim.data.qpos[base_lock.qpos_indices] = base_lock.qpos_values
    sim.data.qvel[base_lock.qvel_indices] = 0.0
    sim.forward()


def select_source_stratum(
    source_index: int,
    *,
    layouts: Sequence[int],
    noise_sigmas: Sequence[float],
    states_per_noise_per_layout: int,
) -> SourceStratum:
    if source_index < 0 or not layouts or not noise_sigmas:
        raise ValueError("invalid source-state sampling configuration")
    if states_per_noise_per_layout <= 0:
        raise ValueError("states_per_noise_per_layout must be positive")
    states_per_layout = len(noise_sigmas) * states_per_noise_per_layout
    layout = int(layouts[(source_index // states_per_layout) % len(layouts)])
    within_layout = source_index % states_per_layout
    sigma_index = within_layout // states_per_noise_per_layout
    replicate = within_layout % states_per_noise_per_layout
    return SourceStratum(
        layout_id=layout,
        style_id=layout,
        base_noise_sigma=float(noise_sigmas[sigma_index]),
        replicate=replicate,
    )


def _pose_from_xy_yaw(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    pose[:3, 3] = [x, y, z]
    return pose


def _offset_planar_pose_local(
    pose_world: np.ndarray, offset_local_xy_m: Sequence[float]
) -> np.ndarray:
    pose = np.asarray(pose_world, dtype=np.float64)
    offset = np.asarray(offset_local_xy_m, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("pose_world must have shape (4, 4)")
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise ValueError("offset_local_xy_m must contain two finite values")
    result = pose.copy()
    result[:2, 3] += result[:2, :2] @ offset
    return result


def _detour_pose_from_start(
    reference_pose_world: np.ndarray,
    start_pose_world: np.ndarray,
    delta_local_xy_m: Sequence[float],
    *,
    preserve_start_yaw: bool,
) -> np.ndarray:
    reference = np.asarray(reference_pose_world, dtype=np.float64)
    start = np.asarray(start_pose_world, dtype=np.float64)
    delta = np.asarray(delta_local_xy_m, dtype=np.float64)
    if reference.shape != (4, 4) or start.shape != (4, 4):
        raise ValueError("reference and start poses must have shape (4, 4)")
    if delta.shape != (2,) or not np.all(np.isfinite(delta)):
        raise ValueError("delta_local_xy_m must contain two finite values")

    start_offset_local = reference[:2, :2].T @ (
        start[:2, 3] - reference[:2, 3]
    )
    target = _offset_planar_pose_local(
        reference,
        start_offset_local + delta,
    )
    if preserve_start_yaw:
        target[:3, :3] = start[:3, :3]
    return target


def _default_task_root_pose(raw_env: Any) -> np.ndarray:
    """Read the task's unperturbed base pose from its registered fixture."""

    fixture = getattr(raw_env, "door_fxtr", None)
    if fixture is None:
        fixture = getattr(raw_env, "init_robot_base_pos", None)
    placement = getattr(raw_env, "compute_robot_base_placement_pose", None)
    if fixture is None or not callable(placement):
        raise RuntimeError(
            "task does not expose an initial placement fixture and "
            "compute_robot_base_placement_pose"
        )
    position, orientation = placement(ref_fixture=fixture)
    position = np.asarray(position, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    if position.shape != (3,) or orientation.shape != (3,):
        raise RuntimeError("task default robot pose must contain 3-D position/orientation")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
        raise RuntimeError("task default robot pose contains non-finite values")
    return _pose_from_xy_yaw(
        float(position[0]), float(position[1]), float(orientation[2]), float(position[2])
    )


def _is_mobile_base_geom(name: str | None) -> bool:
    return name is not None and "mobilebase" in name.lower()


def _has_manipulation_contact(raw_env: Any) -> bool:
    sim = raw_env.sim
    for contact in sim.data.contact[: sim.data.ncon]:
        names = (
            (sim.model.geom_id2name(contact.geom1) or "").lower(),
            (sim.model.geom_id2name(contact.geom2) or "").lower(),
        )
        robot_side = [
            ("robot" in name or "gripper" in name or "panda" in name)
            and "mobilebase" not in name
            for name in names
        ]
        if robot_side[0] == robot_side[1]:
            continue
        other = names[1] if robot_side[0] else names[0]
        if "floor" not in other:
            return True
    return False


def _hash_update_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _canonicalize_model_xml(model: Any) -> str:
    # MuJoCo may omit an explicitly serialized identity refquat after reload.
    return str(model).replace(' refquat="1 0 0 0"', "")


def _state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonicalize_model_xml(state.get("model", "")).encode("utf-8"))
    raw_ep_meta = state.get("ep_meta", {})
    ep_meta = json.loads(raw_ep_meta) if isinstance(raw_ep_meta, str) else raw_ep_meta
    if not isinstance(ep_meta, Mapping):
        raise TypeError("episode metadata must be a JSON object")
    stable_ep_meta = {
        str(key): value
        for key, value in ep_meta.items()
        if key not in VOLATILE_EP_META_KEYS
    }
    digest.update(
        json.dumps(
            stable_ep_meta,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _hash_update_array(digest, "states", np.asarray(state["states"]))
    return digest.hexdigest()


def _observation_hash(observation: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        _hash_update_array(digest, key, np.asarray(observation[key]))
    return digest.hexdigest()


def _controller_objects(raw_env: Any) -> dict[str, Any]:
    composite = raw_env.robots[0].composite_controller
    objects = {"composite": composite}
    for attribute in ("part_controllers", "controllers"):
        value = getattr(composite, attribute, None)
        if isinstance(value, Mapping):
            for name, controller in value.items():
                objects[str(name)] = controller
    return objects


def _capture_controller_state(raw_env: Any) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for object_name, controller in _controller_objects(raw_env).items():
        fields: dict[str, Any] = {}
        for name, value in vars(controller).items():
            if isinstance(value, np.ndarray):
                fields[name] = value.copy()
            elif isinstance(value, (bool, int, float, str)) or value is None:
                fields[name] = copy.deepcopy(value)
            elif isinstance(value, tuple) and all(
                isinstance(item, (bool, int, float, str)) or item is None
                for item in value
            ):
                fields[name] = copy.deepcopy(value)
        state[object_name] = fields
    return state


def _restore_controller_state(
    raw_env: Any, state: Mapping[str, Mapping[str, Any]]
) -> None:
    objects = _controller_objects(raw_env)
    if set(objects) != set(state):
        raise RuntimeError(
            f"controller topology changed: {sorted(objects)} != {sorted(state)}"
        )
    for object_name, fields in state.items():
        dictionary = vars(objects[object_name])
        for name, value in fields.items():
            if name not in dictionary:
                raise RuntimeError(f"controller field disappeared: {object_name}.{name}")
            dictionary[name] = copy.deepcopy(value)


def _controller_state_hash(state: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for object_name, fields in sorted(state.items()):
        digest.update(object_name.encode("utf-8"))
        for name, value in sorted(fields.items()):
            digest.update(name.encode("utf-8"))
            if isinstance(value, np.ndarray):
                _hash_update_array(digest, name, value)
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _contact_hash(raw_env: Any) -> str:
    digest = hashlib.sha256()
    contacts = raw_env.sim.data.contact[: raw_env.sim.data.ncon]
    for index, contact in enumerate(contacts):
        digest.update(
            f"{index}:{int(contact.geom1)}:{int(contact.geom2)}".encode("ascii")
        )
        for name in ("dist", "pos", "frame", "friction"):
            _hash_update_array(digest, name, np.asarray(getattr(contact, name)))
    return digest.hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    ).returncode == 0


class MobiPiPairedAdapter:
    def __init__(self, *, output_root: Path, config: Mapping[str, Any]):
        self.output_root = Path(output_root)
        self.config = dict(config)
        repo = Path(self.config["mobipi_repo"])
        if not repo.is_dir():
            raise FileNotFoundError(f"Mobi-pi repo not found: {repo}")
        observed_commit = _git_head(repo)
        expected_commit = str(self.config.get("code_commit", ""))
        if not expected_commit or expected_commit == "BIND_AT_RUN":
            raise RuntimeError("run config must bind the exact clean code_commit")
        if observed_commit != expected_commit:
            raise RuntimeError(
                f"Mobi-pi commit mismatch: {observed_commit} != {expected_commit}"
            )
        upstream_commit = str(self.config["mobipi_upstream_commit"])
        if not _git_is_ancestor(repo, upstream_commit, observed_commit):
            raise RuntimeError("Mobi-pi upstream base is not an ancestor of code_commit")
        robocasa_repo = repo / "external" / "robocasa"
        observed_robocasa_commit = _git_head(robocasa_repo)
        if observed_robocasa_commit != str(self.config["robocasa_commit"]):
            raise RuntimeError(
                "bundled RoboCasa commit does not match the interface contract"
            )
        sys.path.insert(0, str(repo))

        from mobipi.utils.nav_utils import target_base_pose_to_action
        from mobipi.utils.policy_utils import get_config_for_policy

        self._target_base_pose_to_action = target_base_pose_to_action
        self.policy_config, checkpoint_path = get_config_for_policy(
            self.config["checkpoint_root"],
            self.config["data_root"],
            self.config.get("env_name", "CloseSingleDoor"),
            self.config.get("policy_name", "bc_xfmr"),
            seed=int(self.config.get("checkpoint_seed", 1)),
            dataset_name=self.config.get("dataset_name", "mg-300"),
        )
        expected_checkpoint = Path(self.config["policy_checkpoint_path"])
        if Path(checkpoint_path).resolve() != expected_checkpoint.resolve():
            raise RuntimeError(
                f"checkpoint resolver selected {checkpoint_path}, expected {expected_checkpoint}"
            )
        observed_checkpoint_hash = _sha256_file(expected_checkpoint)
        if observed_checkpoint_hash != str(self.config["policy_checkpoint_hash"]):
            raise RuntimeError(
                "policy checkpoint SHA-256 does not match the interface contract"
            )
        (
            self.model,
            self.rollout_policy,
            _,
            self.checkpoint_env_meta,
            self.checkpoint_shape_meta,
        ) = load_policy_from_checkpoint(
            self.policy_config, checkpoint_path
        )
        self.env: Any = None
        self.source_record: SourceStateRecord | None = None
        self.source_snapshot: SourceSnapshot | None = None
        self.stratum: SourceStratum | None = None
        self.environment_seed: int | None = None
        self.dock_origin_pose_world: np.ndarray | None = None
        self._last_restore_passed = False

    def _unwrapped(self) -> Any:
        return self.env.unwrapped.env

    def _stacked_observation(self) -> Mapping[str, np.ndarray]:
        return self.env._get_stacked_obs_from_history()

    def _origin_pose(self) -> np.ndarray:
        position, rotation = self._unwrapped().robots[0].composite_controller.get_controller_base_pose(
            "right"
        )
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        return pose

    def _eef_pose(self) -> np.ndarray:
        robot = self._unwrapped().robots[0]
        site_ids = getattr(robot, "eef_site_id", None)
        if not isinstance(site_ids, Mapping) or "right" not in site_ids:
            raise RuntimeError("PandaOmron must expose eef_site_id['right']")
        site_id = int(site_ids["right"])
        pose = np.eye(4)
        pose[:3, 3] = self._unwrapped().sim.data.site_xpos[site_id]
        pose[:3, :3] = self._unwrapped().sim.data.site_xmat[site_id].reshape(3, 3)
        return pose

    def _sim_state(self) -> np.ndarray:
        return np.asarray(self._unwrapped().sim.get_state().flatten()).copy()

    def _language(self) -> str:
        return str(self.env._ep_lang_str)

    def _task_progress(self) -> float:
        raw = self._unwrapped()
        task = str(self.config.get("env_name", "CloseSingleDoor"))
        if task == "CloseSingleDoor":
            state = raw.door_fxtr.get_door_state(env=raw)
            return float(np.clip(1.0 - max(float(value) for value in state.values()), 0.0, 1.0))
        if task == "CloseDrawer":
            state = raw.drawer.get_door_state(env=raw)
            return float(np.clip(1.0 - max(float(value) for value in state.values()), 0.0, 1.0))
        if task in {"TurnOnFaucet", "TurnOnSinkFaucet"}:
            return float(bool(raw.sink.get_handle_state(env=raw)["water_on"]))
        if task == "TurnOnMicrowave":
            return float(bool(raw.microwave.get_state()["turned_on"]))
        if task == "TurnOnStove":
            value = abs(float(raw.stove.get_knobs_state(env=raw)[raw.knob]))
            return float(np.clip(value / 0.35, 0.0, 1.0))
        raise NotImplementedError(f"task progress adapter is not verified for {task}")

    def _is_success(self) -> bool:
        return bool(self._unwrapped()._check_success())

    def _base_collision(self) -> bool:
        sim = self._unwrapped().sim
        for contact in sim.data.contact[: sim.data.ncon]:
            first = sim.model.geom_id2name(contact.geom1) or ""
            second = sim.model.geom_id2name(contact.geom2) or ""
            first_base = _is_mobile_base_geom(first)
            second_base = _is_mobile_base_geom(second)
            if first_base == second_base:
                continue
            other = second if first_base else first
            if "floor" not in other.lower():
                return True
        return False

    def _base_speed(self, _: Any) -> tuple[float, float]:
        data = self._unwrapped().sim.data
        linear = data.get_body_xvelp("mobilebase0_base")
        angular = data.get_body_xvelr("mobilebase0_base")
        return float(np.linalg.norm(linear[:2])), float(abs(angular[2]))

    def _capture_frame(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        key = str(self.config.get("video_observation_key", "robot0_agentview_right_image"))
        image = np.asarray(observation[key])[-1]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.dtype != np.uint8:
            scale = 255.0 if float(np.max(image)) <= 1.0 else 1.0
            image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
        return image

    def _record_step(
        self,
        trace: RolloutTrace,
        action: np.ndarray,
        observation: Mapping[str, np.ndarray],
        desired: np.ndarray | None = None,
    ) -> None:
        trace.actions.append(np.asarray(action).copy())
        trace.states.append(self._sim_state())
        trace.origin_poses.append(self._origin_pose())
        trace.eef_poses.append(self._eef_pose())
        trace.base_positions.append(trace.origin_poses[-1][:2, 3].copy())
        trace.manipulation_contacts.append(_has_manipulation_contact(self._unwrapped()))
        if bool(self.config.get("save_video", False)):
            trace.frames.append(self._capture_frame(observation))
        if desired is not None:
            trace.desired_eef_poses.append(desired.copy())
            delta_position = trace.eef_poses[-1][:3, 3] - desired[:3, 3]
            delta_rotation = desired[:3, :3].T @ trace.eef_poses[-1][:3, :3]
            trace.intent_pos_errors.append(float(np.linalg.norm(delta_position)))
            angle = np.arccos(np.clip((np.trace(delta_rotation) - 1.0) / 2.0, -1.0, 1.0))
            trace.intent_rot_errors.append(float(angle))
        trace.collision = trace.collision or self._base_collision()

    def _replace_latest_observation_after_base_lock(
        self, action: np.ndarray
    ) -> Mapping[str, np.ndarray]:
        wrapped_env = getattr(self.env, "env", None)
        get_observation = getattr(wrapped_env, "get_observation", None)
        if not callable(get_observation):
            raise RuntimeError("frame-stack environment cannot refresh observations")
        latest = dict(get_observation())
        latest["timesteps"] = np.array([int(self.env.timestep)])
        latest["actions"] = np.asarray(action)[: int(wrapped_env.action_dimension)].copy()
        missing = sorted(set(self.env.obs_history).difference(latest))
        if missing:
            raise RuntimeError(
                "refreshed observation is missing frame-stack keys: "
                + ", ".join(missing)
            )
        for key, history in self.env.obs_history.items():
            history[-1] = np.asarray(latest[key])[None]
        return self._stacked_observation()

    def _step_with_planar_base_lock(
        self, action: np.ndarray, base_lock: PlanarBaseLock
    ) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
        locked_action = lock_base(action)
        self.env.step(locked_action)
        _apply_planar_base_lock(self._unwrapped(), base_lock)
        observation = self._replace_latest_observation_after_base_lock(locked_action)
        return locked_action, observation

    def _seed_policy(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        import torch

        if bool(self.config.get("strict_torch_determinism", True)):
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _generator_states(self) -> dict[str, object]:
        raw = self._unwrapped()
        states = {}
        for name in ("rng", "randomized_robot_base_pose_rng", "place_robot_for_nav_rng"):
            generator = getattr(raw, name, None)
            if generator is not None and hasattr(generator, "bit_generator"):
                states[name] = copy.deepcopy(generator.bit_generator.state)
        placement = getattr(raw, "placement_initializer", None)
        generator = getattr(placement, "rng", None)
        if generator is not None and hasattr(generator, "bit_generator"):
            states["placement_initializer.rng"] = copy.deepcopy(
                generator.bit_generator.state
            )
        return states

    def _restore_generator_states(self, states: Mapping[str, object]) -> None:
        raw = self._unwrapped()
        for name, state in states.items():
            target = (
                getattr(getattr(raw, "placement_initializer"), "rng")
                if name == "placement_initializer.rng"
                else getattr(raw, name)
            )
            target.bit_generator.state = copy.deepcopy(state)

    def prepare_source_state(self, source_index: int, environment_seed: int) -> None:
        self.environment_seed = environment_seed
        self.stratum = select_source_stratum(
            source_index,
            layouts=self.config.get("layouts", [1, 4, 7, 8, 9]),
            noise_sigmas=self.config.get("base_noise_sigmas", [0.0, 0.03, 0.05, 0.10]),
            states_per_noise_per_layout=int(
                self.config.get("states_per_noise_per_layout", 5)
            ),
        )
        if self.env is not None:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
            del self.env
            self.env = None
        camera_names = list(self.checkpoint_env_meta["env_kwargs"]["camera_names"])
        video_camera = str(
            self.config.get("video_camera", "robot0_agentview_right")
        )
        if bool(self.config.get("save_video", False)) and video_camera not in camera_names:
            camera_names.append(video_camera)
        override = {
            "layout_and_style_ids": [[self.stratum.layout_id, self.stratum.style_id]],
            "camera_names": camera_names,
            "randomize_base_init_pose": self.stratum.base_noise_sigma,
            "seed": environment_seed,
        }
        self.env = create_env_from_checkpoint_metadata(
            self.policy_config,
            self.checkpoint_env_meta,
            self.checkpoint_shape_meta,
            override,
        )
        self.env.reset()

        raw = self._unwrapped()
        current_root = _pose_from_xy_yaw(
            float(raw._init_robot_pos[0]),
            float(raw._init_robot_pos[1]),
            float(raw._init_robot_ori[2]),
            float(raw._init_robot_pos[2]),
        )
        default_root = _default_task_root_pose(raw)
        self.dock_origin_pose_world = (
            default_root @ invert_pose(current_root) @ self._origin_pose()
        )
        self.source_record = None
        self.source_snapshot = None

    def _write_snapshot_artifacts(
        self,
        directory: Path,
        payload: SnapshotPayload,
        record: SourceStateRecord,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.xml").write_text(
            str(payload.env_state["model"]), encoding="utf-8"
        )
        (directory / "ep_meta.json").write_text(
            str(payload.env_state.get("ep_meta", "{}")), encoding="utf-8"
        )
        np.save(directory / "sim_state.npy", np.asarray(payload.env_state["states"]))
        history = {
            key: np.concatenate(list(values), axis=0)
            for key, values in payload.obs_history.items()
        }
        np.savez_compressed(directory / "frame_history.npz", **history)
        metadata = {
            "source_state_id": record.source_state_id,
            "snapshot_hash": payload.snapshot_hash,
            "observation_hash": payload.observation_hash,
            "timestep": payload.timestep,
            "environment_seed": record.environment_seed,
            "controller_hash": payload.controller_hash,
            "contact_hash": payload.contact_hash,
        }
        (directory / "snapshot_meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (directory / "rng_state.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "python": payload.python_rng,
                    "numpy": payload.numpy_rng,
                    "torch": payload.torch_rng,
                    "cuda": payload.cuda_rng,
                    "environment_generators": payload.env_rng_states,
                    "controller": payload.controller_state,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def capture_source_state(self) -> SourceSnapshot:
        if self.env is None or self.stratum is None:
            raise RuntimeError("prepare_source_state must run first")
        if self.environment_seed is None:
            raise RuntimeError("source environment seed was not registered")
        if self._base_collision():
            raise SourceStateIneligibleError(
                "refusing PRECONTACT source with initial mobile-base collision: "
                f"environment_seed={self.environment_seed}"
            )
        import torch

        # MuJoCo contacts are derived buffers rather than part of qpos/qvel.
        # Canonicalize them at the same post-forward boundary used by restore.
        self._unwrapped().sim.forward()
        env_state = copy.deepcopy(self.env.get_state())
        history = copy.deepcopy(self.env.obs_history)
        stacked = self._stacked_observation()
        controller_state = _capture_controller_state(self._unwrapped())
        payload = SnapshotPayload(
            env_state=env_state,
            obs_history=history,
            timestep=int(self.env.timestep),
            python_rng=copy.deepcopy(random.getstate()),
            numpy_rng=copy.deepcopy(np.random.get_state()),
            torch_rng=torch.get_rng_state().clone(),
            cuda_rng=(
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
            env_rng_states=self._generator_states(),
            controller_state=controller_state,
            controller_hash=_controller_state_hash(controller_state),
            contact_hash=_contact_hash(self._unwrapped()),
            snapshot_hash=_state_hash(env_state),
            observation_hash=_observation_hash(stacked),
            progress_before=self._task_progress(),
        )
        environment_seed = self.environment_seed
        task_id = str(self.config.get("env_name", "CloseSingleDoor"))
        source_id = (
            f"{task_id}-l{self.stratum.layout_id}-sig{self.stratum.base_noise_sigma:.2f}"
            f"-seed{environment_seed}"
        )
        snapshot_dir = self.output_root / "snapshots" / source_id
        split_map = self.config.get("source_split_map", {})
        split = (
            DataSplit(str(split_map[source_id]))
            if source_id in split_map
            else assign_group_split(source_id)
        )
        record = SourceStateRecord(
            source_state_id=source_id,
            task_id=task_id,
            task_family=(
                "sustained_articulated_contact"
                if task_id in {"CloseSingleDoor", "CloseDrawer"}
                else "precise_local_interaction"
            ),
            episode_id=source_id,
            instruction=self._language(),
            stage=Stage.PRECONTACT,
            split=split,
            environment_seed=environment_seed,
            policy_name=str(self.config.get("policy_name", "bc_xfmr")),
            policy_checkpoint_hash=str(self.config["policy_checkpoint_hash"]),
            simulator_version="robocasa==0.2.0",
            code_commit=str(self.config["code_commit"]),
            snapshot_hash=payload.snapshot_hash,
            observation_hash=payload.observation_hash,
            snapshot_path=str(snapshot_dir),
            layout_id=self.stratum.layout_id,
            collector_batch=int(
                self.config.get("collector_batch", 0)
            ),
            schedule_checksum=str(self.config.get("schedule_checksum", "")),
        )
        self._write_snapshot_artifacts(snapshot_dir, payload, record)
        self.source_record = record
        self.source_snapshot = SourceSnapshot(record=record, opaque_handle=payload)
        return self.source_snapshot

    def restore_source_state(self, snapshot: SourceSnapshot) -> RestoreEvidence:
        payload = snapshot.opaque_handle
        if not isinstance(payload, SnapshotPayload):
            raise TypeError("Mobi-pi adapter received an incompatible snapshot handle")
        restored = self.env.reset_to(copy.deepcopy(payload.env_state))
        if restored is None:
            raise RuntimeError("reset_to did not return an observation")
        self.env.obs_history = copy.deepcopy(payload.obs_history)
        self.env.timestep = payload.timestep
        _restore_controller_state(self._unwrapped(), payload.controller_state)
        self._unwrapped().sim.forward()
        random.setstate(copy.deepcopy(payload.python_rng))
        np.random.set_state(copy.deepcopy(payload.numpy_rng))
        import torch

        torch.set_rng_state(payload.torch_rng.clone())
        if payload.cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.clone() for state in payload.cuda_rng])
        self._restore_generator_states(payload.env_rng_states)
        state_hash = _state_hash(self.env.get_state())
        observation_hash = _observation_hash(self._stacked_observation())
        controller_hash = _controller_state_hash(
            _capture_controller_state(self._unwrapped())
        )
        contact_hash = _contact_hash(self._unwrapped())
        passed = (
            state_hash == payload.snapshot_hash
            and observation_hash == payload.observation_hash
            and controller_hash == payload.controller_hash
            and contact_hash == payload.contact_hash
        )
        self._last_restore_passed = passed
        return RestoreEvidence(
            passed,
            state_hash,
            observation_hash,
            controller_hash,
            contact_hash,
        )

    def sample_nominal_policy(
        self, snapshot: SourceSnapshot, policy_seed: int
    ) -> NominalMacro:
        del snapshot
        self._seed_policy(policy_seed)
        self.rollout_policy.start_episode(lang=self._language())
        evidence = sample_verified_future_chunk(
            self.rollout_policy, self._stacked_observation(), atol=1e-6
        )
        return NominalMacro(evidence=evidence, policy_seed=policy_seed)

    def _run_policy_tail(
        self,
        observation: Mapping[str, np.ndarray],
        trace: RolloutTrace,
        steps_already_run: int,
        *,
        base_target_pose_world: np.ndarray | None = None,
        base_command_gain: float = 1.0,
        planar_base_lock: PlanarBaseLock | None = None,
    ) -> Mapping[str, np.ndarray]:
        if planar_base_lock is not None and base_target_pose_world is not None:
            raise ValueError("hard base lock and target-pose control are mutually exclusive")
        horizon = int(self.config.get("horizon", 500))
        previous_origin = self._origin_pose()
        dt = 1.0 / self._unwrapped().control_freq
        for _ in range(steps_already_run, horizon):
            if self._is_success() or trace.collision:
                break
            action = np.asarray(self.rollout_policy(observation))
            if planar_base_lock is not None:
                action, observation = self._step_with_planar_base_lock(
                    action, planar_base_lock
                )
                self._record_step(trace, action, observation)
                continue
            if base_target_pose_world is None:
                action = lock_base(action)
            else:
                current_origin = self._origin_pose()
                base_command = self._target_base_pose_to_action(
                    base_target_pose_world,
                    current_origin,
                    previous_origin,
                    dt,
                    legacy=bool(self.config.get("legacy_navigation", True)),
                )
                base_command = np.asarray(base_command) * base_command_gain
                previous_origin = current_origin
                action = with_base_command(action, base_command)
            observation, _, _, _ = self.env.step(action)
            self._record_step(trace, action, observation)
        return observation

    def _execute_stationary_nominal(
        self,
        nominal_chunk: NominalMacro,
        trace: RolloutTrace,
        *,
        record_reference_intents: bool,
    ) -> None:
        """Execute the frozen arm policy with the simulator base joints locked."""

        observation = self._stacked_observation()
        base_lock = _capture_planar_base_lock(self._unwrapped())
        trace.base_reference_xy = self._origin_pose()[:2, 3].copy()
        for nominal in nominal_chunk.chunk:
            if self._is_success():
                break
            origin = self._origin_pose()
            eef = self._eef_pose()
            desired = nominal_world_intent(nominal, origin, eef)
            if record_reference_intents:
                nominal_chunk.e_origin_poses_world.append(origin)
                nominal_chunk.e_eef_poses_world.append(eef)
            action, observation = self._step_with_planar_base_lock(nominal, base_lock)
            self._record_step(trace, action, observation, desired)
        observation = self._run_policy_tail(
            observation,
            trace,
            len(trace.actions),
            planar_base_lock=base_lock,
        )
        del observation

    def _save_rollout(
        self,
        route: RouteType,
        repeat_index: int,
        trace: RolloutTrace,
        candidate_id: str,
    ) -> tuple[str, str, str, str]:
        assert self.source_record is not None
        directory = (
            self.output_root
            / "rollouts"
            / self.source_record.source_state_id
            / f"{candidate_id}-r{repeat_index}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "state_trace.npz"
        action_path = directory / "action_trace.npz"
        video_path = directory / "rollout.mp4"
        event_path = directory / "events.json"
        np.savez_compressed(
            state_path,
            states=np.asarray(trace.states),
            origin_poses=np.asarray(trace.origin_poses),
            eef_poses=np.asarray(trace.eef_poses),
            desired_eef_poses=np.asarray(trace.desired_eef_poses),
            base_positions=np.asarray(trace.base_positions),
        )
        np.savez_compressed(action_path, actions=np.asarray(trace.actions))
        event_path.write_text(
            json.dumps(
                [
                    {
                        "event_type": event.event_type.value,
                        "phase_index": event.phase_index,
                        "boundary": event.boundary.value,
                        "selection_time_available": event.selection_time_available,
                        "parallel_group": event.parallel_group,
                    }
                    for event in compile_option_events(
                        route, terminal=self._is_success() or trace.collision
                    )
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rendered_video = ""
        if bool(self.config.get("save_video", False)):
            frames = trace.frames or [self._capture_frame(self._stacked_observation())]
            import imageio.v2 as imageio

            imageio.mimwrite(
                video_path,
                frames,
                fps=int(self.config.get("video_fps", 20)),
                quality=7,
            )
            rendered_video = str(video_path)
        return rendered_video, str(state_path), str(action_path), str(event_path)

    def _record(
        self,
        route: RouteType,
        trace: RolloutTrace,
        *,
        policy_seed: int,
        route_seed: int,
        repeat_index: int,
        candidate_params: Mapping[str, Any],
        progress_before: float,
        candidate_id: str | None = None,
    ) -> RouteRolloutRecord:
        assert self.source_record is not None
        resolved_candidate_id = candidate_id or f"{route.value.lower()}0"
        video, state_trace, action_trace, event_trace = self._save_rollout(
            route, repeat_index, trace, resolved_candidate_id
        )
        progress_after = self._task_progress()
        if trace.base_positions:
            base_positions = np.asarray(trace.base_positions)
            base_reference_xy = (
                np.asarray(trace.base_reference_xy, dtype=np.float64)
                if trace.base_reference_xy is not None
                else base_positions[0]
            )
            max_base_displacement = float(
                np.max(np.linalg.norm(base_positions - base_reference_xy, axis=1))
            )
        else:
            base_positions = np.empty((0, 2))
            max_base_displacement = 0.0
        if len(base_positions) > 1:
            base_path = float(
                np.sum(
                    np.linalg.norm(
                        np.diff(base_positions, axis=0), axis=1
                    )
                )
            )
        else:
            base_path = 0.0
        intent_pos = (
            float(np.percentile(trace.intent_pos_errors, 95))
            if trace.intent_pos_errors
            else None
        )
        intent_rot = (
            float(np.percentile(trace.intent_rot_errors, 95))
            if trace.intent_rot_errors
            else None
        )
        transform_pos = (
            float(np.max(trace.transform_pos_errors))
            if trace.transform_pos_errors
            else 0.0
        )
        transform_rot = (
            float(np.max(trace.transform_rot_errors))
            if trace.transform_rot_errors
            else 0.0
        )
        transform_passed = (
            transform_pos <= float(self.config.get("transform_pos_tolerance_m", 1e-6))
            and transform_rot <= float(
                self.config.get("transform_rot_tolerance_rad", 1e-6)
            )
        )
        if trace.collision and trace.invalid_reason is None:
            trace.invalid_reason = "base_collision"
        if not transform_passed and trace.invalid_reason is None:
            trace.invalid_reason = "intent_tolerance_exceeded"
        hard_valid = trace.invalid_reason is None and not trace.collision and transform_passed
        contact_before = bool(trace.manipulation_contacts[0]) if trace.manipulation_contacts else False
        contact_after = bool(trace.manipulation_contacts[-1]) if trace.manipulation_contacts else False
        contact_loss = any(trace.manipulation_contacts) and not contact_after
        recorded_candidate_params = dict(candidate_params)
        recorded_candidate_params["max_base_displacement_m"] = max_base_displacement
        recorded_candidate_params["transform_closure_pos_max_m"] = transform_pos
        recorded_candidate_params["transform_closure_rot_max_rad"] = transform_rot
        return RouteRolloutRecord(
            schema_version="1.1",
            source_state_id=self.source_record.source_state_id,
            task_id=self.source_record.task_id,
            task_family=self.source_record.task_family,
            episode_id=self.source_record.episode_id,
            split=self.source_record.split,
            stage=self.source_record.stage,
            route_type=route,
            candidate_id=resolved_candidate_id,
            repeat_index=repeat_index,
            environment_seed=self.source_record.environment_seed,
            policy_seed=policy_seed,
            route_seed=route_seed,
            policy_name=self.source_record.policy_name,
            policy_checkpoint_hash=self.source_record.policy_checkpoint_hash,
            simulator_version=self.source_record.simulator_version,
            code_commit=self.source_record.code_commit,
            snapshot_hash=self.source_record.snapshot_hash,
            observation_hash=self.source_record.observation_hash,
            action_semantics_id=ACTION_SEMANTICS_ID,
            history_protocol_id=HISTORY_PROTOCOL_ID,
            transform_check_passed=transform_passed,
            restore_check_passed=self._last_restore_passed,
            stage_eligible=True,
            hard_valid=hard_valid,
            success=hard_valid and self._is_success(),
            irreversible_failure=trace.collision,
            collision=trace.collision,
            contact_loss=contact_loss,
            task_progress_before=progress_before,
            task_progress_after=progress_after,
            progress_delta=progress_after - progress_before,
            execution_time_s=len(trace.actions) / float(self._unwrapped().control_freq),
            base_path_length_m=base_path,
            route_cost=base_path + 0.001 * len(trace.actions),
            invalid_reason=trace.invalid_reason,
            failure_type=(trace.invalid_reason or (None if self._is_success() else "task_failure")),
            intent_pos_error_p95_m=intent_pos,
            intent_rot_error_p95_rad=intent_rot,
            contact_state_before="contact" if contact_before else "no_contact",
            contact_state_after="contact" if contact_after else "no_contact",
            candidate_params=recorded_candidate_params,
            source_snapshot_path=self.source_record.snapshot_path,
            video_path=video,
            state_trace_path=state_trace,
            action_trace_path=action_trace,
            event_trace_path=event_trace,
            labeler_version="mobipi-close-door-v1",
        )

    def execute_e(
        self,
        snapshot: SourceSnapshot,
        nominal_chunk: NominalMacro,
        *,
        policy_seed: int,
        route_seed: int,
        repeat_index: int,
    ) -> RouteRolloutRecord:
        payload = snapshot.opaque_handle
        progress_before = payload.progress_before
        trace = RolloutTrace()
        self._execute_stationary_nominal(
            nominal_chunk,
            trace,
            record_reference_intents=True,
        )
        nominal_chunk.e_actions = [action.copy() for action in trace.actions]
        nominal_chunk.e_desired_eef_poses_world = [
            pose.copy() for pose in trace.desired_eef_poses
        ]
        return self._record(
            RouteType.EXECUTE,
            trace,
            policy_seed=policy_seed,
            route_seed=route_seed,
            repeat_index=repeat_index,
            candidate_params={
                "base_locked": True,
                "base_hold_controller": "mujoco-planar-joint-lock-v1",
                "base_joint_names": list(PLANAR_BASE_JOINT_NAMES),
            },
            progress_before=progress_before,
        )

    def run_vanilla(
        self,
        snapshot: SourceSnapshot,
        *,
        policy_seed: int,
    ) -> RouteRolloutRecord:
        """Run the public RolloutPolicy exactly once per environment step."""

        payload = snapshot.opaque_handle
        self._seed_policy(policy_seed)
        self.rollout_policy.start_episode(lang=self._language())
        observation = self._stacked_observation()
        trace = RolloutTrace()
        for _ in range(int(self.config.get("horizon", 500))):
            if self._is_success() or trace.collision:
                break
            action = np.asarray(self.rollout_policy(observation))
            observation, _, _, _ = self.env.step(action)
            self._record_step(trace, action, observation)
        return self._record(
            RouteType.EXECUTE,
            trace,
            policy_seed=policy_seed,
            route_seed=0,
            repeat_index=0,
            candidate_params={
                "baseline": "public_RolloutPolicy_per_step",
                "base_locked": False,
                "future_chunk_used": False,
            },
            progress_before=payload.progress_before,
        )

    def _navigate_to_dock(
        self,
        trace: RolloutTrace,
        *,
        candidate_params: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, np.ndarray], np.ndarray, dict[str, Any]]:
        assert self.dock_origin_pose_world is not None
        params = dict(candidate_params or {})
        offset_local = np.asarray(
            params.get("target_offset_local_xy_m", [0.0, 0.0]),
            dtype=np.float64,
        )
        target_pose_world = _offset_planar_pose_local(
            self.dock_origin_pose_world,
            offset_local,
        )
        max_steps = int(params.get("dock_max_steps", self.config.get("dock_max_steps", 120)))
        waypoint_delta_raw = params.get("waypoint_delta_from_start_dock_xy_m")
        waypoint_pose_world: np.ndarray | None = None
        waypoint_max_steps = int(params.get("waypoint_max_steps", max_steps))
        if waypoint_delta_raw is not None:
            waypoint_pose_world = _detour_pose_from_start(
                self.dock_origin_pose_world,
                self._origin_pose(),
                waypoint_delta_raw,
                preserve_start_yaw=bool(
                    params.get("waypoint_preserve_start_yaw", True)
                ),
            )
        position_tolerance_m = float(params.get("position_tolerance_m", 0.002))
        yaw_tolerance_rad = float(
            params.get("yaw_tolerance_rad", np.deg2rad(0.5))
        )
        command_gain = float(params.get("command_gain", 1.0))
        if min(max_steps, waypoint_max_steps) <= 0:
            raise ValueError("dock and waypoint max steps must be positive")
        if min(position_tolerance_m, yaw_tolerance_rad, command_gain) <= 0.0:
            raise ValueError("dock tolerances and command_gain must be positive")
        raw = self._unwrapped()
        previous = self._origin_pose()
        observation = self._stacked_observation()
        joint_positions = raw.sim.data.qpos[raw.robots[0]._ref_joint_pos_indexes].copy()
        dt = 1.0 / raw.control_freq
        navigation_targets: list[tuple[str, np.ndarray, int]] = []
        if waypoint_pose_world is not None:
            navigation_targets.append(("detour", waypoint_pose_world, waypoint_max_steps))
        navigation_targets.append(("dock", target_pose_world, max_steps))
        waypoint_reached = waypoint_pose_world is None
        dock_reached = False
        for stage_name, stage_target, stage_max_steps in navigation_targets:
            stage_reached = False
            for _ in range(stage_max_steps):
                current = self._origin_pose()
                position_error = np.linalg.norm(
                    stage_target[:2, 3] - current[:2, 3]
                )
                yaw_current = np.arctan2(current[1, 0], current[0, 0])
                yaw_target = np.arctan2(stage_target[1, 0], stage_target[0, 0])
                yaw_error = abs(
                    (yaw_target - yaw_current + np.pi) % (2 * np.pi) - np.pi
                )
                if (
                    position_error <= position_tolerance_m
                    and yaw_error <= yaw_tolerance_rad
                ):
                    stage_reached = True
                    break
                command = self._target_base_pose_to_action(
                    stage_target,
                    current,
                    previous,
                    dt,
                    legacy=bool(self.config.get("legacy_navigation", True)),
                )
                command = np.asarray(command) * command_gain
                action = with_base_command(np.zeros(12), command)
                previous = current
                observation, _, _, _ = self.env.step(action)
                raw.sim.data.qpos[raw.robots[0]._ref_joint_pos_indexes] = joint_positions
                raw.sim.forward()
                self._record_step(trace, action, observation)
                if trace.collision:
                    trace.invalid_reason = f"base_collision_during_{stage_name}"
                    break
            if not stage_reached:
                if trace.invalid_reason is None:
                    trace.invalid_reason = f"{stage_name}_timeout"
                break
            if stage_name == "detour":
                waypoint_reached = True
            else:
                dock_reached = True
        realized = {
            **params,
            "target_offset_local_xy_m": offset_local.tolist(),
            "dock_max_steps": max_steps,
            "waypoint_max_steps": waypoint_max_steps,
            "position_tolerance_m": position_tolerance_m,
            "yaw_tolerance_rad": yaw_tolerance_rad,
            "command_gain": command_gain,
            "target_pose_world_xy": target_pose_world[:2, 3].tolist(),
            "waypoint_pose_world_xy": (
                None
                if waypoint_pose_world is None
                else waypoint_pose_world[:2, 3].tolist()
            ),
            "waypoint_reached": waypoint_reached,
            "dock_reached": dock_reached,
        }
        return observation, target_pose_world, realized

    def execute_d(
        self,
        snapshot: SourceSnapshot,
        *,
        policy_seed: int,
        route_seed: int,
        repeat_index: int,
        candidate_id: str = "d0",
        candidate_params: Mapping[str, Any] | None = None,
    ) -> RouteRolloutRecord:
        payload = snapshot.opaque_handle
        trace = RolloutTrace()
        query_ready_metadata: dict[str, Any] = {
            "post_dock_policy_ready": False,
            "history_reset_protocol": HISTORY_PROTOCOL_ID,
        }
        observation, dock_target_pose_world, realized_params = self._navigate_to_dock(
            trace,
            candidate_params=candidate_params,
        )
        if trace.invalid_reason is None:
            self._seed_policy(policy_seed)
            try:
                settle_flush_and_reset_policy(
                    self.env,
                    self.rollout_policy,
                    language=self._language(),
                    velocity_reader=self._base_speed,
                    history_length=int(self.config.get("history_length", 10)),
                    linear_threshold_mps=float(
                        self.config.get("settle_linear_threshold_mps", 0.005)
                    ),
                    angular_threshold_radps=float(
                        self.config.get("settle_angular_threshold_radps", 0.02)
                    ),
                    max_steps=int(self.config.get("settle_max_steps", 200)),
                    step_callback=lambda _env, action, result: self._record_step(
                        trace, action, result[0]
                    ),
                )
            except DockSettleTimeout:
                trace.invalid_reason = "dock_settle_timeout"
        if trace.invalid_reason is None:
            observation = self._stacked_observation()
            dock_macro = sample_verified_future_chunk(
                self.rollout_policy, observation, atol=1e-6
            )
            query_ready_metadata.update(
                {
                    "post_dock_policy_ready": True,
                    "query_ready_timestamp_ns": time.time_ns(),
                    "post_dock_observation_hash": _observation_hash(observation),
                    "history_reset_fingerprint": _observation_hash(observation),
                    "policy_query_seed": policy_seed,
                    "policy_query_first_action_max_abs_error": dock_macro.evidence.max_abs_error,
                }
            )
            manipulation_steps = 0
            for action in dock_macro.chunk:
                if self._is_success():
                    break
                action = lock_base(action)
                observation, _, _, _ = self.env.step(action)
                self._record_step(trace, action, observation)
                manipulation_steps += 1
            observation = self._run_policy_tail(
                observation,
                trace,
                manipulation_steps,
                base_target_pose_world=dock_target_pose_world,
            )
        del observation
        return self._record(
            RouteType.DOCK,
            trace,
            policy_seed=policy_seed,
            route_seed=route_seed,
            repeat_index=repeat_index,
            candidate_params={
                **realized_params,
                **query_ready_metadata,
                "target": "task_compute_robot_base_placement_pose",
                "navigation": "short_closed_loop_direct",
            },
            progress_before=payload.progress_before,
            candidate_id=candidate_id,
        )

    def execute_a(
        self,
        snapshot: SourceSnapshot,
        nominal_chunk: NominalMacro,
        *,
        policy_seed: int,
        route_seed: int,
        repeat_index: int,
        candidate_id: str = "a0",
        candidate_params: Mapping[str, Any] | None = None,
    ) -> RouteRolloutRecord:
        payload = snapshot.opaque_handle
        params = dict(candidate_params or {})
        trace = RolloutTrace()
        observation = self._stacked_observation()
        self._seed_policy(policy_seed)
        self.rollout_policy.start_episode(lang=self._language())
        replayed_nominal = sample_verified_future_chunk(
            self.rollout_policy, observation, atol=1e-6
        )
        nominal_replay_error = float(
            np.max(np.abs(replayed_nominal.chunk - nominal_chunk.chunk))
        )
        if nominal_replay_error > 1e-6:
            raise RuntimeError(
                "A nominal replay differs from E after source restore: "
                f"max abs error {nominal_replay_error:.3e}"
            )
        count = len(nominal_chunk.e_origin_poses_world)
        if count != len(nominal_chunk.e_eef_poses_world):
            raise RuntimeError("E did not produce aligned nominal intent traces")
        if self.dock_origin_pose_world is None:
            raise RuntimeError("canonical dock pose was not prepared")
        assist_target_delta_raw = params.get("target_delta_from_start_dock_xy_m")
        assist_target_offset_local: np.ndarray | None = None
        if assist_target_delta_raw is None:
            assist_target_offset_local = np.asarray(
                params.get("target_offset_local_xy_m", [0.0, 0.0]),
                dtype=np.float64,
            )
            assist_target_pose_world = _offset_planar_pose_local(
                self.dock_origin_pose_world,
                assist_target_offset_local,
            )
            assist_target_mode = "dock_offset"
        else:
            assist_target_pose_world = _detour_pose_from_start(
                self.dock_origin_pose_world,
                self._origin_pose(),
                assist_target_delta_raw,
                preserve_start_yaw=bool(
                    params.get("target_preserve_start_yaw", True)
                ),
            )
            assist_target_mode = "start_relative_detour"
        trajectory = build_truncated_assist_trajectory(
            self._origin_pose(),
            assist_target_pose_world,
            steps=max(count, 1),
            fraction_toward_dock=float(
                params.get(
                    "fraction_toward_dock",
                    self.config.get("assist_fraction_toward_dock", 0.25),
                )
            ),
            max_translation_m=float(
                params.get(
                    "max_translation_m",
                    self.config.get("assist_max_translation_m", 0.05),
                )
            ),
            max_yaw_rad=float(
                params.get(
                    "max_yaw_rad",
                    self.config.get("assist_max_yaw_rad", np.deg2rad(3.0)),
                )
            ),
        )
        exact_zero_assist = (
            trajectory.translation_m == 0.0 and trajectory.yaw_rad == 0.0
        )
        if exact_zero_assist:
            if not nominal_chunk.e_actions:
                raise RuntimeError("E did not provide an action trace for A(0)")
            base_lock = _capture_planar_base_lock(self._unwrapped())
            trace.base_reference_xy = self._origin_pose()[:2, 3].copy()
            observation = self._stacked_observation()
            for index, reference_action in enumerate(nominal_chunk.e_actions):
                if self._is_success():
                    break
                desired = (
                    nominal_chunk.e_desired_eef_poses_world[index]
                    if index < len(nominal_chunk.e_desired_eef_poses_world)
                    else None
                )
                action, observation = self._step_with_planar_base_lock(
                    reference_action, base_lock
                )
                self._record_step(trace, action, observation, desired)
                if trace.collision:
                    trace.invalid_reason = "base_collision_during_zero_assist_replay"
                    break
        else:
            previous_origin = self._origin_pose()
            dt = 1.0 / self._unwrapped().control_freq
            for index in range(count):
                if self._is_success():
                    break
                current_origin = self._origin_pose()
                current_eef = self._eef_pose()
                compensation = compensate_world_intent(
                    nominal_chunk.chunk[index],
                    nominal_origin_pose_world=nominal_chunk.e_origin_poses_world[index],
                    nominal_eef_pose_world=nominal_chunk.e_eef_poses_world[index],
                    assist_origin_pose_world_current=current_origin,
                    assist_origin_pose_world_next=trajectory.poses_world[index + 1],
                    assist_eef_pose_world_current=current_eef,
                )
                trace.transform_pos_errors.append(
                    compensation.transform_closure_pos_error_m
                )
                trace.transform_rot_errors.append(
                    compensation.transform_closure_rot_error_rad
                )
                if compensation.saturated:
                    trace.invalid_reason = "arm_compensation_saturated"
                    break
                base_command = self._target_base_pose_to_action(
                    trajectory.poses_world[index + 1],
                    current_origin,
                    previous_origin,
                    dt,
                    legacy=bool(self.config.get("legacy_navigation", True)),
                )
                previous_origin = current_origin
                action = with_base_command(compensation.action, base_command)
                observation, _, _, _ = self.env.step(action)
                self._record_step(
                    trace, action, observation, compensation.desired_eef_pose_world
                )
                if trace.collision:
                    trace.invalid_reason = "base_collision_during_assist"
                    break
            if trace.invalid_reason is None:
                observation = self._run_policy_tail(
                    observation,
                    trace,
                    len(trace.actions),
                    base_target_pose_world=trajectory.poses_world[-1],
                )
        del observation
        assist_realized_params = {
            **params,
            "assist_target_mode": assist_target_mode,
            "assist_target_pose_world_xy": assist_target_pose_world[:2, 3].tolist(),
            "fraction_toward_dock": trajectory.fraction_toward_dock,
            "translation_m": trajectory.translation_m,
            "yaw_rad": trajectory.yaw_rad,
            "nominal_replay_max_abs_error": nominal_replay_error,
            "base_hold_controller": (
                "mujoco-planar-joint-lock-v1"
                if exact_zero_assist
                else "mobipi-target-pose-v3"
            ),
            "exact_zero_assist_replay": exact_zero_assist,
            "zero_assist_action_source": (
                "paired_execute_trace" if exact_zero_assist else None
            ),
        }
        if assist_target_offset_local is not None:
            assist_realized_params["target_offset_local_xy_m"] = (
                assist_target_offset_local.tolist()
            )
        return self._record(
            RouteType.ASSIST,
            trace,
            policy_seed=policy_seed,
            route_seed=route_seed,
            repeat_index=repeat_index,
            candidate_params=assist_realized_params,
            progress_before=payload.progress_before,
            candidate_id=candidate_id,
        )


def create_adapter(*, output_root: Path, config: Mapping[str, Any]) -> MobiPiPairedAdapter:
    return MobiPiPairedAdapter(output_root=output_root, config=config)
