from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping


class RouteType(str, Enum):
    EXECUTE = "E"
    DOCK = "D"
    ASSIST = "A"
    ABSTAIN = "X"


class Stage(str, Enum):
    PRECONTACT = "precontact"
    CONTACT = "contact"


class DataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _required(row: Mapping[str, Any], name: str) -> Any:
    if name not in row:
        raise ValueError(f"missing record field: {name}")
    return row[name]


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_float(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    return None if value is None else float(value)


@dataclass(frozen=True)
class SourceStateRecord:
    source_state_id: str
    task_id: str
    task_family: str
    episode_id: str
    instruction: str
    stage: Stage
    split: DataSplit
    environment_seed: int
    policy_name: str
    policy_checkpoint_hash: str
    simulator_version: str
    code_commit: str
    snapshot_hash: str
    observation_hash: str
    snapshot_path: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SourceStateRecord":
        record = cls(
            source_state_id=str(_required(row, "source_state_id")),
            task_id=str(_required(row, "task_id")),
            task_family=str(_required(row, "task_family")),
            episode_id=str(_required(row, "episode_id")),
            instruction=str(_required(row, "instruction")),
            stage=Stage(str(_required(row, "stage"))),
            split=DataSplit(str(_required(row, "split"))),
            environment_seed=int(_required(row, "environment_seed")),
            policy_name=str(_required(row, "policy_name")),
            policy_checkpoint_hash=str(_required(row, "policy_checkpoint_hash")),
            simulator_version=str(_required(row, "simulator_version")),
            code_commit=str(_required(row, "code_commit")),
            snapshot_hash=str(_required(row, "snapshot_hash")),
            observation_hash=str(_required(row, "observation_hash")),
            snapshot_path=str(_required(row, "snapshot_path")),
        )
        record.validate()
        return record

    def validate(self) -> None:
        required_strings = {
            "source_state_id": self.source_state_id,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "policy_name": self.policy_name,
            "policy_checkpoint_hash": self.policy_checkpoint_hash,
            "simulator_version": self.simulator_version,
            "code_commit": self.code_commit,
            "snapshot_hash": self.snapshot_hash,
            "observation_hash": self.observation_hash,
            "snapshot_path": self.snapshot_path,
        }
        empty = sorted(name for name, value in required_strings.items() if not value)
        if empty:
            raise ValueError(f"empty source-state fields: {', '.join(empty)}")
        if self.environment_seed < 0:
            raise ValueError("environment_seed must be non-negative")


@dataclass(frozen=True)
class RouteRolloutRecord:
    schema_version: str
    source_state_id: str
    task_id: str
    task_family: str
    episode_id: str
    split: DataSplit
    stage: Stage
    route_type: RouteType
    candidate_id: str
    repeat_index: int
    environment_seed: int
    policy_seed: int
    route_seed: int
    policy_name: str
    policy_checkpoint_hash: str
    simulator_version: str
    code_commit: str
    snapshot_hash: str
    observation_hash: str
    action_semantics_id: str
    history_protocol_id: str
    transform_check_passed: bool
    restore_check_passed: bool
    stage_eligible: bool
    hard_valid: bool
    success: bool
    irreversible_failure: bool
    collision: bool
    contact_loss: bool
    task_progress_before: float
    task_progress_after: float
    progress_delta: float
    execution_time_s: float
    base_path_length_m: float
    route_cost: float
    invalid_reason: str | None = None
    failure_type: str | None = None
    visibility: float | None = None
    policy_compatibility: float | None = None
    reachability: float | None = None
    joint_margin: float | None = None
    intent_pos_error_p95_m: float | None = None
    intent_rot_error_p95_rad: float | None = None
    contact_state_before: str | None = None
    contact_state_after: str | None = None
    candidate_params: Mapping[str, Any] = field(default_factory=dict)
    source_snapshot_path: str = ""
    video_path: str = ""
    state_trace_path: str = ""
    action_trace_path: str = ""
    labeler_version: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "RouteRolloutRecord":
        boolean_fields = (
            "transform_check_passed",
            "restore_check_passed",
            "stage_eligible",
            "hard_valid",
            "success",
            "irreversible_failure",
            "collision",
            "contact_loss",
        )
        booleans = {
            name: _as_bool(_required(row, name), name) for name in boolean_fields
        }
        record = cls(
            schema_version=str(_required(row, "schema_version")),
            source_state_id=str(_required(row, "source_state_id")),
            task_id=str(_required(row, "task_id")),
            task_family=str(_required(row, "task_family")),
            episode_id=str(_required(row, "episode_id")),
            split=DataSplit(str(_required(row, "split"))),
            stage=Stage(str(_required(row, "stage"))),
            route_type=RouteType(str(_required(row, "route_type"))),
            candidate_id=str(_required(row, "candidate_id")),
            repeat_index=int(_required(row, "repeat_index")),
            environment_seed=int(_required(row, "environment_seed")),
            policy_seed=int(_required(row, "policy_seed")),
            route_seed=int(_required(row, "route_seed")),
            policy_name=str(_required(row, "policy_name")),
            policy_checkpoint_hash=str(_required(row, "policy_checkpoint_hash")),
            simulator_version=str(_required(row, "simulator_version")),
            code_commit=str(_required(row, "code_commit")),
            snapshot_hash=str(_required(row, "snapshot_hash")),
            observation_hash=str(_required(row, "observation_hash")),
            action_semantics_id=str(_required(row, "action_semantics_id")),
            history_protocol_id=str(_required(row, "history_protocol_id")),
            task_progress_before=float(_required(row, "task_progress_before")),
            task_progress_after=float(_required(row, "task_progress_after")),
            progress_delta=float(_required(row, "progress_delta")),
            execution_time_s=float(_required(row, "execution_time_s")),
            base_path_length_m=float(_required(row, "base_path_length_m")),
            route_cost=float(_required(row, "route_cost")),
            invalid_reason=(
                None if row.get("invalid_reason") is None else str(row["invalid_reason"])
            ),
            failure_type=(
                None if row.get("failure_type") is None else str(row["failure_type"])
            ),
            visibility=_optional_float(row, "visibility"),
            policy_compatibility=_optional_float(row, "policy_compatibility"),
            reachability=_optional_float(row, "reachability"),
            joint_margin=_optional_float(row, "joint_margin"),
            intent_pos_error_p95_m=_optional_float(row, "intent_pos_error_p95_m"),
            intent_rot_error_p95_rad=_optional_float(row, "intent_rot_error_p95_rad"),
            contact_state_before=(
                None
                if row.get("contact_state_before") is None
                else str(row["contact_state_before"])
            ),
            contact_state_after=(
                None
                if row.get("contact_state_after") is None
                else str(row["contact_state_after"])
            ),
            candidate_params=dict(row.get("candidate_params", {})),
            source_snapshot_path=str(row.get("source_snapshot_path", "")),
            video_path=str(row.get("video_path", "")),
            state_trace_path=str(row.get("state_trace_path", "")),
            action_trace_path=str(row.get("action_trace_path", "")),
            labeler_version=str(row.get("labeler_version", "")),
            **booleans,
        )
        record.validate()
        return record

    def validate(self) -> None:
        required_strings = {
            "schema_version": self.schema_version,
            "source_state_id": self.source_state_id,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "episode_id": self.episode_id,
            "candidate_id": self.candidate_id,
            "policy_name": self.policy_name,
            "policy_checkpoint_hash": self.policy_checkpoint_hash,
            "simulator_version": self.simulator_version,
            "code_commit": self.code_commit,
            "snapshot_hash": self.snapshot_hash,
            "observation_hash": self.observation_hash,
            "action_semantics_id": self.action_semantics_id,
            "history_protocol_id": self.history_protocol_id,
        }
        empty = sorted(name for name, value in required_strings.items() if not value)
        if empty:
            raise ValueError(f"empty rollout fields: {', '.join(empty)}")
        if self.route_type is RouteType.ABSTAIN:
            raise ValueError("X is a derived decision label, not an executable rollout route")
        if min(self.repeat_index, self.environment_seed, self.policy_seed, self.route_seed) < 0:
            raise ValueError("repeat and seed fields must be non-negative")
        if self.success and (
            not self.stage_eligible
            or not self.hard_valid
            or self.irreversible_failure
            or self.collision
        ):
            raise ValueError("an ineligible, invalid, or unsafe rollout cannot be successful")
        if not self.hard_valid and not self.invalid_reason:
            raise ValueError("invalid_reason is required when hard_valid is false")
        if self.stage is Stage.CONTACT and self.route_type is RouteType.DOCK:
            if self.stage_eligible:
                raise ValueError("D must be ineligible during contact in the minimum protocol")
        for name, value in {
            "task_progress_before": self.task_progress_before,
            "task_progress_after": self.task_progress_after,
            "progress_delta": self.progress_delta,
            "execution_time_s": self.execution_time_s,
            "base_path_length_m": self.base_path_length_m,
            "route_cost": self.route_cost,
        }.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in {
            "execution_time_s": self.execution_time_s,
            "base_path_length_m": self.base_path_length_m,
            "route_cost": self.route_cost,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in {
            "task_progress_before": self.task_progress_before,
            "task_progress_after": self.task_progress_after,
            "visibility": self.visibility,
            "policy_compatibility": self.policy_compatibility,
            "reachability": self.reachability,
        }.items():
            if value is not None:
                if not isfinite(value):
                    raise ValueError(f"{name} must be finite")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1]")
        expected_delta = self.task_progress_after - self.task_progress_before
        if abs(expected_delta - self.progress_delta) > 1e-6:
            raise ValueError("progress_delta must equal progress_after - progress_before")
        for name, value in {
            "intent_pos_error_p95_m": self.intent_pos_error_p95_m,
            "intent_rot_error_p95_rad": self.intent_rot_error_p95_rad,
        }.items():
            if value is not None:
                if not isfinite(value):
                    raise ValueError(f"{name} must be finite")
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")

    @property
    def unsafe(self) -> bool:
        return self.irreversible_failure or self.collision

    def to_candidate_outcome(self) -> "CandidateOutcome":
        return CandidateOutcome(
            snapshot_id=self.source_state_id,
            route_type=self.route_type,
            candidate_id=self.candidate_id,
            seed=self.route_seed,
            stage_eligible=self.stage_eligible,
            hard_valid=self.hard_valid and self.restore_check_passed,
            success=self.success,
            irreversible_failure=self.irreversible_failure,
            collision=self.collision,
            contact_loss=self.contact_loss,
            completion_time_s=self.execution_time_s,
            base_path_m=self.base_path_length_m,
            candidate_params=self.candidate_params,
        )


@dataclass(frozen=True)
class CandidateOutcome:
    snapshot_id: str
    route_type: RouteType
    candidate_id: str
    seed: int
    stage_eligible: bool
    hard_valid: bool
    success: bool
    irreversible_failure: bool
    collision: bool
    contact_loss: bool
    completion_time_s: float
    base_path_m: float
    candidate_params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "CandidateOutcome":
        required = {
            "snapshot_id",
            "route_type",
            "candidate_id",
            "seed",
            "stage_eligible",
            "hard_valid",
            "success",
            "irreversible_failure",
            "collision",
            "contact_loss",
            "completion_time_s",
            "base_path_m",
        }
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"missing outcome fields: {', '.join(missing)}")

        outcome = cls(
            snapshot_id=str(row["snapshot_id"]),
            route_type=RouteType(str(row["route_type"])),
            candidate_id=str(row["candidate_id"]),
            seed=int(row["seed"]),
            stage_eligible=bool(row["stage_eligible"]),
            hard_valid=bool(row["hard_valid"]),
            success=bool(row["success"]),
            irreversible_failure=bool(row["irreversible_failure"]),
            collision=bool(row["collision"]),
            contact_loss=bool(row["contact_loss"]),
            completion_time_s=float(row["completion_time_s"]),
            base_path_m=float(row["base_path_m"]),
            candidate_params=dict(row.get("candidate_params", {})),
        )
        outcome.validate()
        return outcome

    def validate(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id must not be empty")
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.completion_time_s < 0:
            raise ValueError("completion_time_s must be non-negative")
        if self.base_path_m < 0:
            raise ValueError("base_path_m must be non-negative")
        if self.success and (not self.stage_eligible or not self.hard_valid):
            raise ValueError("an ineligible or invalid candidate cannot be successful")

    @property
    def unsafe(self) -> bool:
        return self.irreversible_failure or self.collision
