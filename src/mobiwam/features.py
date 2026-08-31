from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .records import RouteType, Stage


PRIVILEGED_KEYS = frozenset(
    {
        "object_pose",
        "object_state",
        "contact_force",
        "success",
        "irreversible_failure",
        "simulator_state",
        "ground_truth_progress",
    }
)


@dataclass(frozen=True)
class ObservableCandidateInput:
    source_state_id: str
    route_type: RouteType
    stage: Stage
    policy_id: str
    observable_history_uri: str
    candidate_params: Mapping[str, Any]
    nominal_intent_uri: str | None

    def validate(self) -> None:
        if not all((self.source_state_id, self.policy_id, self.observable_history_uri)):
            raise ValueError("observable identity/history fields must be non-empty")
        leaked = PRIVILEGED_KEYS.intersection(self.candidate_params)
        if leaked:
            raise ValueError(f"privileged simulator keys leaked into observable input: {sorted(leaked)}")
        if self.route_type is RouteType.DOCK and self.nominal_intent_uri is not None:
            raise ValueError("D does not preserve the selection-time nominal action chunk")


@dataclass(frozen=True)
class SimulatorOnlyLabels:
    success: bool
    irreversible_failure: bool
    collision: bool
    contact_loss: bool
    progress: float

    def validate(self) -> None:
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress label must lie in [0, 1]")
        if self.success and (self.irreversible_failure or self.collision):
            raise ValueError("unsafe simulator outcome cannot be labeled successful")
