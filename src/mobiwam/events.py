from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .records import RouteType


class EventType(str, Enum):
    QUERY = "QUERY"
    EXECUTE = "EXECUTE"
    REPLAN = "REPLAN"
    MOVE = "MOVE"
    SETTLE = "SETTLE"
    OBSERVE = "OBSERVE"
    RESET = "RESET"
    ASSIST = "ASSIST"
    POST_DOCK_POLICY_READY = "POST_DOCK_POLICY_READY"
    SAFE_EXIT = "SAFE_EXIT"
    TERMINAL = "TERMINAL"


class BoundaryType(str, Enum):
    NONE = "none"
    INTERNAL = "typed_internal"
    COMMON_OUTCOME = "common_outcome"


@dataclass(frozen=True)
class OptionEvent:
    event_type: EventType
    phase_index: int
    boundary: BoundaryType = BoundaryType.NONE
    selection_time_available: bool = True
    parallel_group: str | None = None


def _common_outcome(terminal: bool, phase_index: int) -> OptionEvent:
    return OptionEvent(
        EventType.TERMINAL if terminal else EventType.REPLAN,
        phase_index,
        boundary=BoundaryType.COMMON_OUTCOME,
        selection_time_available=False,
    )


def compile_option_events(
    route_type: RouteType, *, terminal: bool = False, assist_chunks: int = 1
) -> tuple[OptionEvent, ...]:
    if route_type is RouteType.EXECUTE:
        return (
            OptionEvent(EventType.QUERY, 0),
            OptionEvent(EventType.EXECUTE, 1),
            _common_outcome(terminal, 2),
        )
    if route_type is RouteType.DOCK:
        return (
            OptionEvent(EventType.MOVE, 0),
            OptionEvent(EventType.SETTLE, 1, selection_time_available=False),
            OptionEvent(EventType.OBSERVE, 2, selection_time_available=False),
            OptionEvent(EventType.RESET, 3, selection_time_available=False),
            OptionEvent(EventType.QUERY, 4, selection_time_available=False),
            OptionEvent(
                EventType.POST_DOCK_POLICY_READY,
                5,
                boundary=BoundaryType.INTERNAL,
                selection_time_available=False,
            ),
            OptionEvent(EventType.EXECUTE, 6, selection_time_available=False),
            _common_outcome(terminal, 7),
        )
    if route_type is RouteType.ASSIST:
        if assist_chunks <= 0:
            raise ValueError("assist_chunks must be positive")
        events: list[OptionEvent] = []
        for chunk_index in range(assist_chunks):
            events.extend((
                OptionEvent(EventType.QUERY, chunk_index * 3),
                OptionEvent(EventType.EXECUTE, chunk_index * 3 + 1, parallel_group="base_arm_dispatch"),
                OptionEvent(EventType.ASSIST, chunk_index * 3 + 1, parallel_group="base_arm_dispatch"),
            ))
        events.append(_common_outcome(terminal, assist_chunks * 3))
        return tuple(events)
    if route_type is RouteType.ABSTAIN:
        return (
            OptionEvent(
                EventType.SAFE_EXIT,
                0,
                boundary=BoundaryType.COMMON_OUTCOME,
            ),
        )
    raise ValueError(f"unsupported route type: {route_type}")


def validate_event_trace(
    route_type: RouteType, events: Sequence[OptionEvent]
) -> None:
    expected = compile_option_events(
        route_type,
        terminal=bool(events and events[-1].event_type is EventType.TERMINAL),
    )
    if tuple(events) != expected:
        raise ValueError(f"event trace does not match typed {route_type.value} graph")
    common = [event for event in events if event.boundary is BoundaryType.COMMON_OUTCOME]
    if len(common) != 1 or common[0] is not events[-1]:
        raise ValueError("exactly one common outcome boundary must terminate the trace")
    if route_type is RouteType.DOCK:
        internal = [event for event in events if event.boundary is BoundaryType.INTERNAL]
        if len(internal) != 1 or internal[0].event_type is not EventType.POST_DOCK_POLICY_READY:
            raise ValueError("D requires exactly one post_dock_policy_ready boundary")
