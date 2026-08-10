"""Explicit state adapters for the robosuite controllers used by Mobi-π.

The installed robosuite composite controllers do not expose a state API.  A
blind ``__dict__`` copy would include simulator handles and derived caches, and
could silently restore the wrong state.  This module therefore records only
the documented mutable control state for the current PandaOmron controller
stack and rejects unsupported controller variants.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Mapping, Tuple


CONTROLLER_STATE_SCHEMA_VERSION = "mobipi-controller-v1"


class ControllerStateError(RuntimeError):
    """Base error for controller state capture or restore."""


class UnsupportedControllerState(ControllerStateError):
    """Raised when a controller variant is not covered by the adapter."""


# These fields are the mutable command / controller-memory fields that survive
# a simulator reset and can affect the next policy action.  Joint positions,
# Jacobians, mass matrices, and other simulator-derived caches are deliberately
# excluded; the restored simulator state must be the source of truth for them.
_PERSISTENT_FIELDS = {
    "OperationalSpaceController": (
        "goal_pos",
        "goal_ori",
        "_goal_update_mode",
        "ori_ref",
        "relative_ori",
        "initial_joint",
        "origin_pos",
        "origin_ori",
        "kp",
        "kd",
        "new_update",
    ),
    "MobileBaseJointVelocityController": (
        "goal_qvel",
        "init_pos",
        "init_ori",
        "new_update",
    ),
    "JointPositionController": (
        "goal_qpos",
        "kp",
        "kd",
        "new_update",
    ),
    "SimpleGripController": (
        "goal_qvel",
        "new_update",
    ),
}

_INTERPOLATOR_FIELDS = ("interpolator", "interpolator_pos", "interpolator_ori")


def _class_label(controller: Any) -> str:
    cls = type(controller)
    return f"{cls.__module__}.{cls.__qualname__}"


def _fields_for(controller: Any) -> Tuple[str, ...]:
    class_name = type(controller).__name__
    fields = _PERSISTENT_FIELDS.get(class_name)
    if fields is None:
        raise UnsupportedControllerState(
            f"unsupported controller class {_class_label(controller)}"
        )
    for field in _INTERPOLATOR_FIELDS:
        if hasattr(controller, field) and getattr(controller, field) is not None:
            raise UnsupportedControllerState(
                f"{_class_label(controller)} has an unsupported {field}"
            )
    missing = [field for field in fields if not hasattr(controller, field)]
    if missing:
        raise UnsupportedControllerState(
            f"{_class_label(controller)} is missing mutable fields: {missing}"
        )
    return fields


def capture_controller_state(composite_controller: Any) -> Dict[str, Any]:
    """Capture the explicit mutable state of a composite controller."""

    part_controllers = getattr(composite_controller, "part_controllers", None)
    if not isinstance(part_controllers, Mapping) or not part_controllers:
        raise UnsupportedControllerState(
            "composite controller has no initialized part_controllers"
        )

    parts = {}
    for part_name, controller in part_controllers.items():
        fields = _fields_for(controller)
        parts[str(part_name)] = {
            "class": _class_label(controller),
            "fields": {
                field: copy.deepcopy(getattr(controller, field)) for field in fields
            },
        }

    return {
        "schema_version": CONTROLLER_STATE_SCHEMA_VERSION,
        "composite_class": _class_label(composite_controller),
        "part_order": list(parts),
        "parts": parts,
    }


def restore_controller_state(
    composite_controller: Any, state: Mapping[str, Any]
) -> None:
    """Restore controller state into the controller recreated by env.reset_to."""

    if state.get("schema_version") != CONTROLLER_STATE_SCHEMA_VERSION:
        raise UnsupportedControllerState(
            "unsupported controller state schema: "
            + str(state.get("schema_version"))
        )
    if state.get("composite_class") != _class_label(composite_controller):
        raise UnsupportedControllerState(
            "composite controller class changed between capture and restore"
        )

    part_controllers = getattr(composite_controller, "part_controllers", None)
    saved_parts = state.get("parts")
    if not isinstance(part_controllers, Mapping) or not isinstance(saved_parts, Mapping):
        raise UnsupportedControllerState("controller state has invalid part mapping")
    if set(part_controllers) != set(saved_parts):
        raise UnsupportedControllerState(
            "controller part coverage changed between capture and restore"
        )

    # Validate every part before mutating any part, so unsupported state cannot
    # leave the newly reset controller half-restored.
    validated = []
    for part_name, controller in part_controllers.items():
        saved = saved_parts[part_name]
        if saved.get("class") != _class_label(controller):
            raise UnsupportedControllerState(
                f"controller class changed for part {part_name!r}"
            )
        fields = _fields_for(controller)
        saved_fields = saved.get("fields")
        if not isinstance(saved_fields, Mapping) or set(saved_fields) != set(fields):
            raise UnsupportedControllerState(
                f"mutable field coverage changed for part {part_name!r}"
            )
        validated.append((controller, fields, saved_fields))

    for controller, fields, saved_fields in validated:
        for field in fields:
            setattr(controller, field, copy.deepcopy(saved_fields[field]))


def env_controller_adapters(env: Any) -> Tuple[Callable[[], Any], Callable[[Any], None]]:
    """Return adapters that resolve the *current* controller at call time.

    ``env.reset_to`` recreates robosuite robot controllers.  Resolving lazily
    avoids restoring state into the pre-reset controller object.
    """

    def resolve():
        try:
            return env.unwrapped.env.robots[0].composite_controller
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise UnsupportedControllerState(
                "could not resolve env.unwrapped.env.robots[0].composite_controller"
            ) from exc

    def get_state():
        return capture_controller_state(resolve())

    def set_state(state):
        restore_controller_state(resolve(), state)

    return get_state, set_state
