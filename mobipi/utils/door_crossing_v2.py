"""Offline, CPU-only hysteresis analysis for dynamic door-plane crossings.

This module intentionally has no simulator, policy, torch, or random-number
dependencies.  It consumes arrays already written by the diagnostic recorder
and never mutates them.  The legacy v1 checker in :mod:`door_diagnostics` is
left untouched; this is a versioned analysis path for forensic comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np


SCHEMA_VERSION = "2.0"
LEGACY_V1_SEMANTICS = "adjacent_positive_to_negative"
DEFAULT_HYSTERESIS_M = 0.005
DEFAULT_UV_TOLERANCE_M = 1.0e-9
DEFAULT_PLANE_MOTION_TOLERANCE_M = 1.0e-9
DEFAULT_CONTROL_FREQUENCY_HZ = 20.0


@dataclass(frozen=True)
class CrossingV2Config:
    """Numerical and geometry rules for the v2 offline analyzer."""

    hysteresis_m: float = DEFAULT_HYSTERESIS_M
    uv_tolerance_m: float = DEFAULT_UV_TOLERANCE_M
    plane_motion_tolerance_m: float = DEFAULT_PLANE_MOTION_TOLERANCE_M
    control_frequency_hz: float = DEFAULT_CONTROL_FREQUENCY_HZ

    def __post_init__(self) -> None:
        if not np.isfinite(self.hysteresis_m) or self.hysteresis_m <= 0.0:
            raise ValueError("hysteresis_m must be finite and positive")
        if not np.isfinite(self.uv_tolerance_m) or self.uv_tolerance_m < 0.0:
            raise ValueError("uv_tolerance_m must be finite and non-negative")
        if not np.isfinite(self.plane_motion_tolerance_m) or self.plane_motion_tolerance_m < 0.0:
            raise ValueError("plane_motion_tolerance_m must be finite and non-negative")
        if not np.isfinite(self.control_frequency_hz) or self.control_frequency_hz <= 0.0:
            raise ValueError("control_frequency_hz must be finite and positive")

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _vector(name: str, value: Any, *, ndim: int, length: int, width: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}")
    if array.shape[0] != length:
        raise ValueError(f"{name} length {array.shape[0]} disagrees with signed distance length {length}")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"{name} must have width {width}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _optional_series(name: str, value: Any, *, length: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _interpolate_scalar(values: Optional[np.ndarray], before: int, fraction: float) -> Optional[float]:
    if values is None:
        return None
    return float(values[before] + fraction * (values[before + 1] - values[before]))


def _interpolate_vector(values: Optional[np.ndarray], before: int, fraction: float) -> Optional[list[float]]:
    if values is None:
        return None
    return (values[before] + fraction * (values[before + 1] - values[before])).astype(np.float64).tolist()


def _interpolation_fraction(before_distance: float, after_distance: float) -> float:
    delta = after_distance - before_distance
    if delta == 0.0:
        return 0.5
    # Interpolate to the geometric plane (signed distance zero), not to a
    # hysteresis boundary.  The event itself is still latched at the first
    # sample that reaches the opposite hysteresis boundary.
    return float(np.clip(-before_distance / delta, 0.0, 1.0))


def _zero_bracket(distance: np.ndarray, start: int, end: int, direction: str) -> tuple[int, int]:
    """Return the first pair of samples that brackets the geometric plane."""

    for index in range(max(start + 1, 1), end + 1):
        before = float(distance[index - 1])
        after = float(distance[index])
        if direction == "positive_to_negative" and before >= 0.0 and after <= 0.0:
            return index - 1, index
        if direction == "negative_to_positive" and before <= 0.0 and after >= 0.0:
            return index - 1, index
    # A threshold event should normally have a sign bracket.  If a malformed
    # or quantized series does not, retain the threshold-adjacent pair rather
    # than inventing a crossing location.
    return end - 1, end


def _finite_region(
    uv: Optional[np.ndarray],
    span: Optional[np.ndarray],
    before: int,
    fraction: float,
    tolerance: float,
) -> tuple[Optional[list[float]], Optional[list[float]], Optional[bool]]:
    if uv is None or span is None:
        return None, None, None
    uv_value = uv[before] + fraction * (uv[before + 1] - uv[before])
    span_value = span[before] + fraction * (span[before + 1] - span[before])
    # The recorder defines u/v from door_p1 to door_p2/door_p3.  This is a
    # conservative rectangular finite-region approximation; it must not be
    # confused with proof of contact with the physical handle/edge.
    inside = bool(
        np.all(uv_value >= -tolerance)
        and np.all(uv_value <= span_value + tolerance)
        and np.all(span_value >= -tolerance)
    )
    return uv_value.astype(np.float64).tolist(), span_value.astype(np.float64).tolist(), inside


def _plane_motion(
    eef: Optional[np.ndarray],
    origin: Optional[np.ndarray],
    normal: Optional[np.ndarray],
    before: int,
    fraction: float,
    tolerance: float,
) -> dict[str, Any]:
    if eef is None or origin is None or normal is None:
        return {
            "available": False,
            "ambiguity": None,
            "classification": "not_available",
        }
    e0, e1 = eef[before], eef[before + 1]
    o0, o1 = origin[before], origin[before + 1]
    n0, n1 = normal[before], normal[before + 1]
    n0_norm = float(np.linalg.norm(n0))
    n1_norm = float(np.linalg.norm(n1))
    if n0_norm == 0.0 or n1_norm == 0.0:
        raise ValueError("door_plane_normal contains a zero vector")
    n0 = n0 / n0_norm
    n1 = n1 / n1_norm
    total_before = float(np.dot(e0 - o0, n0))
    total_after = float(np.dot(e1 - o1, n1))
    # Counterfactual decomposition: move only the EEF while holding the old
    # plane, then move/rotate the plane while holding the new EEF fixed.
    eef_only = float(np.dot(e1 - o0, n0) - total_before)
    plane_only = float(total_after - np.dot(e1 - o0, n0))
    total = float(total_after - total_before)
    denom = abs(eef_only) + abs(plane_only)
    fraction_plane = float(abs(plane_only) / denom) if denom > 0.0 else 0.0
    ambiguity = bool(abs(plane_only) > max(float(tolerance), abs(eef_only)))
    cos_angle = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
    return {
        "available": True,
        "ambiguity": ambiguity,
        "classification": "plane_motion_dominated" if ambiguity else "eef_motion_dominated_or_mixed",
        "eef_delta_m": float(np.linalg.norm(e1 - e0)),
        "plane_origin_delta_m": float(np.linalg.norm(o1 - o0)),
        "normal_angle_rad": float(np.arccos(cos_angle)),
        "distance_change_total_m": total,
        "distance_change_eef_only_m": eef_only,
        "distance_change_plane_only_m": plane_only,
        "plane_motion_fraction": fraction_plane,
        "interpolated_eef_world_pos": (e0 + fraction * (e1 - e0)).astype(np.float64).tolist(),
    }


def analyze_crossing_v2(
    signed_distance_m: Any,
    *,
    door_joint_raw: Any = None,
    door_progress: Any = None,
    eef_plane_projection_uv_m: Any = None,
    door_site_span_uv_m: Any = None,
    eef_world_pos: Any = None,
    door_plane_origin: Any = None,
    door_plane_normal: Any = None,
    config: Optional[CrossingV2Config] = None,
) -> dict[str, Any]:
    """Analyze one already-recorded signed-distance series.

    A positive side is armed only after a sample is ``>= +h``.  Deadband
    samples preserve the armed side.  A positive-to-negative event is latched
    at the first sample ``<= -h``; a later negative-to-positive transition is
    reported separately as a reverse crossing.  The first event is therefore
    not required to be an adjacent jump across both thresholds.
    """

    cfg = config or CrossingV2Config()
    distance = np.asarray(signed_distance_m, dtype=np.float64)
    if distance.ndim != 1 or distance.size == 0:
        raise ValueError(f"signed_distance_m must be a non-empty 1-D array, got {distance.shape}")
    if not np.all(np.isfinite(distance)):
        raise ValueError("signed_distance_m contains NaN or infinite values")
    length = int(distance.shape[0])

    joint = _optional_series("door_joint_raw", door_joint_raw, length=length)
    progress = _optional_series("door_progress", door_progress, length=length)
    uv = None if eef_plane_projection_uv_m is None else _vector(
        "eef_plane_projection_uv_m", eef_plane_projection_uv_m, ndim=2, length=length, width=2
    )
    if door_site_span_uv_m is None:
        span = None
    else:
        raw_span = np.asarray(door_site_span_uv_m, dtype=np.float64)
        if raw_span.shape == (2,):
            if not np.all(np.isfinite(raw_span)):
                raise ValueError("door_site_span_uv_m contains NaN or infinite values")
            span = np.repeat(raw_span[None, :], length, axis=0)
        else:
            span = _vector("door_site_span_uv_m", raw_span, ndim=2, length=length, width=2)
    eef = None if eef_world_pos is None else _vector("eef_world_pos", eef_world_pos, ndim=2, length=length, width=3)
    origin = None if door_plane_origin is None else _vector("door_plane_origin", door_plane_origin, ndim=2, length=length, width=3)
    normal = None if door_plane_normal is None else _vector("door_plane_normal", door_plane_normal, ndim=2, length=length, width=3)
    geometry_available = uv is not None and span is not None
    plane_available = eef is not None and origin is not None and normal is not None

    # v1 is reported for comparison only; no v1 artifact or helper is changed.
    legacy_steps = [
        int(index)
        for index in range(1, length)
        if distance[index - 1] >= cfg.hysteresis_m and distance[index] <= -cfg.hysteresis_m
    ]

    side: Optional[str] = None
    armed_step: Optional[int] = None
    side_entry_step: int = 0
    if distance[0] >= cfg.hysteresis_m:
        side = "positive"
        armed_step = 0
        side_entry_step = 0
    elif distance[0] <= -cfg.hysteresis_m:
        side = "negative"
        side_entry_step = 0
    else:
        side = "deadband"
        side_entry_step = 0
    events: list[dict[str, Any]] = []
    for index in range(1, length):
        value = float(distance[index])
        direction: Optional[str] = None
        if side == "deadband":
            if value >= cfg.hysteresis_m:
                side = "positive"
                armed_step = index
                side_entry_step = index
            elif value <= -cfg.hysteresis_m:
                side = "negative"
                side_entry_step = index
        elif side == "positive" and value <= -cfg.hysteresis_m:
            direction = "positive_to_negative"
        elif side == "negative" and value >= cfg.hysteresis_m:
            direction = "negative_to_positive"
        if direction is None:
            continue

        threshold_before = index - 1
        threshold_after = index
        zero_before, zero_after = _zero_bracket(distance, side_entry_step, index, direction)
        zero_fraction = _interpolation_fraction(float(distance[zero_before]), float(distance[zero_after]))
        deadband_start = int(side_entry_step)
        deadband_indices = np.flatnonzero(
            np.abs(distance[deadband_start + 1 : index]) < cfg.hysteresis_m
        ) + deadband_start + 1
        event: dict[str, Any] = {
            "direction": direction,
            "from_side": "positive" if direction == "positive_to_negative" else "negative",
            "to_side": "negative" if direction == "positive_to_negative" else "positive",
            "armed_step": int(armed_step) if armed_step is not None else None,
            "before_step": int(threshold_before),
            "crossing_step": int(index),
            "interpolated_step": float(zero_before + zero_fraction),
            "interpolated_time_s": float((zero_before + zero_fraction) / cfg.control_frequency_hz),
            "nearest_before": {"step": int(zero_before), "signed_distance_m": float(distance[zero_before])},
            "nearest_after": {"step": int(zero_after), "signed_distance_m": float(distance[zero_after])},
            "threshold_before": {"step": int(threshold_before), "signed_distance_m": float(distance[threshold_before])},
            "threshold_after": {"step": int(threshold_after), "signed_distance_m": value},
            "deadband_steps": int(len(deadband_indices)),
            "deadband_step_indices": deadband_indices.astype(int).tolist(),
            "door_joint_raw": _interpolate_scalar(joint, zero_before, zero_fraction),
            "door_progress": _interpolate_scalar(progress, zero_before, zero_fraction),
            "interpolated_eef_plane_projection_uv_m": None,
            "interpolated_door_site_span_uv_m": None,
            "finite_region_valid": None,
            "finite_region_rule": "0 <= u <= span_u and 0 <= v <= span_v (conservative plane-patch approximation)",
            "plane_motion": _plane_motion(eef, origin, normal, zero_before, zero_fraction, cfg.plane_motion_tolerance_m),
        }
        uv_value, span_value, finite_valid = _finite_region(
            uv, span, zero_before, zero_fraction, cfg.uv_tolerance_m
        )
        event["interpolated_eef_plane_projection_uv_m"] = uv_value
        event["interpolated_door_site_span_uv_m"] = span_value
        event["finite_region_valid"] = finite_valid
        events.append(event)
        side = "negative" if direction == "positive_to_negative" else "positive"
        side_entry_step = index
        armed_step = index if side == "positive" else None

    first_event = events[0] if events else None
    positive_to_negative = [event for event in events if event["direction"] == "positive_to_negative"]
    reverse = [event for event in events if event["direction"] == "negative_to_positive"]
    finite_events = [event for event in positive_to_negative if event["finite_region_valid"] is True]
    ambiguous_events = [event for event in events if event["plane_motion"]["ambiguity"] is True]
    return {
        "schema_version": SCHEMA_VERSION,
        "hysteresis_m": float(cfg.hysteresis_m),
        "control_frequency_hz": float(cfg.control_frequency_hz),
        "state_machine": {
            "armed_positive_rule": "first signed distance >= +hysteresis_m",
            "deadband_rule": "retain the armed side while -hysteresis_m < distance < +hysteresis_m",
            "crossing_rule": "first armed positive sample <= -hysteresis_m",
            "reverse_rule": "after a positive-to-negative event, first sample >= +hysteresis_m",
            "adjacent_jump_required": False,
        },
        "sample_count": length,
        "signed_distance_min_m": float(np.min(distance)),
        "signed_distance_max_m": float(np.max(distance)),
        "legacy_v1_semantics": LEGACY_V1_SEMANTICS,
        "legacy_v1_crossing_steps": legacy_steps,
        "legacy_v1_crossing_count": len(legacy_steps),
        "crossing_count": len(events),
        "crossing_direction": first_event["direction"] if first_event else "none",
        "crossing_directions": [event["direction"] for event in events],
        "relative_plane_side_transition": bool(positive_to_negative),
        "reverse_crossing": bool(reverse),
        "finite_geometry_available": bool(geometry_available),
        "finite_door_region_crossing": bool(finite_events),
        "plane_motion_geometry_available": bool(plane_available),
        "plane_motion_ambiguity": bool(ambiguous_events) if plane_available else None,
        "events": events,
    }


# Descriptive alias for callers that prefer the state-machine name.
crossing_v2_state_machine = analyze_crossing_v2
