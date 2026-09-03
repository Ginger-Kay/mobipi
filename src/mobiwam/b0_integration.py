from __future__ import annotations

from typing import Mapping, Sequence


def validate_persistent_trace(trace: Mapping[str, object], *, min_chunks: int = 3) -> None:
    chunks = int(trace.get("assist_chunk_count", 0))
    queries = int(trace.get("assist_query_count", 0))
    if chunks < min_chunks:
        raise ValueError(f"persistent A requires at least {min_chunks} chunks, got {chunks}")
    if queries != chunks:
        raise ValueError("A policy query cadence must be exactly once per option boundary")


def validate_a0_identity(e_trace: Mapping[str, Sequence[object]], a0_trace: Mapping[str, Sequence[object]]) -> None:
    keys = ("actions", "states", "observations", "terminal")
    for key in keys:
        if key not in e_trace or key not in a0_trace:
            raise ValueError(f"A(0)=E audit missing trajectory field: {key}")
        if list(e_trace[key]) != list(a0_trace[key]):
            raise ValueError(f"A(0)=E mismatch in {key}")


def validate_native_external_camera(metadata: Mapping[str, object]) -> None:
    if metadata.get("camera_type") != "external_world_frame_fixed":
        raise ValueError("B0 requires a fixed external world-frame camera")
    width, height = int(metadata.get("width", 0)), int(metadata.get("height", 0))
    if width < 1920 or height < 1080:
        raise ValueError("external camera must render natively at least 1920x1080")
    if float(metadata.get("upscale_ratio", 1.0)) > 1.0:
        raise ValueError("upscaled low-resolution source is forbidden")


def validate_scene_template(template: Mapping[str, object]) -> None:
    task = str(template.get("task", ""))
    if task == "CloseSingleDoor" and str(template.get("fixture_type", "")) != "single_hinged_kitchen_cabinet":
        raise ValueError("CloseSingleDoor B0 scene must be a single-hinged kitchen cabinet")
    if task == "CloseDrawer" and str(template.get("fixture_type", "")) != "full_size_kitchen_drawer":
        raise ValueError("CloseDrawer B0 scene must be a full-size kitchen drawer")
    if float(template.get("free_swept_corridor_m", 0.0)) < 0.50:
        raise ValueError("scene swept corridor must be at least 0.50m")


def validate_realized_total_travel(actual_net_translation_m: float, cap_m: float) -> None:
    if actual_net_translation_m > cap_m + 1e-6:
        raise ValueError("realized assist travel exceeded rollout-wide cap")
