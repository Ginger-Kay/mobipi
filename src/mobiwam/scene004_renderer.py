"""Single-simulation, single-context renderer for SCENE-004 RI continuation.

This module intentionally operates on frozen MuJoCo XML/state snapshots only.
It does not construct a task environment, call reset/step, or inspect a task
checker.  The executable worker sets EGL variables before importing this
module, then uses one low-level ``MjSim`` and one offscreen context.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mobiwam.scene004 import canonical_hash


NATIVE_WIDTH = 1920
NATIVE_HEIGHT = 1080


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_address(context: Any) -> int | None:
    pointer = getattr(getattr(context, "gl_ctx", None), "_context", None)
    address = getattr(pointer, "address", None)
    if address is None:
        return None
    try:
        return int(address)
    except (TypeError, ValueError):
        return None


def current_context_address() -> int | None:
    from mujoco.egl import egl_ext as EGL

    pointer = EGL.eglGetCurrentContext()
    if not pointer:
        return None
    try:
        return int(pointer.address)
    except (TypeError, ValueError):
        return None


def context_binding(context: Any) -> dict[str, Any]:
    intended = context_address(context)
    current = current_context_address()
    return {
        "intended_context": intended,
        "current_context": current,
        "identity_matches": intended is not None and intended == current,
    }


def load_snapshot_sim(snapshot_dir: Path) -> tuple[Any, Any]:
    """Load one low-level sim from a frozen XML/state pair, without reset."""
    import mujoco
    from robosuite.utils.binding_utils import MjSim

    xml_path = snapshot_dir / "model.xml"
    state_path = snapshot_dir / "sim_state.npy"
    xml = xml_path.read_text()
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth = NATIVE_WIDTH
    model.vis.global_.offheight = NATIVE_HEIGHT
    # Disable GPU multisample resolve: it is a renderer-only setting and
    # removes cross-process edge-pixel nondeterminism without changing the
    # frozen XML/state/camera geometry.
    model.vis.quality.offsamples = 0
    sim = MjSim(model)
    sim.set_state_from_flattened(np.load(state_path, allow_pickle=False))
    sim.forward()
    return sim, model


def create_context(sim: Any, device_id: int) -> Any:
    """Create the sole context after native dimensions are already set."""
    from robosuite.utils.binding_utils import MjRenderContextOffscreen

    context = MjRenderContextOffscreen(
        sim, device_id=int(device_id), max_width=NATIVE_WIDTH, max_height=NATIVE_HEIGHT
    )
    context.gl_ctx.make_current()
    if context.con.offWidth < NATIVE_WIDTH or context.con.offHeight < NATIVE_HEIGHT:
        raise RuntimeError(
            f"native framebuffer too small: {context.con.offWidth}x{context.con.offHeight}"
        )
    return context


def set_snapshot_state(sim: Any, snapshot_dir: Path) -> None:
    """Restore a frozen state into the existing low-level sim; never reset."""
    sim.set_state_from_flattened(np.load(snapshot_dir / "sim_state.npy", allow_pickle=False))
    sim.forward()


def camera_payload_hash(payload: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "cell_key": payload["cell_key"],
            "anchor_xy": payload["anchor_xy"],
            "pose": payload["pose"],
        }
    )


def apply_camera_payload(sim: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a world-fixed camera independent of the frozen snapshot XML."""
    expected = camera_payload_hash(payload)
    if payload.get("camera_hash") != expected:
        raise ValueError("camera payload hash mismatch")
    model = sim.model
    camera_id = model.camera_name2id("freeview")
    if int(model.cam_bodyid[camera_id]) != 0:
        raise RuntimeError("freeview camera is not world-frame fixed")
    anchor = np.asarray(payload["anchor_xy"], dtype=float)
    pose = payload["pose"]
    offset = np.asarray(pose["center_offset_xy"], dtype=float)
    model.cam_pos[camera_id] = [*(anchor + offset), float(pose["height_m"])]
    model.cam_quat[camera_id] = [1.0, 0.0, 0.0, 0.0]
    model.cam_fovy[camera_id] = float(pose["fov_deg"])
    sim.forward()
    return {
        "camera_id": int(camera_id),
        "camera_body_id": int(model.cam_bodyid[camera_id]),
        "camera_hash": expected,
        "world_fixed": True,
        "position": np.asarray(model.cam_pos[camera_id], dtype=float).tolist(),
        "quaternion": np.asarray(model.cam_quat[camera_id], dtype=float).tolist(),
        "fov_deg": float(model.cam_fovy[camera_id]),
    }


def render_rgb(sim: Any, context: Any, camera_name: str = "freeview") -> tuple[np.ndarray, dict[str, Any]]:
    """Make the sole context current immediately before render/read."""
    context.gl_ctx.make_current()
    before = context_binding(context)
    frame = np.asarray(sim.render(camera_name=camera_name, width=NATIVE_WIDTH, height=NATIVE_HEIGHT))[::-1]
    after = context_binding(context)
    if frame.shape != (NATIVE_HEIGHT, NATIVE_WIDTH, 3) or frame.dtype != np.uint8:
        raise RuntimeError(f"invalid native frame {frame.shape} {frame.dtype}")
    receipt = {
        "before_render": before,
        "after_read": after,
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "nonconstant": bool(any(lo != hi for lo, hi in zip(frame.min(axis=(0, 1)), frame.max(axis=(0, 1))))),
        "native": True,
    }
    if not receipt["nonconstant"]:
        raise RuntimeError("native frame is constant")
    return frame, receipt


def close_context(context: Any) -> None:
    """Release the context while still in the worker process."""
    try:
        context.gl_ctx.make_current()
        context.gl_ctx.free()
    finally:
        try:
            context.con.free()
        except Exception:
            pass


def render_seed_group(payload: Mapping[str, Any], output_dir: Path, device_id: int) -> dict[str, Any]:
    """Render E/D/A states sequentially with one sim/context and write receipt."""
    from PIL import Image

    sources = list(payload["sources"])
    if [str(s["stratum"]) for s in sources] != ["E-compatible", "D-required", "A-required"]:
        raise ValueError("source order must be E-compatible, D-required, A-required")
    model_hashes = {sha256(Path(s["snapshot_path"]) / "model.xml") for s in sources}
    if len(model_hashes) != 1:
        raise ValueError("one seed group must share one model XML for one-sim rendering")
    sim, _ = load_snapshot_sim(Path(sources[0]["snapshot_path"]))
    context = create_context(sim, device_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    camera_receipt = apply_camera_payload(sim, payload["camera_payload"])
    try:
        for source in sources:
            snapshot_dir = Path(source["snapshot_path"])
            set_snapshot_state(sim, snapshot_dir)
            camera_receipt = apply_camera_payload(sim, payload["camera_payload"])
            frame, render_receipt = render_rgb(sim, context)
            path = output_dir / f"{source['source_id']}.png"
            Image.fromarray(frame, mode="RGB").save(path)
            frame_rows.append({
                "source_id": source["source_id"],
                "stratum": source["stratum"],
                "snapshot_path": str(snapshot_dir),
                "snapshot_model_sha256": sha256(snapshot_dir / "model.xml"),
                "snapshot_state_sha256": sha256(snapshot_dir / "sim_state.npy"),
                "frame_path": str(path),
                "frame_sha256": sha256(path),
                "frame_bytes": path.stat().st_size,
                "camera_hash": camera_receipt["camera_hash"],
                "render": render_receipt,
            })
    finally:
        close_context(context)
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "gpu": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "native_width": NATIVE_WIDTH,
        "native_height": NATIVE_HEIGHT,
        "task": payload.get("task"),
        "cell": payload.get("cell"),
        "environment_seed": payload.get("environment_seed"),
        "source_count": len(frame_rows),
        "frames": frame_rows,
        "camera": camera_receipt,
        "env_reset_calls": 0,
        "env_step_calls": 0,
        "route_outcome_reads": 0,
        "worker_architecture": "fresh-executable-single-sim-single-context",
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
