"""Outcome-blind deterministic primitives for the B0 scene compiler."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


OFFICIAL_CELLS = ((1, 1), (4, 4), (7, 7), (8, 8), (9, 9))


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


@dataclass(frozen=True)
class SourcePose:
    source_pose_id: str
    x_m: float
    y_m: float
    yaw_rad: float


def source_lattice(cell: tuple[int, int], seed: int) -> tuple[SourcePose, ...]:
    """A finite, stable 3x3x3 lattice in world-frame metres/radians."""
    offsets = (-0.40, 0.0, 0.40)
    yaws = (-0.35, 0.0, 0.35)
    rows = [
        SourcePose(f"l{cell[0]}s{cell[1]}-{i:02d}", x, y, yaw)
        for i, (x, y, yaw) in enumerate((x, y, yaw) for x in offsets for y in offsets for yaw in yaws)
    ]
    # Stable seed only breaks otherwise identical geometry without Python hash().
    return tuple(sorted(rows, key=lambda p: (round(np.hypot(p.x_m, p.y_m), 6), stable_seed(seed, p.source_pose_id), p.source_pose_id)))


def fixture_record(name: str, fixture: object, sim: object) -> dict[str, object]:
    klass = f"{fixture.__class__.__module__}.{fixture.__class__.__name__}"
    size = np.asarray(getattr(fixture, "size", ()), dtype=float).tolist()
    pos = np.asarray(getattr(fixture, "pos", ()), dtype=float).tolist()
    joints = getattr(fixture, "joints", {})
    joint_names = sorted(str(key) for key in (joints.keys() if hasattr(joints, "keys") else joints))
    qpos = {}
    for joint in joint_names:
        try:
            qpos[joint] = float(sim.data.get_joint_qpos(joint))
        except Exception:
            qpos[joint] = None
    return {"name": name, "class": klass, "world_pos_m": pos, "bbox_size_m": size, "joint_names": joint_names, "qpos": qpos}


def validate_fixture(task: str, record: Mapping[str, object]) -> None:
    klass = str(record["class"])
    joints = list(record.get("joint_names", []))
    if task == "CloseSingleDoor":
        if not klass.endswith(".SingleCabinet") or len(joints) != 1 or "hinge" not in joints[0].lower():
            raise ValueError("CloseSingleDoor requires a runtime SingleCabinet with one hinge")
    elif task == "CloseDrawer":
        size = list(record.get("bbox_size_m", []))
        if not klass.endswith(".Drawer") or not joints or max(size, default=0.0) < 0.50:
            raise ValueError("CloseDrawer requires a full-size runtime Drawer")
    else:
        raise ValueError(f"unsupported task {task}")


def dock_target(fixture_position: Sequence[float], fixture_size: Sequence[float]) -> np.ndarray:
    pos, size = np.asarray(fixture_position, float), np.asarray(fixture_size, float)
    if pos.size < 2 or size.size < 2:
        raise ValueError("fixture geometry is incomplete")
    # Front-side, fixture-derived target; never aliases a source pose.
    return np.array([pos[0], pos[1] - size[1] / 2.0 - 0.35], dtype=float)


def validate_d_geometry(start_xy: Sequence[float], dock_xy: Sequence[float]) -> float:
    distance = float(np.linalg.norm(np.asarray(start_xy, float) - np.asarray(dock_xy, float)))
    if distance < 0.30:
        raise ValueError("D source and fixture-derived dock target must differ by >=0.30m")
    return distance


def validate_a_geometry(planned_net_m: float, chunks: int, hard_feasible: bool) -> None:
    if not 0.38 <= planned_net_m <= 0.45 or chunks < 3 or not hard_feasible:
        raise ValueError("A primary a2 must be 0.38--0.45m, >=3 chunks, and hard-feasible")


def continuous_corridor(points: Iterable[Sequence[float]], clearance_m: Iterable[float], *, spacing_m: float) -> dict[str, float]:
    pts, clearance = list(points), list(clearance_m)
    if len(pts) < 2 or len(pts) != len(clearance) or spacing_m > 0.02:
        raise ValueError("continuous corridor requires paired samples at <=0.02m")
    length = sum(float(np.linalg.norm(np.asarray(b) - np.asarray(a))) for a, b in zip(pts, pts[1:]))
    if length < 0.50 or min(clearance) < 0.05:
        raise ValueError("corridor is shorter than 0.50m or violates 0.05m inflated clearance")
    return {"length_m": length, "min_clearance_m": float(min(clearance))}


def validate_native_frame(frame: np.ndarray) -> None:
    if frame.ndim != 3 or frame.shape[0] != 1080 or frame.shape[1] != 1920:
        raise ValueError("camera must return native 1920x1080 RGB, no upscale")
