#!/usr/bin/env python3
"""V3 development-only mapping and semantic video tools."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from v2_policy_a_repair_video import make_adapter, write, code_commit


def now():
    return datetime.now(timezone.utc).isoformat()


def control_audit(root, task):
    from mobiwam.v3_control import LiveControl
    adapter, snapshot, source = make_adapter(root / "control-audit", task, "qualification", False)
    adapter.restore_source_state(snapshot)
    control=LiveControl(adapter)
    rows=control.clearance()
    velocity,solver=control.allocate(np.zeros(6),np.zeros(3),np.zeros(10))
    result={"at":now(),"code_commit":code_commit(),"source":source,
            "nearest":rows[:20],"solver":solver,"velocity":velocity.tolist(),
            "mapped_action":control.mapped_action(velocity,np.zeros(12)).tolist(),
            "swept":control.swept(velocity),"env_step_calls":0,"outcome_reads":0}
    write(root / f"control-audit-{task}.json",result)
    print(json.dumps(result,indent=2),flush=True)


def probe(root, task, supplemental=False, friction=False):
    from mobiwam.adapters.mobipi import _capture_planar_base_lock
    directory = root / ("mapping-friction" if friction else "mapping-supplement" if supplemental else "probes") / task
    if (directory / "probe.json").exists():
        raise RuntimeError("immutable probe already exists")
    adapter, snapshot, source = make_adapter(root / "probes", task, "qualification", False)
    try:
        restore = adapter.restore_source_state(snapshot)
        if not restore.passed:
            raise RuntimeError("source restore failed")
        raw = adapter._unwrapped()
        robot = raw.robots[0]
        controllers = robot.part_controllers
        controller_info = {}
        for name, controller in controllers.items():
            controller_info[name] = {key: np.asarray(getattr(controller, key)).tolist()
                                    for key in ("input_min", "input_max", "output_min", "output_max",
                                                "input_ref_frame", "input_type", "control_dim", "joint_index")
                                    if hasattr(controller, key)}
            controller_info[name]["class"] = str(type(controller))
        context = adapter._live_articulation_context()
        initial_geoms = [{"id": i, "name": str(name), "type": int(raw.sim.model.geom_type[i]),
                          "size": raw.sim.model.geom_size[i].tolist(),
                          "position": raw.sim.data.geom_xpos[i].tolist(),
                          "rotation": raw.sim.data.geom_xmat[i].tolist(),
                          "contype": int(raw.sim.model.geom_contype[i]),
                          "conaffinity": int(raw.sim.model.geom_conaffinity[i])}
                         for i, name in enumerate(raw.sim.model.geom_names)
                         if str(name).startswith("mobilebase0")]
        write(directory / "binding.json", {"at": now(), "source": source, "restore": asdict(restore),
              "code_commit": code_commit(), "pid": os.getpid(), "controllers": controller_info,
              "action_slices": robot._action_split_indexes, "dt": 1 / raw.control_freq,
              "context": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in context.items()},
              "base_geoms": initial_geoms, "model_nq": raw.sim.model.nq, "model_nv": raw.sim.model.nv})
        rows = []
        for label, axis, sign in [("lock", None, 0)] + [
            (f"axis-{axis}-{sign:+}", axis, sign) for axis in ((7, 8) if friction else (7, 8, 9) if supplemental else (7, 8, 9, 0, 1, 2, 3, 4, 5)) for sign in (1, -1)
        ]:
            if not adapter.restore_source_state(snapshot).passed:
                raise RuntimeError("probe restore failed")
            lock = _capture_planar_base_lock(raw)
            start_base = adapter._origin_pose().copy()
            start_eef = adapter._eef_pose().copy()
            action = np.zeros(12)
            action[-1] = -1
            if axis is not None:
                action[axis] = sign * (0.27 if friction else 0.1 if supplemental else 0.02)
            positions, eefs, qvels, goals = [], [], [], []
            for step in range(40 if axis is None else 2 if friction else 10):
                if axis is None or axis < 7:
                    adapter._step_with_planar_base_lock(action, lock)
                else:
                    adapter.env.step(action)
                positions.append(adapter._origin_pose().tolist())
                eefs.append(adapter._eef_pose().tolist())
                qvels.append(raw.sim.data.qvel[lock.qvel_indices].tolist())
                live_controller = adapter._unwrapped().robots[0].part_controllers["base"]
                goals.append({"goal_qvel": np.asarray(live_controller.goal_qvel).tolist(),
                              "actuator_min": np.asarray(live_controller.actuator_min).tolist(),
                              "actuator_max": np.asarray(live_controller.actuator_max).tolist(),
                              "ctrl": raw.sim.data.ctrl.tolist()})
                if adapter._base_collision():
                    raise RuntimeError(f"base contact during mapping {label} step {step}")
            rows.append({"label": label, "axis": axis, "action": action.tolist(),
                         "start_base": start_base.tolist(), "start_eef": start_eef.tolist(),
                         "base": positions, "eef": eefs, "base_qvel": qvels, "base_goal_qvel": goals})
            write(directory / "probe-partial.json", {"rows": rows, "task_success_reads": 0})
            print(task, label, "base delta", np.asarray(positions)[-1, :3, 3] - start_base[:3, 3], flush=True)
        write(directory / "probe.json", {"at": now(), "rows": rows, "task_success_reads": 0,
              "dt": 1 / raw.control_freq, "source": source, "code_commit": code_commit()})
    finally:
        close = getattr(adapter.env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["probe", "probe-base", "probe-friction", "control-audit"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["CloseDrawer", "CloseSingleDoor"])
    args = parser.parse_args()
    if args.command == "control-audit":
        control_audit(args.artifact_root,args.task)
    else:
        probe(args.artifact_root, args.task, args.command == "probe-base", args.command == "probe-friction")
