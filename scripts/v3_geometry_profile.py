#!/usr/bin/env python3
"""Compile independent V3 movement measurements and base-clearance diagnostics."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def compile_profile(root):
    profiles = {}
    for task in ("CloseDrawer", "CloseSingleDoor"):
        binding = json.loads((root / "probes" / task / "binding.json").read_text())
        probe = json.loads((root / "probes" / task / "probe.json").read_text())
        friction = json.loads((root / "mapping-friction" / task / "probe.json").read_text())
        lock = probe["rows"][0]
        base = np.asarray(lock["base"])[:, :2, 3]
        d = float(np.linalg.norm(base - np.asarray(lock["start_base"])[:2, 3], axis=1).max())
        noise = float(np.linalg.norm(np.diff(base, axis=0), axis=1).max() / probe["dt"])
        increments = [np.linalg.norm(np.diff(np.concatenate([np.asarray(row["start_base"])[None], np.asarray(row["base"])])[:, :2, 3], axis=0), axis=1)
                      for row in friction["rows"][1:]]
        r = float(max(float(x.max()) for x in increments))
        # The horizontal narrow dimension of the physical pedestal collision box.
        feet = next(g for g in binding["base_geoms"] if g["name"] == "mobilebase0_pedestal_feet_col")
        assert feet["type"] == int(mujoco.mjtGeom.mjGEOM_BOX)
        W = 2 * min(feet["size"][:2])
        if d > .02 or r <= 0:
            raise RuntimeError("invalid lock/resolution measurements")
        profiles[task] = {"W_m": W, "d_m": d, "r_m": r, "m_min_m": max(5*d, 2*r, .1*W),
                          "base_speed_noise_bound_mps": noise, "control_dt_s": probe["dt"],
                          "lock_duration_s": len(base)*probe["dt"],
                          "W_source": feet["name"] + " narrow horizontal collision-box dimension",
                          "r_source": "maximum observed one-control-step translation in signed 0.27 friction-response probes",
                          "measurement_error": "finite-step resolution r is a conservative operational bound; d is measured planar lock error",
                          "base_lock_max_m": .02, "A_min_chunks": 3, "A_sync_fraction_min": .4,
                          "A_assist_path_fraction_min": .8, "D_predock_path_fraction_min": .9,
                          "manifold_p95_max_m": .04, "orientation_peak_max_rad": .35,
                          "progress_monotonic_fraction_min": .9, "progress_step_tolerance": 1e-4,
                          "handle_contact_joint_motion_fraction_min": .8, "horizon_steps": 500,
                          "travel_cap_m": .46, "base_collision_inflation_m": .05}
    path = root / "movement-profile-v1.json"
    if path.exists():
        raise RuntimeError("movement profile already frozen")
    write(path, {"frozen_at": datetime.now(timezone.utc).isoformat(), "formula": "max(5*d,2*r,0.10*W)",
                 "qualification_outcomes_before_freeze": 0, "profiles": profiles})
    print(json.dumps(profiles, indent=2), flush=True)


def geometry(root):
    for task in ("CloseDrawer", "CloseSingleDoor"):
        b = json.loads((root / "probes" / task / "binding.json").read_text())
        p = Path(b["source"]["snapshot_path"])
        m = mujoco.MjModel.from_xml_path(str(p / "model.xml")); d = mujoco.MjData(m)
        s = np.load(p / "sim_state.npy"); d.qpos[:] = s[1:1+m.nq]; d.qvel[:] = s[1+m.nq:]
        mujoco.mj_forward(m,d); q = d.qpos.copy()
        bases = [g["id"] for g in b["base_geoms"] if g["contype"]]
        names = [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or "" for i in range(m.ngeom)]
        world = [i for i,n in enumerate(names) if (m.geom_contype[i] or m.geom_conaffinity[i])
                 and not n.startswith(("robot", "mobilebase", "gripper")) and "floor" not in n]
        rows = []
        for v in ([0,0], [.06,0], [-.06,0], [0,.06], [0,-.06]):
            pairs = []
            for frac in np.linspace(0,1,7):
                d.qpos[:] = q; d.qpos[:2] += np.asarray(v)*frac; mujoco.mj_forward(m,d)
                value = min((float(mujoco.mj_geomDistance(m,d,g,h,.2,np.zeros(6))),g,h) for g in bases for h in world)
                pairs.append({"fraction":float(frac), "distance_m":value[0], "pair":[names[value[1]],names[value[2]]]})
            rows.append({"generalized_base_translation":v,"samples":pairs,"inflated_pass":min(x["distance_m"] for x in pairs)>=.05})
        write(root / f"base-geometry-{task}.json", {"at":datetime.now(timezone.utc).isoformat(), "rows":rows,
              "meaning":"base-only translation diagnostic; not full E/D/A path validation", "outcome_reads":0})
        print(task, [(r["generalized_base_translation"], min(x["distance_m"] for x in r["samples"])) for r in rows], flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("command", choices=["profile","geometry"])
    parser.add_argument("root",type=Path); args=parser.parse_args()
    (compile_profile if args.command=="profile" else geometry)(args.root)
