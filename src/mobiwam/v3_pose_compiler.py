"""Bounded, outcome-blind same-cell pose and whole-path IK compiler."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize

from mobiwam.mobipi_actions import axis_angle_to_matrix, matrix_to_axis_angle


def serialize(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=serialize)+"\n")


def compile_candidates(control,root:Path,task:str):
    m=control.model
    data=mujoco.MjData(m)
    data.qpos[:]=control.data.qpos
    original=data.qpos.copy()
    mujoco.mj_forward(m,data)
    context=control.context
    qidx=context["qpos_index"]
    axis=np.asarray(context["axis"])
    origin=np.asarray(context["origin"])
    handle=control.handle_pose()
    eef_rotation=data.site_xmat[control.site].reshape(3,3).copy()
    outward=-axis if context["joint_type"]=="prismatic" else np.cross(axis,handle[:3,3]-origin)
    # Choose the outward normal facing the current base, before all candidates.
    if outward@(control.adapter._origin_pose()[:3,3]-handle[:3,3])<0: outward=-outward
    outward/=np.linalg.norm(outward)
    base_near=[r for r in control.clearance(gradients=True) if r["pair"][0].startswith("mobilebase0")]
    closest=min(base_near,key=lambda r:r["signed_clearance_m"])
    gradient=np.asarray(closest["gradient"][:2]); gradient/=max(np.linalg.norm(gradient),1e-12)
    min_shift=max(0.,-closest["signed_clearance_m"]+.01)
    # Five geometrically spaced proposals only. Range is tied to the measured
    # footprint (0.50m), clearance deficit and existing travel cap.
    shifts=[min_shift+.025*i for i in range(5)]
    freeze={"frozen_at":datetime.now(timezone.utc).isoformat(),"task":task,
            "trigger":control.clearance()[:5],"base_outward_generalized_direction":gradient,
            "base_shift_m":shifts,"precontact_offset_m":.12,"contact_offset_m":.015,
            "arm_orientation":"original source world orientation transported by live fixture manifold",
            "candidate_seed":20260907,"candidate_order":list(range(5)),
            "ranking":["full E/D/A hard validity","minimum clearance","joint margin","policy view","path","stable ID"],
            "fixture_q_unchanged":float(original[qidx]),"outcome_reads":0}
    directory=root/"pose-compiler"/task
    if (directory/"proposals-freeze.json").exists():
        prior=json.loads((directory/"proposals-freeze.json").read_text())
        if not np.allclose(prior["base_shift_m"],shifts) or not np.allclose(prior["base_outward_generalized_direction"],gradient):
            raise RuntimeError("frozen proposals changed")
        directory=directory/"hard-feasibility-repair"
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/"decision.json").exists():raise RuntimeError("repaired candidate computation already complete")
        write(directory/"repair.json",{"reason":"pose/collision merit tradeoff rejected near-feasible knots; impose pose tolerance and collision as joint hard constraints", "proposal_freeze":"../proposals-freeze.json", "new_proposals":0,"outcome_reads":0})
    else:
        write(directory/"proposals-freeze.json",freeze)
    qarm=np.asarray(control.arm.qpos_index,int)
    joint_ids=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in qarm]
    lower=m.jnt_range[joint_ids,0]+.015; upper=m.jnt_range[joint_ids,1]-.015

    def solve(target,rotation):
        initial=np.clip(data.qpos[qarm],lower+1e-8,upper-1e-8)
        # First solve exact live 6D IK; then optimize collision residuals for
        # the actual nearest pairs, preserving the same target and joint limits.
        def residual(q,near_pairs):
            data.qpos[qarm]=q;mujoco.mj_forward(m,data)
            pos=data.site_xpos[control.site]-target
            rot=matrix_to_axis_angle(rotation@data.site_xmat[control.site].reshape(3,3).T)
            values=[*pos,*(rot*.15),*((q-initial)*.002)]
            for g,h,margin in near_pairs:
                distance=mujoco.mj_geomDistance(m,data,g,h,.12,np.zeros(6))
                values.append(min(0.,distance-margin)*3)
            return np.asarray(values)
        pairs=[]
        for stage in range(3):
            fit=least_squares(lambda q:residual(q,pairs),initial,bounds=(lower,upper),max_nfev=80,
                              ftol=1e-7,xtol=1e-7,gtol=1e-7)
            data.qpos[qarm]=fit.x;mujoco.mj_forward(m,data)
            rows=control.clearance(data)
            poserr=float(np.linalg.norm(data.site_xpos[control.site]-target))
            roterr=float(np.linalg.norm(matrix_to_axis_angle(rotation@data.site_xmat[control.site].reshape(3,3).T)))
            if poserr<.01 and roterr<.15 and (not rows or rows[0]["signed_clearance_m"]>=-1e-6):break
            near_names={tuple(r["pair"]) for r in rows if r["signed_clearance_m"]<.02}
            pairs=[(g,h,margin) for g,h,margin in control.pairs if (control.names[g],control.names[h]) in near_names]
            initial=fit.x
        if not (poserr<.01 and roterr<.15 and (not rows or rows[0]["signed_clearance_m"]>=-1e-6)):
            near_names={tuple(r["pair"]) for r in rows if r["signed_clearance_m"]<.025}
            pairs=[(g,h,margin) for g,h,margin in control.pairs if (control.names[g],control.names[h]) in near_names]
            def constraints(q):
                data.qpos[qarm]=q;mujoco.mj_forward(m,data)
                pos=np.linalg.norm(data.site_xpos[control.site]-target)
                rot=np.linalg.norm(matrix_to_axis_angle(rotation@data.site_xmat[control.site].reshape(3,3).T))
                return np.asarray([*[mujoco.mj_geomDistance(m,data,g,h,.12,np.zeros(6))-margin for g,h,margin in pairs],.00999-pos,.1499-rot])
            hard=minimize(lambda q:float(np.sum((q-initial)**2))*.0001,fit.x,method="SLSQP",
                          bounds=list(zip(lower,upper)),constraints=[{"type":"ineq","fun":constraints}],
                          options={"maxiter":160,"ftol":1e-11})
            data.qpos[qarm]=hard.x;mujoco.mj_forward(m,data)
            rows=control.clearance(data)
            poserr=float(np.linalg.norm(data.site_xpos[control.site]-target))
            roterr=float(np.linalg.norm(matrix_to_axis_angle(rotation@data.site_xmat[control.site].reshape(3,3).T)))
            fit.x=hard.x
        return {"passed":poserr<.01 and roterr<.15 and (not rows or rows[0]["signed_clearance_m"]>=-1e-6),
                "position_error_m":poserr,"orientation_error_rad":roterr,"nearest":rows[:3],
                "joint_margin_rad":float(np.min(np.minimum(fit.x-lower,upper-fit.x))),"iterations":int(fit.nfev)}

    candidates=[]
    for index,shift in enumerate(shifts):
        data.qpos[:]=original; data.qpos[control.base.qpos_index[:2]]+=gradient*shift
        source_receipt=solve(handle[:3,3]+outward*.12,eef_rotation)
        row={"candidate_id":f"pose-{index}","source_ik":source_receipt,"hard_valid":False,"routes":{}}
        np.save(directory/f"pose-{index}-qpos.npy",data.qpos)
        if source_receipt["passed"]:
            source=data.qpos.copy()
            for route in ("E","D","A"):
                data.qpos[:]=source
                route_rows=[]
                # Base shift for D is pre-contact, A is coupled to articulation.
                travel=.06*gradient
                if route=="D":
                    for fraction in np.linspace(0,1,7)[1:]:
                        data.qpos[control.base.qpos_index[:2]]=source[control.base.qpos_index[:2]]+travel*fraction
                        rec=solve(handle[:3,3]+outward*.12,eef_rotation);route_rows.append(rec)
                        if not rec["passed"]:break
                if all(r["passed"] for r in route_rows):
                    for offset in np.linspace(.12,.015,12):
                        rec=solve(handle[:3,3]+outward*offset,eef_rotation);route_rows.append(rec)
                        if not rec["passed"]:break
                if all(r["passed"] for r in route_rows):
                    for fraction in np.linspace(0,1,41):
                        q=float(original[qidx]+fraction*(context["q_goal"]-original[qidx]));data.qpos[qidx]=q
                        if route=="A":data.qpos[control.base.qpos_index[:2]]=source[control.base.qpos_index[:2]]+travel*fraction
                        if context["joint_type"]=="prismatic":
                            target=handle[:3,3]+axis*(q-original[qidx])+outward*.015; rotation=eef_rotation
                        else:
                            R=axis_angle_to_matrix(axis*(q-original[qidx]))
                            target=origin+R@(handle[:3,3]+outward*.015-origin);rotation=R@eef_rotation
                        rec=solve(target,rotation);route_rows.append(rec)
                        if not rec["passed"]:break
                row["routes"][route]={"passed":all(r["passed"] for r in route_rows),"knots":route_rows}
            row["hard_valid"]=all(r["passed"] for r in row["routes"].values())
        candidates.append(row);write(directory/f"pose-{index}-validity.json",row)
        print(task,row["candidate_id"],"source",source_receipt["passed"],"full",row["hard_valid"],flush=True)
    eligible=[r for r in candidates if r["hard_valid"]]
    eligible.sort(key=lambda r:(-r["source_ik"]["nearest"][0]["signed_clearance_m"],-r["source_ik"]["joint_margin_rad"],r["candidate_id"]))
    decision={"completed_at":datetime.now(timezone.utc).isoformat(),"selected":eligible[0]["candidate_id"] if eligible else None,
              "candidate_count":5,"full_path_hard_valid_count":len(eligible),"outcome_reads":0,
              "status":"primary_frozen" if eligible else "bounded_geometry_feasibility_hold"}
    write(directory/"decision.json",decision)
    return decision
