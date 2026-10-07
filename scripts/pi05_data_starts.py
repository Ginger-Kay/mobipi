"""Freeze six posed starts per parent using native zero-dynamics geometry only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from teleop_reference import Reference, write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.adapters.mobipi import _capture_controller_state
from mobiwam.reference_collision import SweptGeometry
from mobiwam.pi05_motion import arm_indices, frustum_compatibility

ARM = np.array([0., -.7853981633974483, 0., -2.356194490192345, 0., 1.5707963267948966, .7853981633974483])
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();root=a.run/'design'
    if (root/'start-design.json').exists():raise ValueError('existing start design is frozen')
    split=json.loads((a.run/'inventory/source-split.json').read_text())
    # Three fixed alternatives per slot, total18. Distances/angles are native
    # units; coordinates transform through measured furniture and base Jacobian.
    patterns=[[(0.,0.,0.,1.),(.04,0.,0.,1.),(.08,0.,0.,1.)],
              [(0.,.08,0.,1.),(.04,.08,0.,1.),(.08,.08,0.,1.)],
              [(.12,0.,0.,1.),(.16,0.,0.,1.),(.20,0.,0.,1.)],
              [(0.,-.08,0.,1.),(.04,-.08,0.,1.),(.08,-.08,0.,1.)],
              [(.24,0.,0.,1.),(.28,0.,0.,1.),(.32,0.,0.,1.)],
              [(0.,0.,0.,.8),(.04,0.,0.,.8),(.08,0.,0.,.8)]]
    # Axis1 is measured radial direction from furniture to anchor base; axis2
    # perpendicular within furniture XY. No arbitrary world-axis guess.
    records=[];candidates=[];coverage=[]
    parents=split['parent_groups']
    for parent_index,g in enumerate(parents):
        po=root/'parents'/g['config_id'];po.mkdir(parents=True,exist_ok=True)
        src=Path(g['source']);ref=Reference(argparse.Namespace(output=str(po/'native'),task=g['task'],layout=1,style=0,seed=g['environment_seed'],self_test=True,source=str(src),replay_attempt=None,resume_attempt=None,width=640,height=360))
        try:
            restore_saved_integration(ref)
            def prohibited(*args,**kwargs):raise AssertionError('start design must not call env.step')
            ref.env.step=prohibited
            m,d=ref.model_data();base=ref.robot.part_controllers['base'];arm=ref.robot.part_controllers['right']
            qids,ad,limits=arm_indices(ref);bids=np.asarray(base.qpos_index);bdofs=np.asarray(base.qvel_index)
            fixture=ref.env.drawer if g['task']=='CloseDrawer' else ref.env.door_fxtr
            fid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,fixture.name)
            if fid<0: fid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,fixture.name+'_main')
            # Source target binding and joint IDs establish actual native target.
            fj=[j for j in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or '').startswith(fixture.name) and int(m.jnt_type[j]) in (2,3)]
            assert len(fj)==1
            fqid=int(m.jnt_qposadr[fj[0]])
            if fid<0:fid=int(m.jnt_bodyid[fj[0]])
            furniture_p=d.xpos[fid].copy();furniture_R=d.xmat[fid].reshape(3,3).copy()
            robotbid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ref.base_body)
            anchor=ref.integration().copy();anchor_q=d.qpos.copy();anchor_base=d.xpos[robotbid].copy()
            local=furniture_R.T@(anchor_base-furniture_p);radial=local[:2]/max(np.linalg.norm(local[:2]),1e-9)
            lateral=np.array([-radial[1],radial[0]])
            jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,robotbid)
            xy=jp[:2,bdofs[:2]].copy();assert abs(np.linalg.det(xy))>1e-6
            check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005)
            denominator=float(fixture.size[1]*.55) if g['task']=='CloseDrawer' else float(np.pi/2)
            initial_opening=abs(float(anchor_q[fqid]))/denominator
            selected_states=[]
            for slot,alternatives in enumerate(patterns,1):
                selected=None
                for alternative,(distance,side,yaw,opening_factor) in enumerate(alternatives,1):
                    ident=f'{g["config_id"]}-candidate-{3*(slot-1)+alternative:02d}'
                    mujoco.mj_setState(m,d,anchor,ref.kind)
                    d.qpos[qids]=ARM
                    # Release any target grasp before moving a posed base.
                    gripids=np.asarray(ref.robot._ref_gripper_joint_pos_indexes['right'])
                    d.qpos[gripids]=np.array([.03,-.03])
                    delta_local=np.r_[distance*radial+side*lateral,0.]
                    world_delta=furniture_R@delta_local
                    d.qpos[bids[:2]]=anchor_q[bids[:2]]+np.linalg.solve(xy,world_delta[:2])
                    d.qpos[bids[2]]=anchor_q[bids[2]]+yaw
                    d.qpos[fqid]=anchor_q[fqid]*opening_factor
                    d.qvel[:]=0.;d.qacc_warmstart[:]=0.;mujoco.mj_forward(m,d)
                    reason=[];margin=float(np.min(np.minimum(d.qpos[qids]-limits[:,0],limits[:,1]-d.qpos[qids])))
                    if margin<=.015:reason.append('initial arm joint margin')
                    _,distances=check.distances(d.qpos,'manipulate');clearance=float(np.min(distances,initial=.1))
                    if clearance<.0005:reason.append('initial forbidden geometry clearance below original0.5mm')
                    if np.max(abs(d.qpos[gripids]-np.array([.03,-.03])))>1e-6:reason.append('gripper pose mismatch')
                    visibility=float(frustum_compatibility(m,d,furniture_p,ref.policy_cameras))
                    if visibility<=0:reason.append('target outside all policy camera frusta')
                    if float(d.xpos[robotbid,2])<-.001 or abs(float(d.xpos[robotbid,2]-anchor_base[2]))>1e-6:reason.append('base support height mismatch')
                    if ref.env._check_success():reason.append('already successful initial task')
                    for previous in selected_states:
                        if np.max(abs(previous-d.qpos))<1e-6:reason.append('duplicate qpos within1e-6')
                    row=dict(candidate_id=ident,parent_group=g['parent_group'],parent_config=g['config_id'],slot=slot,alternative=alternative,
                        distance_offset_m=distance,lateral_offset_m=side,yaw_offset_rad=yaw,opening_factor=opening_factor,
                        arm_qpos=ARM.tolist(),gripper_qpos=[.03,-.03],initial_joint_margin_rad=margin,initial_clearance_m=clearance,
                        target_frustum_fraction=visibility,native_initial_opening=initial_opening*opening_factor,
                        base_world_xyz=d.xpos[robotbid].tolist(),base_furniture_local_xyz=(furniture_R.T@(d.xpos[robotbid]-furniture_p)).tolist(),
                        base_generalized=d.qpos[bids].tolist(),rejection_reasons=reason,selected=False)
                    # Evaluate all18 candidates, while selecting first static-valid
                    # per slot. E is statically eligible; future reach is unknown.
                    if not reason and selected is None:
                        selected=ident;row['selected']=True;selected_states.append(d.qpos.copy())
                        config=f'{g["config_id"]}-start-{slot:02d}';co=root/'configs'/config;co.mkdir(parents=True)
                        dest=co/src.name;dest.mkdir()
                        for name in ('model.xml','ep_meta.json','rng.json','target-binding.json'):shutil.copy2(src/name,dest/name)
                        shutil.copy2(src.parent/'env_config.json',co/'env_config.json')
                        candidate_state=np.empty(mujoco.mj_stateSize(m,ref.kind));mujoco.mj_getState(m,d,candidate_state,ref.kind)
                        np.save(dest/'integration.npy',candidate_state)
                        # Refresh controller goals/caches through native restore API.
                        sourceinfo=json.loads((src/'source.json').read_text());sourceinfo.update(created_at=now(),parent_source=str(src),parent_group=g['parent_group'],config_id=config,
                            initializer='native mjSTATE_INTEGRATION then fixed neutral arm/open gripper, furniture-frame base offset and native task joint; mj_forward only; zero task action',
                            scientific_distribution='posed-start; not natural random deployment',camera=sourceinfo['camera'])
                        write_json(dest/'source.json',sourceinfo)
                        original=ref.source;ref.source=dest;restore_saved_integration(ref);ref.source=original
                        m,d=ref.model_data()
                        assert np.max(abs(d.qpos-selected_states[-1]))<=1e-6
                        check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005)
                        write_json(co/'initial-state-controller.json',dict(at=now(),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),task_native_opening=initial_opening*opening_factor,
                            controller=_capture_controller_state(ref.env),integration_restore=ref.restore_receipt,env_step_calls=0,
                            source_sha256={n:sha(dest/n) for n in ['model.xml','integration.npy','rng.json','ep_meta.json','target-binding.json']}))
                        rec=dict(g,config_id=config,source=str(dest),slot=slot,tier=1 if slot<=3 else 2,design_candidate=ident,posed_start=True,
                            feature_binding='must encode actual config restored RGB; parent anchor features prohibited',row=row)
                        records.append(rec)
                        coverage.append(dict(task=g['task'],parent_group=g['parent_group'],config_id=config,family_id=g['family_id'],split=g['role'],slot=slot,tier=rec['tier'],
                            distance_offset_m=distance,lateral_offset_m=side,yaw_offset_rad=yaw,native_opening=initial_opening*opening_factor,initial_clearance_m=clearance,arm='one fixed neutral posture',
                            geometry_E_static=True,geometry_D='pending existing planner',geometry_A='pending existing planner'))
                    candidates.append(row)
                print(json.dumps(dict(at=now(),parent=g['config_id'],slot=slot,selected=selected,configs=len(records))),flush=True)
        finally:ref.env.close()
    design=dict(at=now(),status='all six slots frozen before policy fit or new outcome',parent_count=20,config_count=len(records),selected=records,candidates=candidates,
        generation={'max_candidates_per_parent':18,'patterns':patterns,'units':'distance/lateral meters, yaw radians, opening relative multiplicative factor; native opening uses original task denominator',
            'arm':ARM.tolist(),'arm_postures':1,'selection':'first zero-dynamics legal candidate within each coverage slot; no task outcome, no requirement all routes executable',
            'duplicate_parameter_tolerances':{'translation_m':1e-6,'yaw_rad':1e-6,'opening_fraction':1e-6,'state_native_units':1e-6},'initialization':'posed-start with released gripper; no dynamics settle'},
        zero_policy_forward=True,zero_env_step=True,config_shortfall=120-len(records),task_geometry_physics_unchanged=True)
    write_json(root/'start-design.json',design)
    with (root/'start-coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
    print(json.dumps(dict(at=now(),configs=len(records),candidate_count=len(candidates),shortfall=120-len(records))),flush=True)

if __name__=='__main__':main()
