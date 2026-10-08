"""Finite original-arm/gripper drawer starts; no policy or env.step screening."""
import argparse, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import mujoco
import numpy as np
from teleop_reference import Reference, write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.adapters.mobipi import _capture_controller_state
from mobiwam.pi05_motion import arm_indices, docks, collaborative_paths
from mobiwam.reference_collision import SweptGeometry
from mobiwam.task_video_identity import observe_native
from pi05_candidate_features import feature_record

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
NAMES=('model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json')
OFFSETS=((.02,0,0),(-.02,0,0),(0,.02,0),(0,-.02,0),(0,0,np.pi/180),(0,0,-np.pi/180))

def bind(r,g,ident,offset=None):
    out=r/'design/anchors'/ident;out.mkdir(parents=True)
    src=Path(g['source']);dest=out/src.name;dest.mkdir()
    for n in NAMES: shutil.copy2(src/n,dest/n)
    shutil.copy2(src.parent/'env_config.json',out/'env_config.json')
    ref=Reference(argparse.Namespace(output=str(out/'native'),task='CloseDrawer',layout=1,style=0,seed=g['environment_seed'],self_test=True,source=str(dest),replay_attempt=None,resume_attempt=None,width=640,height=360))
    try:
        restore_saved_integration(ref)
        def no_step(*args,**kwargs): raise AssertionError('static qualification must not advance physics')
        ref.env.step=no_step
        m,d=ref.model_data();base=ref.robot.part_controllers['base'];bids=np.asarray(base.qpos_index);bdofs=np.asarray(base.qvel_index)
        anchor=d.qpos.copy();velocity=d.qvel.copy();saved=ref.integration().copy();fixture=ref.env.drawer
        check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005)
        contact=[]
        for ct in d.contact[:d.ncon]:
            ns=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(i)) or '' for i in (ct.geom1,ct.geom2)]
            if any(fixture.name in n for n in ns) and any('gripper' in n or 'finger' in n for n in ns): contact.append(ns)
        reasons=[]; mapping=None
        if offset is not None:
            if contact: reasons.append('original contact initialization; base perturbation prohibited')
            else:
                bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ref.base_body)
                jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid)
                xy=jp[:2,bdofs[:2]].copy(); R=d.xmat[bid].reshape(3,3).copy(); world0=d.xpos[bid].copy()
                worlddelta=R@np.array([offset[0],offset[1],0.]);yaw_response=float(jr[2,bdofs[2]])
                assert abs(np.linalg.det(xy))>1e-6 and abs(yaw_response)>1e-6
                d.qpos[bids[:2]]+=np.linalg.solve(xy,worlddelta[:2]);d.qpos[bids[2]]+=offset[2]/yaw_response
                mujoco.mj_forward(m,d)
                assert np.max(abs(d.qpos[np.setdiff1d(np.arange(m.nq),bids)]-anchor[np.setdiff1d(np.arange(m.nq),bids)]))<=1e-6
                assert np.max(abs(d.qvel-velocity))<=1e-6
                assert np.max(abs((d.xpos[bid]-world0)-worlddelta))<=1e-6
                mapping=dict(base_body=ref.base_body,qpos_indices=bids.tolist(),qvel_indices=bdofs.tolist(),world_xy_jacobian=xy.tolist(),base_world_rotation=R.tolist(),yaw_response=yaw_response,offset_frame='original native base body local xy',requested_local=list(offset),realized_world_delta=(d.xpos[bid]-world0).tolist())
                np.save(dest/'integration.npy',ref.integration())
                info=json.loads((dest/'source.json').read_text());info.update(created_at=now(),parent_source=str(src),parent_group=g['parent_group'],config_id=ident,initializer='native static base-only local-plane perturbation; arm/gripper/fixture/qvel preserved; no physics step',perturbation=mapping)
                write_json(dest/'source.json',info)
                expected=d.qpos.copy();restore_saved_integration(ref);m,d=ref.model_data()
                assert np.max(abs(d.qpos-expected))<=1e-6 and np.max(abs(d.qvel-velocity))<=1e-6
                check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005)
        qids,_,limits=arm_indices(ref);margin=float(np.min(np.minimum(d.qpos[qids]-limits[:,0],limits[:,1]-d.qpos[qids])))
        _,distances=check.distances(d.qpos,'manipulate');clearance=float(np.min(distances,initial=.1))
        if margin<=.015: reasons.append('initial joint margin <=.015rad')
        if clearance<.0005: reasons.append('initial forbidden clearance <.5mm')
        if ref.env._check_success(): reasons.append('initial native task already successful')
        target=json.loads((dest/'target-binding.json').read_text());native=observe_native(ref,dict(task='CloseDrawer',fixture_name=target['fixture_name'],fixture_class=target['fixture_class'],model_sha256=sha(dest/'model.xml')))
        fields=dict(at=now(),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),controller=_capture_controller_state(ref.env),restore=ref.restore_receipt,env_step_calls=0,arm_gripper_fixture_preserved=True)
        write_json(out/'initial-state-controller.json',fields)
        plans={};routes={}
        if not reasons:
            before=ref.integration().copy()
            for route,fn in [('D',docks),('A',collaborative_paths)]:
                try: plans[route]=fn(ref)
                except ValueError as exc: plans[route]=dict(primary=None,reason=str(exc),candidates=[])
            for route in 'EDA':
                f=feature_record(ref,route,plans.get(route),0)
                routes[route]=bool(f is not None and f['hard_valid'])
            assert np.max(abs(ref.integration()-before))<=1e-6
        else: routes={route:False for route in 'EDA'}
        row=dict(g,config_id=ident,parent_config_id=g['config_id'],source=str(dest),original_source=str(src),slot=1,tier=1,status='X' if reasons else 'bound',rejection_reasons=reasons,contact_initialized=bool(contact),original_target_contacts=contact,offset=list(offset) if offset is not None else [0,0,0],mapping=mapping,native=native,joint_margin_rad=margin,initial_forbidden_clearance_m=clearance,hard_valid_routes=routes,env_step_calls=0,policy_forward_calls=0,source_sha256={n:sha(dest/n) for n in NAMES})
        write_json(out/'binding.json',dict(at=now(),**row));print(json.dumps(dict(config=ident,status=row['status'],hard_valid=routes)),flush=True)
        return row
    finally:
        if getattr(ref,'pi05_renderer',None): ref.pi05_renderer.close()
        ref.env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--old-run',type=Path,required=True);a=p.parse_args();r=a.run
    split=json.loads((a.old_run/'inventory/source-split.json').read_text());groups=[g for g in split['parent_groups'] if g['task']=='CloseDrawer']
    assert [sum(g['role']==role for g in groups) for role in ('train','development','evaluation')]==[6,2,2]
    assert not set(g['parent_group'] for g in groups if g['role']!='train')&set(split['old_fit2_policy_fit_ancestors'])
    write_json(r/'inventory/source-split.json',dict(split,at=now(),parent_groups=groups,source_inventory=str(a.old_run/'inventory/source-split.json'),counts={'train':6,'development':2,'evaluation':2}))
    qualified=json.loads((a.old_run/'design/start-design.json').read_text())['selected'];byid={g['config_id']:g for g in qualified}
    selected=[];candidates=[]
    for g in groups:
        original=dict(g,source=byid[g['config_id']]['source']);s0=bind(r,original,g['config_id']);s0.update(start_index=0,start_type='original');selected.append(s0)
        if g['role']=='evaluation':
            accepted=[]
            for i,offset in enumerate(OFFSETS,1):
                row=bind(r,original,g['config_id']+f'-candidate-{i:02d}',offset)
                row['candidate_order']=i;row['selected']=row['status']=='bound' and len(accepted)<2
                if row['selected']:
                    row.update(start_index=len(accepted)+1,start_type='perturbed');accepted.append(row);selected.append(row)
                candidates.append(row)
            for si in range(len(accepted)+1,3):
                selected.append(dict(g,config_id=g['config_id']+f'-unavailable-s{si}',source=None,start_index=si,start_type='perturbed',status='X',rejection_reasons=['no remaining finite static-valid candidate'],hard_valid_routes={t:False for t in 'EDA'},slot=1,tier=1))
    for i,g in enumerate(selected,1): g['slot']=i
    write_json(r/'design/start-design.json',dict(at=now(),selected=selected,candidates=candidates,finite_candidate_order=[list(x) for x in OFFSETS],zero_env_step=True,zero_policy_forward=True,field_tolerance=1e-6,original_arm_gripper_fixture_preserved=True))

if __name__=='__main__': main()
