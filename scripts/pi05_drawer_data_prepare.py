"""Lazy, zero-action start construction using the frozen STRESS transform."""
import argparse, json, shutil
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from pi05_drawer_stress_prepare import pose_transform, contacts, NAMES, write
from pi05_drawer_pipeline import read, now
from teleop_reference import Reference
from human_scene_pilot import restore_saved_integration
from mobiwam.adapters.mobipi import _capture_controller_state
from mobiwam.reference_collision import SweptGeometry
from mobiwam.pi05_motion import arm_indices, docks, collaborative_paths
from mobiwam.pi05_adapter import observation

def prepare(r, state):
    out=r/'design/anchors'/state['config_id']; binding=out/'binding.json'
    if binding.exists(): return read(binding)
    out.mkdir(parents=True, exist_ok=True)
    src=Path(state['original_source']);dest=out/src.name;dest.mkdir(exist_ok=True)
    for name in NAMES:shutil.copy2(src/name,dest/name)
    shutil.copy2(src.parent/'env_config.json',out/'env_config.json')
    ref=Reference(argparse.Namespace(output=str(out/'native'),task='CloseDrawer',layout=1,style=0,seed=state['environment_seed'],self_test=True,source=str(dest),replay_attempt=None,resume_attempt=None,width=640,height=360))
    try:
        restore_saved_integration(ref);m,d=ref.model_data()
        ref.env.step=lambda *a,**k: (_ for _ in ()).throw(AssertionError('preparation forbids dynamics'))
        old=read(src.parent/'initial-state-controller.json')
        assert np.max(abs(d.qpos-np.asarray(old['qpos'])))<=1e-6
        assert np.max(abs(d.qvel-np.asarray(old['qvel'])))<=1e-6
        base=ref.robot.part_controllers['base'];bids=np.asarray(base.qpos_index);bdofs=np.asarray(base.qvel_index)
        bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ref.base_body);fixture=ref.env.drawer
        target=read(dest/'target-binding.json');assert target['fixture_name']==fixture.name and target['fixture_class']=='Drawer'
        hid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,fixture.handle_name);assert hid>=0
        world=d.xpos[bid].copy();handle=d.geom_xpos[hid].copy();u=handle[:2]-world[:2];assert np.linalg.norm(u)>1e-6
        u/=np.linalg.norm(u);left=np.array([-u[1],u[0]]);q0=d.qpos.copy();vel=d.qvel.copy();before_contacts=contacts(m,d,fixture.name)
        delta=np.zeros(3);yaw=0.
        if state['axis']=='yaw':yaw=state['magnitude']
        else:delta[:2]=state['magnitude']*(u if state['axis']=='u' else left)
        mapping=pose_transform(m,d,bid,bids,bdofs,delta,yaw)
        other=np.setdiff1d(np.arange(m.nq),bids)
        assert np.max(abs(d.qpos[other]-q0[other]))<=1e-6 and np.max(abs(d.qvel-vel))<=1e-6
        mapping.update(handle_geom=fixture.handle_name,handle_center_world=handle.tolist(),u=u.tolist(),left=left.tolist(),nonbase_qpos_preserved=True,qvel_preserved=True)
        np.save(dest/'integration.npy',ref.integration());info=read(dest/'source.json')
        info.update(created_at=now(),parent_source=str(src),config_id=state['config_id'],initializer='base-only frozen world transform; no IK or dynamics',perturbation=mapping)
        write(dest/'source.json',info)
        qexpected=d.qpos.copy();restore_saved_integration(ref);m,d=ref.model_data()
        assert np.max(abs(d.qpos-qexpected))<=1e-6 and np.max(abs(d.qvel-vel))<=1e-6
        after_contacts=contacts(m,d,fixture.name);check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005)
        _,distances=check.distances(d.qpos,'manipulate');clearance=float(np.min(distances,initial=.1));qids,_,limits=arm_indices(ref)
        margin=float(np.min(np.minimum(d.qpos[qids]-limits[:,0],limits[:,1]-d.qpos[qids])))
        valid=all(pair in after_contacts for pair in before_contacts) and margin>.015 and clearance>=.0005 and not ref.env._check_success()
        write(out/'initial-state-controller.json',dict(at=now(),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),controller=_capture_controller_state(ref.env),restore=ref.restore_receipt,env_step_calls=0))
        # Refresh policy RGB and retain all native state inputs, without CLIP or fitting.
        inputs,anchor=observation(ref);np.savez_compressed(out/'initial-policy-RGB-state.npz',**inputs)
        camera=read(dest/'source.json')['camera'];Image.fromarray(ref.frame(camera).copy()).save(out/'initial-preview.jpg')
        routes={t:valid for t in 'EDA'}
        if valid:
            initial=ref.integration().copy()
            for route,fn in [('D',docks),('A',collaborative_paths)]:
                try:plan=fn(ref)
                except ValueError as exc:plan=dict(primary=None,candidates=[],reason=str(exc))
                write(out/(route+'-plan.json'),plan);routes[route]=plan.get('primary') is not None
            assert np.max(abs(ref.integration()-initial))<=1e-6
        row=dict(state,source=str(dest),status='bound' if valid else 'X',mapping=mapping,hard_valid_routes=routes,initial_state=str(out/'initial-state-controller.json'),initial_RGB=str(out/'initial-policy-RGB-state.npz'),env_step_calls=0,policy_forward_calls=0,initial_joint_margin_rad=margin,initial_forbidden_clearance_m=clearance)
        write(binding,row);return row
    finally:
        for name in ('renderer','observation_renderer','pi05_renderer'):
            renderer=getattr(ref,name,None)
            if renderer:renderer.close()
        ref.env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--state',type=int,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.run,read(a.run/'design/definitions-freeze.json')['states'][a.state-1])),flush=True)
