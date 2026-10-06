"""Saved-state diagnostic at v2 protection; zero new env.step or outcome."""
import argparse
import json
from pathlib import Path
from io import BytesIO
import h5py
import numpy as np
from teleop_reference import Reference,write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.pi05_adapter import world_intent
from mobiwam.pi05_motion import whole_body_action,QPProtectionStop,observed_articulation_coupling


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--attempt',type=int,default=0);p.add_argument('--actuated-grip',action='store_true');p.add_argument('--coupled-grip',action='store_true');p.add_argument('--native-tracking',action='store_true');a=p.parse_args()
    receipt=next((a.run/'episodes/policy-dev-step-2500-adapter-v2').glob('slot-01*/engineering-attempt-0/completed.json'))
    q=json.loads(receipt.read_text());attempt=Path(q['attempt']);source=attempt.parents[1];out=a.run/'preflight'/('v2-failure-coupling-diagnostic' if a.attempt==0 else f'v2-failure-grip-diagnostic-{a.attempt}');out.mkdir(exist_ok=False)
    ref=Reference(argparse.Namespace(output=str(out),task='CloseDrawer',layout=1,style=0,seed=109,self_test=True,source=str(source),replay_attempt=None,resume_attempt=None,width=640,height=360))
    try:
        restore_saved_integration(ref)
        with h5py.File(attempt/'demo.hdf5') as f:ref.env.sim.set_state_from_flattened(f['data/demo_0/states'][-1])
        ref.env.sim.forward();ref.robot.composite_controller.update_state();ref.robot.part_controllers['right'].update(force=True)
        def forbidden(*args,**kwargs):raise AssertionError('saved state diagnostic must not step')
        ref.env.step=forbidden
        index=q['steps']%5;latest=sorted(receipt.parent.glob('query-*.npz'))[-1]
        with np.load(latest,allow_pickle=False) as z:
            intent=world_intent(z['actions'][index],dict(base_world_p=z['base_world_p'],base_world_R=z['base_world_R']))
        _,d=ref.model_data();base=ref.robot.part_controllers['base'];arm=ref.robot.part_controllers['right'];dofs=np.r_[base.qvel_index,arm.qvel_index]
        coupling=observed_articulation_coupling(ref,dofs);result=dict(record_type='saved_state_diagnostic',new_env_step_calls=0,new_task_outcomes=0,parent_receipt=str(receipt),bilateral_contact_coupling=bool(coupling),comparisons={})
        for enabled in (False,True):
            try:
                action,proof,v=whole_body_action(ref,intent,d.qpos[base.qpos_index].copy(),locked_base=True,co_motion=enabled)
                result['comparisons'][str(enabled)]=dict(status='feasible',proof=proof,action=action.tolist())
            except QPProtectionStop as exc:result['comparisons'][str(enabled)]=dict(status='protected',reason=str(exc))
        if a.actuated_grip:
            physical_before=d.qpos.copy()
            try:
                action,proof,v=whole_body_action(ref,intent,d.qpos[base.qpos_index].copy(),locked_base=True,co_motion=True,actuated_grip=True,coupled_grip=a.coupled_grip,native_tracking=a.native_tracking)
                result['actuated_grip']=dict(status='feasible',proof=proof,action=action.tolist(),physical_qpos_exactly_unchanged=bool(np.array_equal(physical_before,d.qpos)))
                if a.coupled_grip:
                    goals=np.asarray(proof['actuated_gripper']['target_qpos']);assert abs(goals.sum())<1e-12
                    result['actuated_grip']['native_coupled_goal_verified']=True
                if a.native_tracking:
                    from mobiwam.pi05_grip_projection import native_coupled_closure
                    result['native_closure']=native_coupled_closure(ref,intent['grasp'])
            except QPProtectionStop as exc:result['actuated_grip']=dict(status='protected',reason=str(exc),physical_qpos_exactly_unchanged=bool(np.array_equal(physical_before,d.qpos)))
        from mobiwam.pi05_grip_projection import protect_grip
        m,d=ref.model_data();q_before=d.qpos.copy();original=ref.robot.gripper['right'].current_action.copy()
        command,hold,proof=protect_grip(ref,1.)
        result['grip_projection']=dict(command=command,body_hold=hold,proof=proof,physical_qpos_exactly_unchanged=bool(np.array_equal(q_before,d.qpos)),old_controller_target=original.tolist(),new_controller_target=ref.robot.gripper['right'].current_action.tolist())
        write_json(out/'diagnostic.json',result);print(json.dumps(result),flush=True)
    finally:ref.env.close()


if __name__=='__main__':main()
