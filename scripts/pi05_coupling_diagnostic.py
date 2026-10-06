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
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    receipt=next((a.run/'episodes/policy-dev-step-2500-adapter-v2').glob('slot-01*/engineering-attempt-0/completed.json'))
    q=json.loads(receipt.read_text());attempt=Path(q['attempt']);source=attempt.parents[1];out=a.run/'preflight/v2-failure-coupling-diagnostic';out.mkdir(exist_ok=False)
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
        write_json(out/'diagnostic.json',result);print(json.dumps(result),flush=True)
    finally:ref.env.close()


if __name__=='__main__':main()
