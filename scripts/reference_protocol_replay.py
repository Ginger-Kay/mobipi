"""Replay development actions including recorded dock controller-reset events."""
import argparse
import json
from pathlib import Path
from teleop_reference import Reference, write_json, stamp
import numpy as np
import h5py
from mobiwam.replay_diagnostics import state_fields, summarize_drift


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--output',required=True)
    p.add_argument('--state-only',action='store_true',help='Verify dynamics without generating another video')
    args=p.parse_args()
    attempt=Path(args.attempt).resolve()
    result=json.loads((attempt/'result.json').read_text())
    events=result['events']
    if any(e['event']!='dock_settled_reobserve_feedback_reset' for e in events):
        raise ValueError('Unknown control event; do not silently omit it')
    resets={e['step'] for e in events}
    reset_payloads={e['step']:e for e in events}
    ref=Reference(argparse.Namespace(output=args.output,task='CloseDrawer',layout=0,style=0,seed=7,self_test=True,source=None,replay_attempt=str(attempt),resume_attempt=None,width=1920,height=1080))
    original_step=ref.env.step
    count=0;applied=[]
    def step(action):
        nonlocal count
        if count in resets:
            ref.env._get_observations(force_update=True)
            ref.robot.composite_controller.update_state()
            ref.robot.part_controllers['right'].set_goal_update_mode('achieved')
            ref.robot.part_controllers['right'].set_goal(np.zeros(6))
            if 'arm_nullspace_goal' in reset_payloads[count]:
                goal=np.asarray(reset_payloads[count]['arm_nullspace_goal'],float)
                if goal.shape!=(7,) or not np.isfinite(goal).all():raise ValueError('Invalid nullspace event')
                ref.robot.part_controllers['right'].initial_joint=goal.copy()
            applied.append(count)
        count+=1
        return original_step(action)
    ref.env.step=step
    try:
        if args.state_only:
            ref.restore()
            with h5py.File(attempt/'demo.hdf5') as h:
                actions=h['data/demo_0/actions'][:];states=h['data/demo_0/states'][:]
            if len(states)!=len(actions)+1 or not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError('invalid replay state/action alignment')
            initial_error=float(np.max(abs(ref.env.sim.get_state().flatten()-states[0])))
            if initial_error>1e-10:raise ValueError('replay Source differs from recorded initial state')
            errors=[]
            replayed=[ref.env.sim.get_state().flatten().copy()]
            for i,action in enumerate(actions):
                ref.env.step(action)
                actual=ref.env.sim.get_state().flatten().copy()
                replayed.append(actual)
                errors.append(float(np.max(abs(actual-states[i+1]))))
            model,_=ref.model_data()
            np.savez_compressed(Path(args.output)/'replayed-states.npz',states=np.asarray(replayed))
            write_json(Path(args.output)/'field-errors.json',
                       summarize_drift(states,np.asarray(replayed),state_fields(model)))
            write_json(Path(args.output)/'state-errors.json',dict(initial_error=initial_error,state_errors=errors))
        else:errors=ref.replay()
        if set(applied)!=resets:raise AssertionError('Not all recorded events applied')
        write_json(Path(args.output)/'completed.json',dict(ended_at=stamp(),attempt=str(attempt),steps=count,applied_reset_steps=applied,max_state_abs_error=max(errors),checker_success=bool(ref.env._check_success()),video_generated=not args.state_only,scope='protocol-aware action replay; no trajectory correction or state injection'))
    finally:
        ref.env.step=original_step
        if ref.observation_renderer is not None:ref.observation_renderer.close()
        if ref.renderer is not None:ref.renderer.close()
        ref.env.close()

if __name__=='__main__':main()
