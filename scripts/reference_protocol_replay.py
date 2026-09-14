"""Replay development actions including recorded dock controller-reset events."""
import argparse
import json
from pathlib import Path
from teleop_reference import Reference, write_json, stamp
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    attempt=Path(args.attempt).resolve()
    result=json.loads((attempt/'result.json').read_text())
    events=result['events']
    if any(e['event']!='dock_settled_reobserve_feedback_reset' for e in events):
        raise ValueError('Unknown control event; do not silently omit it')
    resets={e['step'] for e in events}
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
            applied.append(count)
        count+=1
        return original_step(action)
    ref.env.step=step
    try:
        errors=ref.replay()
        if set(applied)!=resets:raise AssertionError('Not all recorded events applied')
        write_json(Path(args.output)/'completed.json',dict(ended_at=stamp(),attempt=str(attempt),steps=count,applied_reset_steps=applied,max_state_abs_error=max(errors),checker_success=bool(ref.env._check_success()),scope='protocol-aware action replay; no trajectory correction or state injection'))
    finally:
        ref.env.step=original_step
        if ref.observation_renderer is not None:ref.observation_renderer.close()
        if ref.renderer is not None:ref.renderer.close()
        ref.env.close()

if __name__=='__main__':main()
