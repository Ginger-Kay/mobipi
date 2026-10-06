"""Task-owned zero-outcome motion planner integration on one dev Source."""
import argparse
import hashlib
import json
from pathlib import Path
from teleop_reference import Reference,write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.pi05_motion import docks


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--slot',type=int,default=1);a=p.parse_args()
    slot=json.loads((a.run/'policy/policy-dev-roster.json').read_text())['slots'][a.slot-1]
    out=a.run/'preflight'/f'motion-slot-{a.slot:02d}';out.mkdir(exist_ok=False)
    ref=Reference(argparse.Namespace(output=str(out),task=slot['task'],layout=1,style=0,seed=slot['environment_seed'],self_test=True,source=slot['source'],replay_attempt=None,resume_attempt=None,width=640,height=360))
    try:
        restore_saved_integration(ref)
        def prohibited(*args,**kwargs):raise AssertionError('zero-outcome preflight env.step forbidden')
        ref.env.step=prohibited
        result=docks(ref);write_json(out/'plans.json',result)
        print(json.dumps(dict(slot=a.slot,valid=sum(x['hard_valid'] for x in result['candidates']),primary=result['primary']['candidate_id'] if result['primary'] else None,env_step_calls=0)),flush=True)
    finally:ref.env.close()


if __name__=='__main__':main()
