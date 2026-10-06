"""Qualify only the executed D stow/navigation/settle/fresh-query prefix."""
import argparse,hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from mobiwam.reference_collision import SweptGeometry
from mobiwam.task_video_identity import source_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    q=json.loads(a.receipt.read_text());assert q['route']=='D';sem=json.loads((a.receipt.parent/'route-semantics.json').read_text())
    assert sem['D_fresh_query_after_settle'] and q['policy_queries']>0 and q['steps']>sem['prefix_steps']
    attempt=Path(q['attempt']);out=attempt/'D-prefix-safety.json';assert not out.exists()
    source=attempt.parents[1];sha=hashlib.sha256((source/'model.xml').read_bytes()).hexdigest();m=source_model(str(source/'model.xml'),sha)
    binding=json.loads((source/'target-binding.json').read_text());z=np.load(attempt/'formal-native-substeps.npz',allow_pickle=False)
    n=int(np.sum(z['step_index']<sem['prefix_steps']));assert np.all(z['step_index'][:n]<sem['prefix_steps'])
    monitor=json.loads((attempt/'formal-native-substeps-receipt.json').read_text());margin=json.loads((attempt/'joint-margin-monitor.json').read_text())
    forbidden=monitor.get('forbidden_contact');prefix_contact=bool(forbidden and forbidden['step']<sem['prefix_steps'])
    t=time.monotonic();sweep=SweptGeometry(m,target_prefix=binding['fixture_name'],margin=.0005).path(z['qpos'][:n+1],z['phases'][:n])
    safe=bool(sweep['valid'] and not prefix_contact and margin['minimum'] and margin['minimum']['margin_rad']>.015)
    result=dict(at=datetime.now(timezone.utc).isoformat(),passed=safe,prefix_steps=sem['prefix_steps'],native_intervals=n,actual_continuous_prefix_sweep=sweep,
        forbidden_contact_in_prefix=prefix_contact,joint_monitor_whole_episode=margin,fresh_query_after_settle=True,actual_pi05_control_steps=q['steps']-sem['prefix_steps'],
        parent_receipt=str(a.receipt),new_rollouts=0,elapsed_seconds=time.monotonic()-t,
        scope='D prefix and actual fresh query/action wiring, not task success or full manipulation safety',formal_train_ready=False)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
