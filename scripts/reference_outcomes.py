"""Audit development feedback runs without promoting them to formal OBC data."""
import argparse
import datetime
import json
from pathlib import Path
import h5py
import numpy as np


def audit_attempt(p):
    result=json.loads((p/'result.json').read_text())
    trace=[json.loads(x) for x in (p/'trace.jsonl').read_text().splitlines()]
    feedback=[json.loads(x) for x in (p/'feedback.jsonl').read_text().splitlines()]
    source=json.loads((p.parent.parent/'source.json').read_text())['trace']
    before=trace[0]['before'];end=trace[-1]['after'];opening=float(before['target']['door'])
    base_step=np.array([np.linalg.norm(np.array(t['after']['base_pos'])[:2]-np.array(t['before']['base_pos'])[:2]) for t in trace])
    arm_step=np.array([np.linalg.norm(np.array(t['after']['arm_qpos'])-np.array(t['before']['arm_qpos'])) for t in trace])
    contact=np.array([any('gripper0_' in str(c) and any(s in str(c) for s in ['microwave_main_group','stack_4_main_group_2']) for c in t['after']['contacts']) for t in trace])
    closing=np.array([t['after']['target']['door']<t['before']['target']['door']-1e-5 for t in trace])
    manipulation=contact & closing
    nav=[i for i,x in enumerate(feedback) if x['stage']=='navigate']
    nav_change=max([abs(trace[i]['after']['target']['door']-opening) for i in nav],default=0.)
    with h5py.File(p/'demo.hdf5') as h:
        g=h['data/demo_0'];shapes={name:list(g[name].shape) for name in ['actions','states']}
        images={name:list(v.shape) for name,v in g['obs'].items() if 'image' in name}
        finite=bool(np.isfinite(g['actions'][:]).all() and np.isfinite(g['states'][:]).all())
        alignment=shapes['actions'][0]==len(trace)==len(feedback) and shapes['states'][0]==len(trace)+1 and len(images)==3 and all(s==[len(trace)+1,256,256,3] for s in images.values())
    replays=[]
    for f in p.glob('replay-*/result.json'):
        rr=json.loads(f.read_text());replays.append({k:rr[k] for k in ['steps','checker_success','max_state_abs_error']})
    stop_path=p/'contact-stop.json'
    stopped=json.loads(stop_path.read_text()) if stop_path.exists() else None
    nonfinger=False
    if stopped:
        for contact in stopped['contacts']:
            names=[contact.get('geom1') or '',contact.get('geom2') or '']
            robot=[n for n in names if n.startswith(('robot0_','gripper0_','mobilebase0_'))]
            if len(robot)==1 and 'finger' not in robot[0]:nonfinger=True
    predock=bool(stopped and p.parent.name=='D' and feedback[stopped['step']]['stage']!='manipulate')
    return dict(path=str(p.resolve()),route=p.parent.name,steps=len(trace),checker_success=result['checker_success'],reason=result['reason'],
        opening_start=opening,opening_end=end['target']['door'],
        outcomes=dict(success=float(result['checker_success']),progress=float(np.clip((opening-end['target']['door'])/max(opening,1e-6),0,1)),base_path_m=float(sum(base_step)),elapsed_sim_s=len(trace)*.05,observed_nonfinger_contact_stop=nonfinger,observed_pre_dock_contact_stop=predock,observed_environment_contact_stop=bool(stopped)),
        label_masks=dict(success=True,progress=True,base_path=True,completion_time=bool(result['checker_success']),irreversible_collision=False),
        note='Collision/irreversibility definition not closed. Failed duration is elapsed time, not a completed-task time label. Outcomes only; feature association and planning validity require separate verification, not a train-ready batch.',
        contact_closing_steps=int(sum(manipulation)),actual_base_arm_overlap_during_contact_closing=float(np.mean((base_step[manipulation]>1e-5)&(arm_step[manipulation]>1e-4))) if any(manipulation) else None,
        max_base_translation_m=float(max(np.linalg.norm(np.array(t['after']['base_pos'])[:2]-np.array(before['base_pos'])[:2]) for t in trace)),
        navigation_target_opening_change=nav_change,D_navigation_manipulated_target=bool(nav_change>1e-3) if p.parent.name=='D' else None,
        source_qpos_max_error=float(np.max(abs(np.array(before['qpos'])-source['qpos']))),source_qvel_max_error=float(np.max(abs(np.array(before['qvel'])-source['qvel']))),
        shapes=shapes,images=images,finite=finite,alignment=alignment,replays=replays,formal_train_ready=False,strict_route_qualification='pending_or_invalid_see_D_navigation')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',action='append',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    rows=[]
    for root in args.root:
        for f in sorted(Path(root).glob('*/run/source-*/*/attempt-*/result.json')):
            rows.append(audit_attempt(f.parent))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='development_only',attempts=rows),indent=2))
    for r in rows:print(r['route'],r['steps'],r['checker_success'],r['reason'],round(r['opening_end'],4),'D_nav_invalid',r['D_navigation_manipulated_target'],r['path'])

if __name__=='__main__':main()
