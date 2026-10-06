"""Aggregate actual receipts without synthesizing outcomes or touching inputs."""
import argparse,csv,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    now=datetime.now(timezone.utc).isoformat();paper=r/'paper-evidence';paper.mkdir(exist_ok=True)
    rows=[]
    for p in sorted((r/'episodes').glob('*/slot-*/engineering-attempt-*/completed.json')):
        q=json.loads(p.read_text());attempt=Path(q['attempt']);audit=attempt/'sprint-safety-audit.json'
        safety=json.loads(audit.read_text()) if audit.exists() else {}
        started=datetime.fromisoformat(q['started_at']);ended=datetime.fromisoformat(q['ended_at'])
        rows.append(dict(evaluation=p.parents[2].name,slot=q['slot'],task=q['task'],route=q['route'],checkpoint=q['checkpoint_step'],
            adapter=q.get('adapter_version','v1'),parent_group=q['parent_group'],config_id=q['config_id'],family_id=q['family_id'],
            status=q['status'],usable_outcome=q['usable_scientific_outcome'],native_success=q['native_success'],
            safety_qualified_success=safety.get('safety_qualified_success'),all_safety_pass=safety.get('all_safety_pass'),
            clearance_pass=safety.get('clearance_pass'),native_collision=safety.get('native_collision'),
            steps=q['steps'],sim_seconds=q['steps']*.05,wall_seconds=(ended-started).total_seconds(),
            queries=q['policy_queries'],query_seconds=q['query_seconds'],base_generalized_drift=q['base_drift_max_generalized'],
            receipt=str(p),attempt=str(attempt),main_video=str(attempt/'original.mp4') if (attempt/'original.mp4').exists() else '',
            panoramic_video=str(attempt/'panoramic.mp4') if (attempt/'panoramic.mp4').exists() else '',human_review='pending',
            component_code=json.loads((p.parent/'process.json').read_text())['code_commit']))
    if rows:
        with (paper/'policy-development-outcomes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    failures=[]
    for p in sorted((r/'episodes').glob('*/slot-*/engineering-attempt-*/failure.json')):
        q=json.loads(p.read_text());failures.append(dict(path=str(p),usable_executed_prefix=q.get('usable_outcome',False),steps=q.get('completed_control_steps',0),error=q['traceback'].splitlines()[-1]))
    summary=dict(at=now,completed_receipts=len(rows),valid_development=sum(x['usable_outcome'] for x in rows),
        native_success=sum(x['native_success'] for x in rows),safety_qualified_success=sum(x['safety_qualified_success'] is True for x in rows),
        pending_safety=sum(x['all_safety_pass'] is None for x in rows),statuses=dict(Counter(x['status'] for x in rows)),
        completed_control_steps=sum(x['steps'] for x in rows),completed_episode_wall_seconds=sum(x['wall_seconds'] for x in rows),
        query_seconds=sum(x['query_seconds'] for x in rows),engineering_failures=failures,paired_valid=0,online_valid=0,
        task_readiness='not established; each task requires2/3 safe success under identical frozen candidate',formal_train_ready=False)
    (paper/'progress-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    phase=json.loads((r/'phase-state.json').read_text());phase.update(updated_at=now,agent_last_active_at=now,phase='P1_native_capability_development',
        development_valid_episodes=summary['valid_development'],code_commits={k:subprocess.check_output(['git','-C',str(r/'runtime'/k),'rev-parse','HEAD'],text=True).strip() for k in ('control','mobipi','openpi')})
    for key,fit in [('fit1','20261006T154000Z-unified-dual-lora-fit1'),('relative-fit2','20261006T181000Z-query-relative-fit2')]:
        path=r/'policy'/fit;status=path/'status.json';result=path/'result.json'
        if status.exists():phase['jobs'].setdefault(key,{}).update(actual=json.loads(status.read_text()))
        if result.exists():phase['jobs'].setdefault(key,{}).update(status='completed',result=str(result.relative_to(r)),actual_result=json.loads(result.read_text()))
    phase['development_rounds']=4
    phase['next_action']='Finish declared absolute5000-v3 checks, verify actuated-gripper constraints, then predeclared relative500/1000/2000 development candidates; no final outcomes before readiness'
    phase['engineering_issues']['CONVERT-ASSET-001']['status']='resolved; all failed versions retained'
    phase['engineering_issues']['TRAIN-NNX-001']['status']='resolved; both50step diagnostics passed'
    phase['engineering_issues']['E-CONTROL-001'].update(status='v1/v2 failed; v3 and v4 mechanical constraint diagnosis ongoing',revision_round=4)
    phase['jobs']['relative-fit2']=dict(phase['jobs'].get('relative-fit2',{}),fit_id='20261006T181000Z-query-relative-fit2',command='policy/run-relative-fit2.sh',steps=2000,max_wall_seconds=3600,budget='policy/fit2-budget-freeze.json')
    (r/'phase-state.json').write_text(json.dumps(phase,indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':main()
