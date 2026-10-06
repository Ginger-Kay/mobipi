"""Compare the actual fixed paired table and separately executed online units."""
import argparse,csv,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

def outcome(receipt):
    if receipt is None:return dict(status='unrun',native_success=None,safety_success=None,collision=None,progress=None,path_m=None,duration_s=None,receipt=None)
    q=json.loads(receipt.read_text());attempt=Path(q['attempt']);audit=attempt/'sprint-safety-audit.json';safe=json.loads(audit.read_text()) if audit.exists() else {}
    usable=q['usable_scientific_outcome'] and not q['status'].startswith('X_')
    success=(safe.get('safety_qualified_success') if q['native_success'] else False) if usable else None
    z=np.load(attempt/'formal-native-substeps.npz',allow_pickle=False);trace=attempt/'trace.jsonl';progress=None
    if q['steps'] and trace.exists():
        rows=trace.read_text().splitlines();first=json.loads(rows[0]);last=json.loads(rows[-1]);initial=float(first['before']['target']['door']);terminal=float(last['after']['target']['door'])
        if initial>1e-6:progress=float(np.clip((initial-terminal)/initial,0,1))
    elif len(z['phases'])==0:progress=0.
    contact=json.loads((attempt/'formal-native-substeps-receipt.json').read_text())
    return dict(status=q['status'],native_success=q['native_success'] if usable else None,safety_success=success,collision=(contact.get('forbidden_contact') is not None) if usable else None,progress=progress,
        path_m=float(np.linalg.norm(np.diff(z['qpos'][:,:2],axis=0),axis=1).sum()),duration_s=float(z['sim_time'][-1]-z['sim_time'][0]),receipt=str(receipt))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
    freeze=json.loads((r/'evaluation/final-freeze.json').read_text());out=r/'paper-evidence';rows=[]
    def locate(tag,slot):return next((r/'episodes'/tag).glob(f'slot-{slot:02d}*/engineering-attempt-0/completed.json'),None)
    for g in freeze['predictions']:
        table={route:outcome(locate('paired-v6-final-'+route,g['slot'])) for route in 'EDA'}
        methods={**{'fixed'+route:route for route in 'EDA'},**g['selected']}
        known=[route for route in 'EDA' if table[route]['status']!='unrun' and table[route]['safety_success'] is not None]
        oracle=min(known,key=lambda route:(-int(table[route]['safety_success']),int(bool(table[route]['collision'])),-(table[route]['progress'] or 0),
            table[route]['path_m'],table[route]['duration_s'],'EDA'.index(route))) if known else 'X'
        methods['oracle']=oracle
        for method,route in methods.items():
            q=table[route] if route!='X' else dict(status='X_no_legal_candidate',native_success=None,safety_success=None,collision=None,progress=None,path_m=None,duration_s=None,receipt=None)
            rows.append(dict(evaluation='paired_lookup',source=g['config_id'],parent_group=g['parent_group'],family_id=g['family_id'],method=method,route=route,**q))
        for method in freeze['online_methods']:
            q=outcome(locate('online-v6-'+method,g['slot']))
            if g['selected'][method]=='X':q['status']='X_no_legal_candidate'
            rows.append(dict(evaluation='actual_online',source=g['config_id'],parent_group=g['parent_group'],family_id=g['family_id'],method=method,route=g['selected'][method],**q))
    with (out/'final-method-outcomes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aggregates=[]
    for evaluation in ('paired_lookup','actual_online'):
        for method in sorted({x['method'] for x in rows if x['evaluation']==evaluation}):
            selected=[x for x in rows if x['evaluation']==evaluation and x['method']==method]
            successes=[x for x in selected if x['safety_success'] is True]
            aggregates.append(dict(evaluation=evaluation,method=method,denominator=len(selected),executed=sum(x['receipt'] is not None for x in selected),
                safe_success=sum(x['safety_success'] is True for x in selected),native_success=sum(x['native_success'] is True for x in selected),
                unknown_safe_success=sum(x['safety_success'] is None for x in selected),unrun=sum(x['status']=='unrun' for x in selected),
                X=sum(x['status'].startswith('X_') for x in selected),collision=sum(x['collision'] is True for x in selected),
                protective_stop=sum('protective_stop' in x['status'] or 'joint_margin' in x['status'] or 'native_forbidden' in x['status'] for x in selected),
                budget_stop=sum('budget_stop' in x['status'] or 'timeout' in x['status'] for x in selected),
                safe_success_lower=sum(x['safety_success'] is True for x in selected),
                safe_success_upper=sum(x['safety_success'] is True or (x['safety_success'] is None and not x['status'].startswith('X_')) for x in selected),
                all_executed_path_sum_m=float(sum(x['path_m'] for x in selected if x['path_m'] is not None)),
                all_executed_terminal_duration_sum_s=float(sum(x['duration_s'] for x in selected if x['duration_s'] is not None)),
                failed_terminal_duration_mean_s=float(np.mean([x['duration_s'] for x in selected if x['safety_success'] is False])) if any(x['safety_success'] is False for x in selected) else None,
                successful_path_mean_m=float(np.mean([x['path_m'] for x in successes])) if successes else None,
                successful_duration_mean_s=float(np.mean([x['duration_s'] for x in successes])) if successes else None))
    with (out/'final-method-summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregates[0]));w.writeheader();w.writerows(aggregates)
    common=[]
    for evaluation,baseline in [('paired_lookup','fixedE'),('actual_online','train-best-fixed')]:
        by_method={method:{x['source']:x for x in rows if x['evaluation']==evaluation and x['method']==method} for method in {x['method'] for x in rows if x['evaluation']==evaluation}}
        for method,selected in sorted(by_method.items()):
            if method==baseline:continue
            sources=[source for source,x in selected.items() if x['safety_success'] is True and by_method[baseline][source]['safety_success'] is True]
            common.append(dict(evaluation=evaluation,method=method,baseline=baseline,common_safe_success_parents=len(sources),sources=';'.join(sources),
                path_delta_mean_m=float(np.mean([selected[s]['path_m']-by_method[baseline][s]['path_m'] for s in sources])) if sources else None,
                duration_delta_mean_s=float(np.mean([selected[s]['duration_s']-by_method[baseline][s]['duration_s'] for s in sources])) if sources else None))
    with (out/'common-success-cost-comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(common[0]));w.writeheader();w.writerows(common)
    (out/'final-comparison-status.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),final_parents=len(freeze['predictions']),families=len({x['family_id'] for x in freeze['predictions']}),
        scope='prospective held-out known-development, partial task/routes; descriptive sample, no significance or formal claim',all_failures_and_missing_retained=True,
        oracle_coverage='only executed routes with resolved safe outcomes',cost_comparison='successful means descriptive; no universal speed claim without common success set',
        methods=aggregates,formal_train_ready=False,human_review='pending'),indent=2)+'\n')
    print(json.dumps(aggregates),flush=True)
if __name__=='__main__':main()
