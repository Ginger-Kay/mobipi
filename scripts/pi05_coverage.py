"""Keep the full predeclared roster, missing routes and late labels visible."""
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path

def write_csv(path,rows):
    if rows:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;out=r/'paper-evidence'
    roster=json.loads((r/'data/paired-source-roster.json').read_text());release=json.loads((r/'policy/harness-route-release.json').read_text())
    binding=json.loads((r/'training/dataset-binding.json').read_text()) if (r/'training/dataset-binding.json').exists() else {'records':[]}
    included={x['receipt'] for x in binding['records'] if x['receipt'] is not None};rows=[]
    for slot,g in enumerate(roster['slots'],1):
        for route in 'EDA':
            tag='paired-v6-final-'+route if g['role']=='final' else 'paired-v6-train-dev-'+route
            p=next((r/'episodes'/tag).glob(f'slot-{slot:02d}*/engineering-attempt-0/completed.json'),None)
            ready=release['tasks'].get(g['task'],{}).get(route,{}).get('released',False)
            q=json.loads(p.read_text()) if p else {}
            audit=Path(q['attempt'])/'sprint-safety-audit.json' if p else None
            safe=json.loads(audit.read_text()) if audit and audit.exists() else {}
            status=q.get('status','unrun_task_not_released' if g['task'] not in release['tasks'] else 'unrun_route_not_released' if not ready else 'released_queue_pending')
            rows.append(dict(slot=slot,task=g['task'],role=g['role'],route=route,parent_group=g['parent_group'],config_id=g['config_id'],family_id=g['family_id'],
                released=ready,status=status,native_success=q.get('native_success'),safety_qualified_success=safe.get('safety_qualified_success'),
                OBC_supervision_included=str(p) in included if p else False,late_train_dev_outcome=bool(p and g['role'] in ('train','dev') and str(p) not in included),
                receipt=str(p) if p else None,human_review='pending'))
    write_csv(out/'full-roster-route-coverage.csv',rows)
    harness=[]
    for p in sorted((r/'episodes').glob('harness-dev-v6-*/slot-*/engineering-attempt-0/completed.json')):
        q=json.loads(p.read_text());sem=json.loads((p.parent/'route-semantics.json').read_text());audit=Path(q['attempt'])/'sprint-safety-audit.json'
        safe=json.loads(audit.read_text()) if audit.exists() else {}
        harness.append(dict(evaluation=p.parents[2].name,slot=q['slot'],task=q['task'],route=q['route'],native_success=q['native_success'],status=q['status'],
            route_semantics_pass=q['route_semantics_pass'],safe_success=safe.get('safety_qualified_success'),all_safety_pass=safe.get('all_safety_pass'),
            real_queries=q['policy_queries'],D_fresh_query_after_settle=sem['D_fresh_query_after_settle'],
            A_max_native_overlap_seconds=sem.get('A_native_contact_overlap_maximum_seconds'),A_continuous_controls=sem.get('A_maximum_continuous_overlap_controls'),
            A_contact_translation_m=sem.get('A_contact_base_translation_m'),A_contact_yaw_rad=sem.get('A_contact_base_yaw_rad'),
            A_private_version=sem.get('A_private_version','A1') if q['route']=='A' else None,receipt=str(p)))
    write_csv(out/'harness-qualification.csv',harness)
    summary=dict(at=datetime.now(timezone.utc).isoformat(),minimum_roster_parents=len(roster['slots']),nominal_route_slots=len(rows),executed=sum(x['receipt'] is not None for x in rows),
        released_slots=sum(x['released'] for x in rows),unreleased_task_slots=sum(x['status']=='unrun_task_not_released' for x in rows),
        unreleased_route_slots=sum(x['status']=='unrun_route_not_released' for x in rows),released_pending=sum(x['status']=='released_queue_pending' for x in rows),
        completed_parent_groups=len({x['parent_group'] for x in rows if x['receipt'] is not None}),
        supervised_train_parents=binding.get('train_parent_groups'),training_records_used=sum(x['OBC_supervision_included'] and x['role']=='train' for x in rows),
        validation_records_used=sum(x['OBC_supervision_included'] and x['role']=='dev' for x in rows),late_outcomes=sum(x['late_train_dev_outcome'] for x in rows),
        A_semantic_pass=sum(x['route']=='A' and x['route_semantics_pass'] for x in harness),
        CloseDrawer_A_semantic_pass=sum(x['route']=='A' and x['task']=='CloseDrawer' and x['route_semantics_pass'] for x in harness),
        microwave_A_semantic_pass=sum(x['route']=='A' and x['task']=='CloseSingleDoor' and x['route_semantics_pass'] for x in harness),
        A_safe_task_success=sum(x['route']=='A' and x['safe_success'] is True for x in harness),
        D_safe_task_success=sum(x['route']=='D' and x['safe_success'] is True for x in harness),development_harness_records=len(harness),
        old_sealed_accessed=False,formal_train_ready=False,human_review='pending')
    (out/'coverage-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
