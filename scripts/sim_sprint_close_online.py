"""Freeze all sixteen online slots, reliable labels, costs and no-success demos."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from sim_sprint_inventory import read,write,csvwrite

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    assert read(r/'online-queue/status.json')['status']=='completed_all_frozen_slots'
    roster=read(r/'online-queue/freeze.json')['groups'];rows=[];inventory=read(r/'data/inventory-v2.json')
    for x in inventory:
        if x['view']=='C_policy_auto':
            x['observed_head_mask']=[True,True,False,False,False]
            x['training_head_mask']=x['head_mask'];x['head_mask_semantics']='training view eligibility; known native diagnostic outcomes retained separately'
    for group in roster:
        for method in group['methods']:
            out=r/'episodes'/group['group_id']/method
            if (out/'not-executed.json').exists():
                q=read(out/'not-executed.json');rows.append(dict(group_id=group['group_id'],method=method,status='preflight_rejected',route='X',
                    native_success=None,safety_qualified_success=None,reason=q['reason'],started=False,controller='reference_fallback',
                    source=q['source'],record=str(out/'not-executed.json')));continue
            if not (out/'completed.json').exists():
                rows.append(dict(group_id=group['group_id'],method=method,status='engineering_missing',route=None,native_success=None,
                    safety_qualified_success=None,reason='see failure.json',started=read(out/'failure.json').get('scientific_episode_started',False),controller='reference_fallback',record=str(out/'failure.json')));continue
            q=read(out/'completed.json');attempt=Path(q['attempt']);s=read(attempt/'sprint-safety-audit.json');sel=read(out/'selection.json')
            trace=read(out/'terminal-live-state.json')['after'];native=sel['native_identity'];path=0.
            for line in (attempt/'trace.jsonl').open():
                t=json.loads(line);path+=float(np.linalg.norm(np.asarray(t['after']['base_pos'])[:2]-np.asarray(t['before']['base_pos'])[:2]))
            # Add the partial tail displacement from the last complete state.
            tail=read(out/'terminal-live-state.json')['after']['base_pos']
            if q['result']['steps']:
                last=t['after']['base_pos'];path+=float(np.linalg.norm(np.asarray(tail)[:2]-np.asarray(last)[:2]))
            row=dict(group_id=q['group_id'],task=q['task'],method=method,status='executed',route=q['route'],started=True,
                native_success=q['native_success'],safety_qualified_success=s['safety_qualified_success'],native_collision=s['native_collision'],
                clearance_pass=s['clearance_pass'],joint_margin_rad=s['joint_margin']['minimum']['margin_rad'],reason=q['result']['reason'],
                progress=float(np.clip(1-trace['target']['door'],0,1)),base_path_m=path,terminal_duration_s=q['terminal_duration_s'],
                success_duration_s=None if not q['native_success'] else q['terminal_duration_s'],steps=q['result']['steps'],
                encoder_seconds=sel['encoder_seconds'],selector_seconds=sel['selector_seconds'],planner_preflight_seconds=sel['precomputed_planner_seconds'],
                base_policy_inference_calls=0,geometry_encoder_is_diagnostic_extra=method=='geometry',controller='reference_fallback',human_interventions=0,
                record=str(out/'completed.json'),attempt=str(attempt),source=q['result']['source'],video=str(attempt/'original.mp4'),panorama=str(attempt/'panoramic.mp4'))
            rows.append(row)
            inventory.append(dict(record_id=attempt.name,view='A_new_reference_online_validation',family='MW-PILOT-family',parent_source=next(x['source'] for x in read(r/'freeze.json')['main_slots'] if x['group_id']==q['group_id']),
                source_id=attempt.parents[1].name,config_id=q['group_id'],task=q['task'],actual_target=native['fixture_name'],checker=native['checker'],route=q['route'],
                controller='inherited reference feedback, new development transfer',policy='no BC; selector '+method,human_or_auto='auto',record_type='new_online_validation',
                original_split='development-validation',observations_path=str(attempt/'demo.hdf5'),actions_path=str(attempt/'demo.hdf5'),trace_path=str(attempt/'trace.jsonl'),
                model_path=str(attempt.parents[1]/'model.xml'),video_path=str(attempt/'original.mp4'),panoramic_path=str(attempt/'panoramic.mp4'),manifest_path=str(attempt/'task-video-manifest.json'),
                success=q['native_success'],native_collision=s['native_collision'],clearance_qualification=s['actual_native_swept_geometry'],joint_qualification=s['all_safety_pass'],human_grip='no human intervention',
                stop_reason=row['reason'],progress=row['progress'],base_path_m=path,terminal_duration_s=row['terminal_duration_s'],
                observed_head_mask=[True]*5,training_head_mask=[False]*5,head_mask=[False]*5,
                exclusion_reason='new development validation, transfer difference disclosed; no retraining or merge with inherited24/12 labels',
                audit=str(attempt/'sprint-safety-audit.json'),attempt=str(attempt),exists=True,autonomous_label=True,real_native_feedback=True,
                is_BC_autonomous_execution=False,eligible_for_inherited_training=False))
    assert len(rows)==16
    csvwrite(r/'comparison/online-results.csv',rows);write(r/'comparison/online-results.json',rows)
    summaries=[]
    for method in ('learned','geometry'):
        chosen=[x for x in rows if x['method']==method];ran=[x for x in chosen if x['status']=='executed']
        summaries.append(dict(method=method,planned_sources=len(chosen),executed_sources=len(ran),preflight_rejected=sum(x['status']=='preflight_rejected' for x in chosen),
            engineering_missing=sum(x['status']=='engineering_missing' for x in chosen),native_success=sum(x['native_success'] for x in ran),safety_qualified_success=sum(x['safety_qualified_success'] for x in ran),
            native_collision=sum(x['native_collision'] for x in ran),total_complete_control_steps=sum(x['steps'] for x in ran),
            mean_base_path_m=float(np.mean([x['base_path_m'] for x in ran])) if ran else None,
            mean_failure_terminal_duration_s=float(np.mean([x['terminal_duration_s'] for x in ran if not x['native_success']])) if ran else None,
            success_duration_s=None,only_one_task_actually_executed=True,scope='reference-controller actual online, same-family development validation; no BC or two-task demo'))
    csvwrite(r/'comparison/online-summary.csv',summaries);write(r/'comparison/online-summary.json',summaries)
    write(r/'data/inventory-v3.json',inventory);csvwrite(r/'data/inventory-v3.csv',inventory)
    write(r/'data/inventory-current.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),inventory=str(r/'data/inventory-v3.json'),
        previous=str(r/'data/inventory-v2.json'),independent_source_increment=0,training_views_not_merged=True,
        counts={v:sum(x['view']==v for x in inventory) for v in sorted(set(x['view'] for x in inventory))}))
    demos=[x for x in rows if x.get('safety_qualified_success')]
    write(r/'videos/demo-selection.json',dict(created_at=datetime.now(timezone.utc).isoformat(),rule='atmost3 native success AND qualified safety from actual online records',
        selected=demos[:3],success_demos_available=len(demos),historical_or_human_videos_substituted=False,
        selected_failure_cases=[x for x in rows if x['status']=='executed' and x['method']=='learned'],
        evidence_tier='inherited L1 only; current new online negative; no L2/L3 or successful autonomous demo',selection_bias='no success demo exists'))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
