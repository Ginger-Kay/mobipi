"""Read manifest-backed development inputs; never access a sealed Source."""
import argparse, csv, json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/share/personal/chensiyu/haokaijiang/MobiWAM')
HUMAN = ROOT/'artifacts/MMWAM-OBC-002-DR/development-human-scene-expansion/20261003T140621Z-two-day-preparation'
R3 = ROOT/'artifacts/MMWAM-OBC-002-DR/DR-v0.4/r3-contact-rule-removal/20261003T055050Z-r3-receipt-reuse'
P = ROOT/'artifacts/MMWAM-OBC-002-DR/DR-v0.4/20260930T115330Z-live-preflight'

def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2)+'\n')
def csvwrite(p, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    old=read(R3/'dataset/route-records.json')
    assert all(x['split'] in ('train','validation') for x in old)
    allowed={x['group_id'] for x in old}
    # Only roster metadata is read here; filter before opening any object path.
    metadata=read(P/'formal-preoutcome-freeze-v04-r1.json')
    sources=[x for x in metadata['primary'] if x['group_id'] in allowed and x['split'] in ('train','validation')]
    bygroup={x['group_id']:x for x in sources}
    write(r/'data/reference-sources.json',sources)
    inventory=[]
    for x in old:
        p=Path(x['attempt']);s=bygroup[x['group_id']]
        inventory.append(dict(record_id=p.name,view='A_reference_auto',family=x['task'],parent_source=s['raw_source'],
            source_id=Path(s['source']).name,config_id=x['group_id'],model_path=str(Path(s['source'])/'model.xml'),
            task=x['task'],checker='native _check_success',route=x['route'],controller='frozen DR-v0.4 reference feedback',
            policy='none',monitor=x['rule_version'],human_or_auto='auto',record_type='historical_primary',episode=x['group_id'],
            attempt=str(p),observations_path=str(p/'demo.hdf5'),actions_path=str(p/'demo.hdf5'),trace_path=str(p/'trace.jsonl'),
            video_path=x['video'],manifest_path=str(p/'task-video-manifest.json'),original_split=x['split'],
            success=x['success'],native_collision=x['failure'],clearance_qualification=x['safety_status'],human_grip='unknown',
            stop_reason=x['raw_reason'],progress=x['progress'],base_path_m=x['base_path_m'],terminal_duration_s=x['completion_time_s'],
            head_mask=x['label_valid_mask'],exclusion_reason='' if all(x['label_valid_mask']) else 'old_rule_censored_terminal',
            video_sha256=x['video_sha256'],existing_receipt=x['audit'],exists=p.exists(),autonomous_label=True))
    # Keep all practice/primary/supplement failures and engineering entries separate.
    human_rows=list(csv.DictReader((HUMAN/'attempt-index.csv').open()))
    latest={}
    for p in sorted((HUMAN/'qualification').glob('**/qualification.json')):
        q=read(p)
        if q.get('attempt'): latest[str(q['attempt'])]=(p,q)
    for x in human_rows:
        p=Path(x['result_path']).parent if x['result_path'] else None
        result=read(p/'result.json') if p and (p/'result.json').exists() else {}
        meta=read(p/'collection-metadata.json') if p and (p/'collection-metadata.json').exists() else {}
        qual=latest.get(str(p),(None,{}))
        inventory.append(dict(record_id=x['attempt_id'],view='B_human',family=meta.get('scene_family_id','unknown'),
            parent_source=meta.get('source_lineage'),source_id=x['source_id'],config_id=meta.get('scene_id','unknown'),
            model_path=str(p.parents[1]/'model.xml') if p else '',task=meta.get('task','unknown'),checker='native _check_success',
            route=x['route'],controller=meta.get('controller','human keyboard'),policy='none',monitor='human exact-pad rule',
            human_or_auto='human' if x['record_type']!='engineering_record_smoke' else 'engineering',record_type=x['record_type'],
            episode=meta.get('scene_id'),attempt=str(p) if p else '',observations_path=x['trajectory_path'],actions_path=x['trajectory_path'],
            trace_path=str(p/'trace.jsonl') if p else '',video_path=x['original_video_path'],panoramic_path=x['panoramic_video_path'],
            manifest_path=str(p/'task-video-manifest.json') if p else '',original_split='human_development',success=result.get('checker_success'),
            native_collision='native_forbidden_contact_stop'==result.get('reason'),clearance_qualification=qual[1].get('clearance','unknown'),
            human_grip=x['human_observation'] or 'unknown',stop_reason=result.get('reason',x['stop_reason']),progress=None,base_path_m=None,
            terminal_duration_s=result.get('elapsed_sim_s'),head_mask=[False]*5,exclusion_reason='human supervision is not autonomous outcome labels',
            qualification_receipt=str(qual[0]) if qual[0] else '',exists=bool(p and p.exists()),autonomous_label=False))
    csvwrite(r/'data/inventory.csv',inventory);write(r/'data/inventory.json',inventory)
    for view in ('A_reference_auto','B_human','C_policy_auto'):
        csvwrite(r/'data'/f'{view}.csv',[x for x in inventory if x['view']==view] or [dict(record_id='',view=view,status='no_new_policy_outcomes_yet')])
    scene_rows=list(csv.DictReader((HUMAN/'scene-index.csv').open()))
    main=[x for x in scene_rows if x['scene_id'].split('-')[1] in ('O','B','H','M') and x['scene_id'].endswith(('-01','-02'))]
    assert len(main)==16
    order={'O':0,'B':1,'H':2,'M':3}
    main.sort(key=lambda x:(x['task'],x['scene_id'][-2:],order[x['category']]))
    slots=[]
    for x in main:
        for route in ('E','D','A'):
            slots.append(dict(slot=len(slots)+1,group_id=x['scene_id'],family=x['scene_family_id'],task=x['task'],
                source=x['source'],config=x['prepared_config'],route=route,split='train' if x['scene_id'].endswith('01') else 'development-validation',
                environment_seed=int(x['environment_seed']),status='not_executed'))
    ability=[];eligible=[]
    for x in sources:
        if x['split']!='train':continue
        plan=read(x['candidate_features']);binding=read(Path(x['source'])/'target-binding.json')
        is_ok=bool(plan['records'][0]['hard_valid']) and not binding['checker_success']
        eligible.append(dict(group_id=x['group_id'],source_id=Path(x['source']).name,task=x['task'],eligible=is_ok,
            basis='existing no-added-obstacle native family; same PandaOmron; precomputed E geometry hard-valid; initial checker false',source=x['source']))
    for task in ('CloseDrawer','CloseSingleDoor'):
        candidates=sorted((x for x in eligible if x['task']==task and x['eligible']),key=lambda x:x['source_id'])
        for x in candidates[:3]:ability.append(dict(slot=len(ability)+1,route='E',**bygroup[x['group_id']]))
    checkpoints={
        'CloseDrawer':str(ROOT/'checkpoints/MMWAM-OBC-001/robocasa/bc_xfmr/04-12-CloseDrawer/seed_1_CloseDrawer_mg-300/20250413055056/models/model_epoch_1000.pth'),
        'CloseSingleDoor':str(ROOT/'checkpoints/inherited/chensiyu-20260830/robocasa/bc_xfmr/04-12-CloseSingleDoor/seed_1_CloseSingleDoor_mg-300/20250413055045/models/model_epoch_1000.pth')}
    write(r/'policy/ability-roster.json',dict(created_at=datetime.now(timezone.utc).isoformat(),slots=ability,candidates=eligible,
        selection='first3 complete source_id lexicographic, train primary only, geometry and identity only; no outcome filter',
        checkpoints=checkpoints,policy_sampling_seed=20261006,evaluation_seed=20261006,ability_route_budget_charged_to_48=True,
        overlap_with_human16=False,additional_main_slots_not_created=True))
    freeze=dict(created_at=datetime.now(timezone.utc).isoformat(),experiment_id='MMWAM-OBC-002-SIM-SPRINT-v1',phase='P0_completed_P1_next',
        main_slots=slots,main_route_budget=48,online_budget=16,ability_roster='policy/ability-roster.json',
        initializer='policy-init-v1',initial_frame_padding=True,history_length=10,action_chunk=10,horizon_seconds=120,
        restore_tolerance=1e-6,clearance_margin_m=.0005,joint_margin_rad=.015,training_seed=17,
        primary_order='per task O01 B01 H01 M01 O02 B02 H02 M02; E D A',
        controller_distribution='policy and reference views separate; fallback only after readiness failure',
        sealed_test_incident='preflight/deviations.jsonl; geometry metadata seed103 accidentally read; no test used in training/execution',
        neural_fit_budget=2,structures=['Linear(1048,32)-ReLU-Linear(32,5)','Linear(1048,5)'],fixed_final_step=2000,
        optional_multistage='skip unless actual online short-task table complete and safety module available',formal_train_ready=False)
    write(r/'freeze.json',freeze)
    summary=dict(A_reference_auto=len(old),B_human=len(human_rows),C_policy_auto=0,main_groups=len(main),main_routes=len(slots),
        capability_slots=len(ability),human_primary=sum(x['record_type']=='primary' for x in human_rows),
        human_supplement=sum(x['record_type']=='reference_supplement' for x in human_rows),formal_train_ready=False)
    write(r/'progress/P0.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
