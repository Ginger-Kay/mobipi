"""Read saved train/validation receipts only; generate a missing-aware R2 table."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from mobiwam.dr_v04_r2_summary import evaluate,ADMITTED,ROUTES

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=load(run/'manifest.json');binding=load(manifest['binding']);freeze=load(binding['parent_freeze'])
    if sha(binding['parent_freeze'])!=binding['parent_freeze_sha256']:raise ValueError('freeze hash mismatch')
    roster=[next(r for r in freeze['primary'] if r['group_id']==g) for g in freeze['task_interleaved_train_validation_order']]
    admission=load(run/'seed109-admission.json')
    if admission['group_id']!=ADMITTED or not admission['researcher_authorized_recollection_exception'] or admission['human_review']!='approved':raise ValueError('unapproved seed109 exception')
    rows=[];videos=[]
    for group in roster:
        g=group['group_id'];is_admitted=g==ADMITTED
        for route in ROUTES:
            row=dict(group_id=g,task=group['task'],split=group['split'],route=route,
                outcome_previously_observed=is_admitted,human_review='approved' if is_admitted else 'pending',
                source_input_sha256=group['source_input_sha256'],geometry_preoutcome_choice=group['geometry_preoutcome_choice'],
                machine_eligible_for_gate=False,success=None,failure=None,progress=None,base_path_m=None,completion_time_s=None,
                status='not_dispatched_or_recording',attempt=None,video=None,raw_reason=None)
            audit_path=Path(admission['route_audits'][route]) if is_admitted else run/'audit'/f'{g}-{route}.json'
            attempt=None
            if audit_path.exists():
                audit=load(audit_path)
                if (audit['group_id'],audit['route'],audit['split'])!=(g,route,group['split']):raise ValueError('audit identity mismatch')
                if audit['source_input_sha256']!=group['source_input_sha256']:raise ValueError('audit Source mismatch')
                attempt=Path(audit['attempt']);row.update(status=audit['status'],audit=str(audit_path),audit_sha256=sha(audit_path),
                    machine_eligible_for_gate=audit['machine_eligible_for_gate'],success=audit['checker_success'],failure=audit['irreversible_or_collision'],
                    progress=audit['task_progress_after'],base_path_m=audit['actual_base_path_m'],completion_time_s=audit['completion_time_s'],raw_reason=audit['raw_executor_reason'])
            elif not is_admitted:
                dispatched=list((run/'batches').glob(f'*/*-{g}/route-{route}-dispatched.json'))
                if len(dispatched)>1:raise ValueError('duplicate prospective route')
                if dispatched:
                    attempt=Path(load(dispatched[0])['path']);raw=load(attempt/'result.json')
                    row.update(status='recorded_pending_machine_audit',raw_reason=raw['reason'],raw_checker_success=raw['checker_success'])
            if attempt is not None:
                native=load(attempt/'task-video-manifest.json');ident=native['binding']
                if (ident['group_id'],ident['route'])!=(g,route):raise ValueError('native identity mismatch')
                if not is_admitted and ident['run_id']!=binding['run_id']:raise ValueError('run_id mismatch')
                movie=attempt/'original.mp4';entry=native['files']['original.mp4']
                if not movie.is_file() or movie.stat().st_size!=entry['size']:raise ValueError('native movie missing/size mismatch')
                if is_admitted and entry['sha256']!=admission['approved_video_sha256'][route]:raise ValueError('approved side video mismatch')
                row.update(attempt=str(attempt),video=str(movie),video_sha256=entry['sha256'],frames=native['steps'])
                video={k:row.get(k) for k in ('group_id','task','split','route','video','video_sha256','frames','completion_time_s','raw_reason','status','human_review')}
                video['checker_success']=row['success'] if row['success'] is not None else row.get('raw_checker_success')
                videos.append(video)
            rows.append(row)
    # Scientific qualification cannot be inferred from a mere lack of errors.
    # Only a separately reviewed full-run engineering closure can discharge it.
    closure=run/'gate/runtime-qualification-closure.json';closed=False
    if closure.exists():
        q=load(closure)
        closed=q.get('all_36_source_identities_and_restore_closed') is True and q.get('hard_invalid_or_stratum_leakage_ruled_out') is True and q.get('systemic_errors_absent') is True and q.get('full_route_visibility_and_taxonomy_closed') is True
        if q.get('binding_sha256')!=sha(manifest['binding']):raise ValueError('qualification closure binding differs')
    evaluation=evaluate(rows,roster,closed)
    evaluation.update(created_at=datetime.now(timezone.utc).isoformat(),execution_binding=manifest['binding'],execution_binding_sha256=sha(manifest['binding']),
        analysis_code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parent.parent),'rev-parse','HEAD'],text=True).strip(),
        seed109_exception='whole R1 package; known outcome admitted by R2; original first 3 excluded; sensitivity predeclared',
        new_outcomes=sum(r['attempt'] is not None and r['group_id']!=ADMITTED for r in rows),
        human_approved_sources=1,new_human_review='pending',source_denominator=36,sealed_test_sources=12,sealed_test_read=False)
    from mobiwam.dr_v04_r2_pack import pack
    evaluation['paired_supervision_candidate']=pack(run,Path(binding['parent_freeze']),roster,rows,out,Path(__file__).resolve().parent.parent)
    write(out/'route-outcomes.json',rows);write(out/'mechanical-gate.json',evaluation)
    with (out/'route-outcomes.csv').open('w') as f:
        keys=['group_id','task','split','route','status','success','failure','progress','base_path_m','completion_time_s','raw_reason','machine_eligible_for_gate','human_review','outcome_previously_observed','attempt','video']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    with (out/'videos.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(videos[0]) if videos else ['group_id']);writer.writeheader();writer.writerows(videos)
    header='# Native side videos — full researcher review pending for all R2 groups\n\nSampled-frame diagnostics do not replace full-video review.\n\n'
    text=header+'|Source|Route|Frames|Reason|Review|Absolute native video|\n|---|---|---:|---|---|---|\n'
    for v in videos:text+=f"|{v['group_id']}|{v['route']}|{v['frames']}|{v['raw_reason']}|{v['human_review']}|{v['video']}|\n"
    (out/'videos.md').write_text(text)
    priority=[v for v in videos if v['group_id']!=ADMITTED and (v['group_id'] in binding['allowed_group_ids'][:4] or v['checker_success'] is not True or v['status']!='machine_audit_pass_pending_research_review')]
    write(out/'priority-review-videos.json',priority)
    write(run/'gate/latest-summary.json',dict(output=str(out),created_at=evaluation['created_at'],gate_status=evaluation['gate_status']))
    print(json.dumps({k:evaluation[k] for k in ('new_outcomes','machine_eligible_routes','complete_machine_dataset','gate_status')}))

if __name__=='__main__':main()
