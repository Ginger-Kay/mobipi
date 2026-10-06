"""Append a delivery inventory with final human receipts and policy failures."""
import argparse,json,csv
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from sim_sprint_inventory import HUMAN,csvwrite,write,read

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    rows=read(r/'data/inventory.json');selections={};observations={};qualifications={}
    for p in sorted((HUMAN/'qualification').glob('**/*reference-selection*.json')):
        x=read(p);attempt=x.get('selected_attempt') or x.get('attempt')
        if attempt:selections[str(attempt)]=(p,x)
    for p in sorted((HUMAN/'qualification').glob('**/*operator*observation*.json')):
        x=read(p);attempt=x.get('attempt') or x.get('selected_attempt')
        if attempt:observations[str(attempt)]=(p,x)
    for p in sorted((HUMAN/'qualification').glob('**/qualification.json')):
        x=read(p)
        if x.get('attempt'):qualifications[str(x['attempt'])]=(p,x)
    for x in rows:
        if x['view']!='B_human':continue
        attempt=Path(x['attempt']);source=attempt.parents[1]
        if (source/'target-binding.json').exists():
            b=read(source/'target-binding.json');x.update(task=b['task'],checker='native '+b['fixture_class']+' _check_success',model_path=str(source/'model.xml'),
                actual_target=b['fixture_name'],actual_target_joints=b['joints'])
        meta=read(attempt/'collection-metadata.json') if (attempt/'collection-metadata.json').exists() else {}
        x.update(controller_version=meta.get('code_commit'),source_lineage=meta.get('source_lineage'),
            reference_selection_receipt=str(selections[str(attempt)][0]) if str(attempt) in selections else '',
            reference_selection=selections.get(str(attempt),(None,{}))[1],
            human_grip_receipt=str(observations[str(attempt)][0]) if str(attempt) in observations else '',
            human_grip=observations.get(str(attempt),(None,{}))[1] or x['human_grip'])
        if str(attempt) in qualifications:
            p,q=qualifications[str(attempt)];x['qualification_receipt']=str(p);x['clearance_qualification']=q.get('clearance','unknown')
            x['machine_qualified_success']=q.get('machine_qualified_success',False)
        trace=attempt/'trace.jsonl';path=0.;first=None;last=None;old=None
        if trace.exists():
            for line in trace.open():
                y=json.loads(line)
                if first is None:first=y['before']
                before=np.array(y['before']['base_pos']);after=np.array(y['after']['base_pos'])
                path+=float(np.linalg.norm(after[:2]-before[:2]));last=y['after']
        if last:
            x.update(progress=float(np.clip(1-last['target']['door'],0,1)),base_path_m=path,
                terminal_duration_s=last['sim_time']-first['sim_time'],human_metric_scope='completed control prefix; native partial tail if any separately preserved')
        x['head_mask']=[False]*5;x['exclusion_reason']='human outcome; never automatic policy supervision'
        x['path_exists']={k:Path(x[k]).exists() for k in ('model_path','observations_path','actions_path','trace_path','video_path','manifest_path') if x.get(k)}
    for p in sorted((r/'policy').glob('ability-*/completed.json')):
        q=read(p);attempt=Path(q['attempt']);source=attempt.parents[1];binding=read(source/'target-binding.json')
        rows.append(dict(record_id=attempt.name,view='C_policy_auto',family=q['task'],parent_source=read(p.parent/'policy-init-restore.json')['parent_source'],
            source_id=source.name,config_id=q['group_id'],model_path=str(source/'model.xml'),task=q['task'],checker='native '+binding['fixture_class']+' _check_success',
            actual_target=binding['fixture_name'],actual_target_joints=binding['joints'],route='E',controller=q['controller'],policy='BC_Transformer_GMM',
            controller_version=read(p.parent/'process.json')['code_commit'],policy_binding=str(p.parent/'policy-binding.json'),monitor='exact-pad0.5mm/joint0.015 native guard',
            human_or_auto='auto',record_type='frozen_E_ability_diagnostic',episode=q['group_id'],attempt=str(attempt),
            observations_path=str(attempt/'demo.hdf5'),actions_path=str(attempt/'demo.hdf5'),trace_path=str(attempt/'trace.jsonl'),video_path=str(attempt/'original.mp4'),
            panoramic_path=str(attempt/'panoramic.mp4'),manifest_path=str(attempt/'task-video-manifest.json'),original_split='train_development_ability',success=q['native_success'],
            native_collision=q['status']=='native_forbidden_contact_stop',clearance_qualification='not qualified success; stop retained',human_grip='no human intervention',
            stop_reason=q['status'],progress=None,base_path_m=None,terminal_duration_s=q['steps']*.05,head_mask=[False]*5,
            exclusion_reason='ability diagnostic is not new16 primary policy training distribution; incomplete terminal continuous head audit',exists=True,
            autonomous_label=False,real_policy_closed_loop=True,initial_frame_padding=True,initialization_version='policy-init-v1',failure_receipt=str(attempt/'safety-stop.json')))
    for p in sorted((r/'policy').glob('ability-*/failure.json')):
        q=read(p);rows.append(dict(record_id=p.parent.name,view='C_policy_engineering',family='see frozen roster',source_id='see frozen roster',
            config_id=q['group_id'],task='see frozen roster',route='E',human_or_auto='auto',record_type='mechanical_failure_attempt',
            attempt=q.get('attempt') or '',success=None,head_mask=[False]*5,exclusion_reason=q['error'],stop_reason='engineering_missing',
            failure_receipt=str(p),autonomous_label=False))
    out=r/'data/inventory-v2.json';assert not out.exists();write(out,rows);csvwrite(r/'data/inventory-v2.csv',rows)
    for view in ('A_reference_auto','B_human','C_policy_auto','C_policy_engineering'):csvwrite(r/'data'/f'{view}-v2.csv',[x for x in rows if x['view']==view])
    write(r/'data/inventory-current.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),inventory=str(out),previous=str(r/'data/inventory.json'),
        counts={view:sum(x['view']==view for x in rows) for view in ('A_reference_auto','B_human','C_policy_auto','C_policy_engineering')},
        masks_and_splits_not_merged=True,independent_source_increment=0,human_origins_not_autonomous=True))
    print(json.dumps(read(r/'data/inventory-current.json')),flush=True)

if __name__=='__main__':main()
