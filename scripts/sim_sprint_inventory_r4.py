"""Register inherited R4 repetitions without merging them into training labels."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
from sim_sprint_inventory import read,write,csvwrite

R4=Path('/share/personal/chensiyu/haokaijiang/MobiWAM/artifacts/MMWAM-OBC-002-DR/DR-v0.4/r4-simulator-inference-diagnosis/20261003T071853Z-r4-six-and-diagnosis')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run;rows=read(r/'data/inventory-v3.json')
    allowed={x['group_id'] for x in read(r/'data/reference-sources.json')}
    receipts=sorted((R4/'episodes').glob('*/engineering-attempt-1/completed.json'));assert len(receipts)==6
    for p in receipts:
        q=read(p);assert q['group_id'] in allowed
        attempt=Path(q['attempt']);binding=read(attempt.parents[1]/'target-binding.json')
        rows.append(dict(record_id=attempt.name,view='A_R4_historical_repetition',family=q['task'],source_id=attempt.parents[1].name,
            parent_source=q['group_id'],config_id=q['group_id'],task=q['task'],actual_target=binding['fixture_name'],checker='native _check_success',
            route=q['route'],controller='reference feedback selected by historical1045 Linear',policy='no frozen BC',human_or_auto='auto',record_type='historical_seen_validation_repeat',
            model_path=str(attempt.parents[1]/'model.xml'),manifest_path=q['native_task_manifest'],
            observations_path=str(attempt/'demo.hdf5'),actions_path=str(attempt/'demo.hdf5'),trace_path=str(attempt/'trace.jsonl'),
            video_path=q['original_video'],panoramic_path=q['panoramic_video'],
            labeled_path=str(R4/'videos'/f"{q['slot']:02d}-{q['group_id']}"/'watch-labeled.mp4'),attempt=str(attempt),
            original_split='historical_validation_reuse',success=q['result']['checker_success'],native_collision=None,
            clearance_qualification='not performed in R4',human_grip='no human intervention',stop_reason=q['result']['reason'],
            progress=None,base_path_m=None,terminal_duration_s=None,head_mask=[False]*5,training_head_mask=[False]*5,
            exclusion_reason='same old Source repetition with historical selector; not new independent Source or extra training row',
            observed_head_mask=[True,False,False,False,False],existing_receipt=str(p),formal_train_ready=False,exists=attempt.exists(),autonomous_label=False))
    write(r/'data/inventory-v4.json',rows);csvwrite(r/'data/inventory-v4.csv',rows)
    write(r/'data/inventory-current.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),inventory=str(r/'data/inventory-v4.json'),previous=str(r/'data/inventory-v3.json'),
        counts={v:sum(x['view']==v for x in rows) for v in sorted(set(x['view'] for x in rows))},independent_source_increment=0,
        training_rows_unchanged=108,R4_historical_successes=4,R4_historical_failures=2,training_views_not_merged=True,
        inventory_scope='current R3 supervised36Source108 routes, R4 six seen-source repetitions, human expansion all51 indexed records, current BC attempts and new16 validation online; older lineage referenced by parent receipts'))
    print('added R4 historical6 repetitions, training unchanged108',flush=True)
if __name__=='__main__':main()
