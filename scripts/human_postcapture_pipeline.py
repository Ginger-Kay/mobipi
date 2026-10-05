"""Explicit one-shot postcapture queue. Plan by default; never starts training."""
import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mobiwam.human_postcapture import (read_json, atomic_json, signature, claim,
                                      summarize_sweeps, pair_records, stamp)


def discover(batch, scenes=None):
    selected = None if scenes is None else set(scenes)
    if selected is not None and not selected:
        raise ValueError("Scene scope must not be empty")
    records=[];errors=[]
    for row in csv.DictReader((batch/'attempt-index.csv').open()):
        if row['record_type']!='primary':continue
        try:
            attempt=Path(row['result_path']).resolve().parent
            if not attempt.is_relative_to(batch.resolve()):raise ValueError('Attempt outside requested batch')
            if attempt.name!=row['attempt_id']:raise ValueError('Index attempt identity differs')
            result=read_json(attempt/'result.json');meta=read_json(attempt/'collection-metadata.json')
            if selected is not None and meta.get('scene_id') not in selected:continue
            if not result.get('ended_at') or not meta.get('ended_at'):raise ValueError('Attempt not finalized')
            if meta['record_type']!='primary' or result['route']!=row['route']:raise ValueError('Index/metadata classification differs')
            records.append(dict(attempt=str(attempt),scene=meta['scene_id'],route=meta['route'],
                                signature=signature(attempt),freeze_receipt=meta['freeze_receipt'],
                                source=meta['source_id'],config_version=meta['config_version']))
        except Exception as e:errors.append(dict(attempt=row['attempt_id'],error=str(e),outcome_preserved=True,automatic_retry=False))
    paths=[r['attempt'] for r in records]
    if len(paths)!=len(set(paths)):raise ValueError('Duplicate finalized attempt in raw index')
    keys=[(r['source'],r['config_version'],r['route']) for r in records]
    duplicates={k for k in keys if keys.count(k)>1}
    if duplicates:
        errors.append(dict(error='Duplicate primary route; hold affected groups before execution',groups=sorted(duplicates)))
        bad={(k[0],k[1]) for k in duplicates}
        records=[r for r in records if (r['source'],r['config_version']) not in bad]
    return records,errors


def run_stage(attempt,stage,ledger,sig,commands,output,workers):
    receipt=claim(ledger,attempt,stage,sig)
    atomic_json(receipt,dict(at=stamp(),status='running',attempt=str(attempt),stage=stage,
                            pid=os.getpid(),input_signature=sig,commands=commands,output=str(output)))
    def invoke(item):
        name,cmd=item
        with (output/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/(name+'-process.json'),dict(at=stamp(),pid=proc.pid,command=cmd))
            code=proc.wait()
        if code:raise RuntimeError(f'{stage}/{name} failed; no automatic retry')
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(invoke,commands))
        atomic_json(receipt,dict(at=stamp(),status='completed',attempt=str(attempt),stage=stage,input_signature=sig,output=str(output)))
    except Exception as e:
        atomic_json(receipt,dict(at=stamp(),status='engineering_hold',attempt=str(attempt),stage=stage,input_signature=sig,output=str(output),error=str(e),automatic_retry=False))
        raise


def process(row,root,ledger,workers):
    attempt=Path(row['attempt']);out=root/attempt.name;out.mkdir(exist_ok=False)
    scripts=Path(__file__).resolve().parent;sig=row['signature']
    cfg=read_json(row['freeze_receipt'])['config'];atomic_json(out/'pilot.json',cfg)
    command=[sys.executable,str(scripts/'check_human_recording.py'),'--attempt',str(attempt),'--scene',row['scene'],
             '--route',row['route'],'--output',str(out/'integrity')]
    run_stage(attempt,'integrity-mw-handle-v2',ledger,sig,[('integrity',command)],out,1)
    checked=read_json(out/'integrity/result.json')
    command=[sys.executable,str(scripts/'human_reference_audit.py'),'replay','--cpu-only','--attempt',str(attempt),
             '--output',str(out/'replay'),'--pilot',str(out/'pilot.json')]
    run_stage(attempt,'replay',ledger,sig,[('replay',command)],out,1)
    replay=read_json(out/'replay/result.json')
    if not replay['reproducible']:raise ValueError('Original action replay is not reproducible; retain evidence')
    total=checked['dimensions']['native_intervals'];width=(total+workers-1)//workers;commands=[]
    for i,begin in enumerate(range(0,total,width)):
        end=min(begin+width,total)
        commands.append((f'sweep-{i}',[sys.executable,str(scripts/'human_reference_audit.py'),'sweep','--cpu-only',
               '--attempt',str(attempt),'--output',str(out/f'sweep-{i}'),'--begin-interval',str(begin),'--end-interval',str(end)]))
    run_stage(attempt,'clearance',ledger,sig,commands,out,workers)
    sweep=summarize_sweeps([read_json(out/name/'result.json') for name,_ in commands],total)
    if signature(attempt)!=sig:raise ValueError('Input changed during postcapture processing')
    result=dict(at=stamp(),attempt=str(attempt),input_signature=sig,integrity=str(out/'integrity/result.json'),
                replay=str(out/'replay/result.json'),clearance=sweep,
                machine_qualified_success=bool(checked['success_streak_10'] and replay['reproducible'] and sweep['valid']),
                preserved_failure_record=not checked['checker_success'],review_status='pending',
                reference_selected=False,autonomous_executor_label=False,formal_train_ready=False)
    atomic_json(out/'qualification.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--scene',action='append',help='Restrict this queue to exact scene IDs; repeat for multiple scenes')
    p.add_argument('--execute-new',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('CPU workers must be 1..4')
    if a.execute_new and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only execution requires masked CUDA')
    a.output.mkdir(parents=True,exist_ok=False)
    cache=read_json(a.cache);cached={e['attempt']:e for e in cache['entries']}
    records,errors=discover(a.batch,a.scene);planned=[];completed=[];pairs=[]
    for row in records:
        entry=cached.get(row['attempt'])
        if entry:
            if entry['input_signature']!=row['signature']:
                errors.append(dict(attempt=row['attempt'],error='Cached audited input changed; hold instead of replay'));continue
            if not Path(entry['qualification']).is_file() or not Path(entry['integrity']).is_file():
                errors.append(dict(attempt=row['attempt'],error='Cached receipt missing'));continue
            checked=read_json(entry['integrity'])
            if checked.get('attempt')!=row['attempt'] or checked.get('input_signature')!=row['signature'] or not checked.get('integrity_pass'):
                errors.append(dict(attempt=row['attempt'],error='Cached integrity binding differs'));continue
            pairs.append(checked)
            planned.append(dict(**row,action='reuse_existing_audit',qualification=entry['qualification']))
        else:planned.append(dict(**row,action='new_integrity_then_single_replay_and_clearance'))
    atomic_json(a.output/'plan.json',dict(at=stamp(),records=planned,queue_scope_scene_ids=a.scene,errors=errors,execute_new=a.execute_new,training=False))
    if a.execute_new:
        for row in planned:
            if row['action']=='reuse_existing_audit':continue
            try:
                result=process(row,a.output,a.ledger,a.workers);completed.append(result)
                pairs.append(read_json(result['integrity']))
                # A derived cache for the next call; never overwrite the input cache.
                cached[row['attempt']]=dict(attempt=row['attempt'],input_signature=row['signature'],
                                           qualification=str(a.output/Path(row['attempt']).name/'qualification.json'),integrity=result['integrity'])
            except Exception as e:errors.append(dict(attempt=row['attempt'],error=str(e),automatic_retry=False))
    try:paired=pair_records(pairs)
    except Exception as e:
        paired=[];errors.append(dict(error=str(e),scope='pairing'))
    atomic_json(a.output/'next-cache.json',dict(at=stamp(),entries=list(cached.values())))
    atomic_json(a.output/'summary.json',dict(at=stamp(),mode='execute_new' if a.execute_new else 'plan_only',
        queue_scope_scene_ids=a.scene,finalized_primary_records=len(records),reused_audits=sum(r['action']=='reuse_existing_audit' for r in planned),
        new_audits_completed=len(completed),new_replay_attempts=sum((a.output/Path(r['attempt']).name/'replay-process.json').is_file() for r in planned),
        paired=paired,errors=errors,review_status='pending',training_started=False,formal_train_ready=False))
    print('POSTCAPTURE',len(records),'primary records;',len(completed),'new audits;',len(errors),'holds',flush=True)


if __name__=='__main__':main()
