"""Single-GPU prospective queue; CPU audits; no scientific retry or learning.

The collector exclusively creates each group directory and each route-start
ledger. This supervisor does not restart a partial group. Audit recomputation
is separate from collection and cannot create route outcomes.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def stamp(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(Path(p).read_text())
def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit_one(root,freeze,group,route,attempt,out):
    if out.exists():
        old=load(out)
        if old['attempt']!=str(attempt) or old['group_id']!=group or old['route']!=route:
            raise ValueError('audit identity differs')
        return old
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    cmd=[sys.executable,str(root/'scripts/dr_v04_formal_audit.py'),'--freeze',str(freeze),
         '--group-id',group,'--route',route,'--attempt',str(attempt),'--output',str(out)]
    with out.with_suffix('.log').open('a') as log:
        rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    if rc:raise RuntimeError('audit process failed: '+str(out.with_suffix('.log')))
    result=load(out)
    from mobiwam.reference_controller_events import load_replay_events
    # The collector already verified one real replay. Retain its native event
    # protocol and audit receipts; do not invoke a second replay here.
    manifest=load(attempt/'task-video-manifest.json')
    if not manifest['binding']['run_id']:raise ValueError('native run identity empty')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--stage',choices=('pilot','remainder'),required=True)
    a=p.parse_args();run=a.run.resolve();binding=load(a.binding)
    root=Path(__file__).resolve().parent.parent;freeze=Path(binding['parent_freeze'])
    frozen=load(freeze);order=binding['allowed_group_ids'];rows={r['group_id']:r for r in binding['scientific_rows']}
    if len(order)!=35 or binding['new_route_budget']!=105 or binding['training_authorized'] is not False:
        raise ValueError('wrong R2 budget or learning boundary')
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=binding['execution_code_commit']:
        raise ValueError('supervisor code not frozen')
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain']).strip():raise ValueError('dirty runtime')
    lock=(run/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.stage=='remainder':
        review=load(run/'preflight/first-batch-runtime-review.json')
        if review.get('runtime_checks_pass') is not True or review.get('group_ids')!=order[:4]:
            raise ValueError('first batch runtime closure missing')
    selected=list(enumerate(order,1))[:4] if a.stage=='pilot' else list(enumerate(order,1))[4:]
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=3)
    futures={};collected=[];pending=[]
    audit_root=run/'audit';video_root=run/'videos';audit_root.mkdir(exist_ok=True);video_root.mkdir(exist_ok=True)
    status=dict(stage=a.stage,started_at=stamp(),pid=os.getpid(),run_id=binding['run_id'],
                binding=str(a.binding),binding_sha256=sha(a.binding),groups_completed=[],
                state='running',training=0,sealed_test=0,human_review='pending')
    write(run/'status.json',status)
    def submit_available(group,out):
        for route in rows[group]['route_order']:
            key=(group,route)
            receipt=out/f'route-{route}-replay.json'
            if key in futures or not receipt.exists():continue
            replay=load(receipt)
            if replay['reproducible'] is not True:raise ValueError('nonreproducible route; hold queue')
            attempt=Path(load(out/f'route-{route}-dispatched.json')['path'])
            if load(attempt/'task-video-manifest.json')['binding']['run_id']!=binding['run_id']:
                raise ValueError('native run_id differs')
            ap=audit_root/f'{group}-{route}.json'
            futures[key]=pool.submit(audit_one,root,freeze,group,route,attempt,ap)
            print(stamp(),'audit_started',group,route,flush=True)
    def finish_batch(batch):
        audits=[]
        for group in batch:
            for route in rows[group]['route_order']:
                key=(group,route)
                if key not in futures:raise ValueError('missing route audit '+repr(key))
                result=futures[key].result();audits.append(result)
                if not result['replay_reproducible']:raise ValueError('audit replay failed')
                attempt=Path(result['attempt']);native=attempt/'original.mp4'
                target=video_root/f'{group}-{route}-native-target-side.mp4'
                if not target.exists():os.link(native,target)
                meta=dict(group_id=group,task=rows[group]['task'],split=rows[group]['split'],route=route,
                          path=str(target),original=str(native),sha256=result['original_video_sha256'],
                          duration_s=result['completion_time_s'],frames=result['original_video_frames'],
                          checker_success=result['checker_success'],reason=result['raw_executor_reason'],
                          human_review='pending',machine_eligible_for_gate=result['machine_eligible_for_gate'])
                write(video_root/f'{group}-{route}.json',meta)
        checkpoint=dict(completed_at=stamp(),groups=batch,routes=len(audits),
            machine_eligible_routes=sum(x['machine_eligible_for_gate'] for x in audits),
            task_successes=sum(x['checker_success'] for x in audits),
            failure_or_ineligible=[dict(group_id=x['group_id'],route=x['route'],reason=x['raw_executor_reason'],
                failure_label=x['irreversible_or_collision']) for x in audits if not x['checker_success'] or not x['machine_eligible_for_gate']],
            human_review='pending',formal_train_ready=False)
        write(run/'batches'/f'checkpoint-{order.index(batch[-1])+1:02d}.json',checkpoint)
        status['latest_checkpoint']=checkpoint;write(run/'status.json',status)
        print(stamp(),'checkpoint',json.dumps(checkpoint),flush=True)
    try:
        for index,group in selected:
            if shutil.disk_usage(run).free<30*1024**3:raise RuntimeError('shared capacity below 30GiB; pause writes')
            batch_no=0 if index<=4 else 1+(index-5)//6
            out=run/'batches'/f'{batch_no:02d}'/f'{index:02d}-{group}'
            ledger=run/'batches'/f'group-{index:02d}-dispatch.json'
            if ledger.exists() or out.exists():
                # Recovery may consume a completed group but never re-execute it.
                if not (out/'completed.json').exists():
                    raise RuntimeError('partial/previous dispatch requires route-level recovery; no automatic replay of outcomes: '+str(out))
                if load(out/'completed.json')['route_outcomes']!=3:raise ValueError('incomplete group')
                submit_available(group,out)
            else:
                cmd=[sys.executable,str(root/'scripts/dr_v04_formal_collect.py'),'--freeze',str(freeze),
                     '--execution-binding',str(a.binding),'--run-id',binding['run_id'],'--group-id',group,'--output',str(out)]
                ledger.parent.mkdir(exist_ok=True,parents=True)
                with ledger.open('x') as f:json.dump(dict(group_id=group,index=index,batch=batch_no,command=cmd,created_at=stamp(),binding_sha256=sha(a.binding)),f,indent=2)
                log_path=run/'batches'/f'group-{index:02d}.log'
                with log_path.open('a') as log:
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    status.update(active_group=group,collector_pid=proc.pid,log=str(log_path));write(run/'status.json',status)
                    print(stamp(),'group_started',index,group,'pid',proc.pid,flush=True)
                    while proc.poll() is None:
                        submit_available(group,out)
                        time.sleep(5)
                    submit_available(group,out)
                    write(run/'batches'/f'group-{index:02d}-exit.json',dict(returncode=proc.returncode,at=stamp(),pid=proc.pid))
                    if proc.returncode:raise RuntimeError('collector exited; preserve partial outcome and hold: '+str(log_path))
                if not (out/'completed.json').exists():raise ValueError('collector lacks completion receipt')
            camera=load(out/'camera-preview/camera.json')
            if not camera['visibility_pass'] or not camera['source_integration_unchanged']:raise ValueError('camera/restore precheck failed')
            collected.append(group);pending.append(group);status['groups_completed']=collected[:]
            status.update(active_group=None,collector_pid=None);write(run/'status.json',status)
            if (a.stage=='pilot' and index==4) or (a.stage=='remainder' and ((index-4)%6==0 or index==35)):
                finish_batch(pending);pending=[]
        status.update(state='pilot_machine_audit_complete_pending_runtime_review' if a.stage=='pilot' else 'collection_and_machine_audits_complete',ended_at=stamp())
        write(run/'status.json',status)
    except BaseException as exc:
        status.update(state='hold_engineering_diagnosis',error=repr(exc),held_at=stamp())
        write(run/'status.json',status);raise
    finally:pool.shutdown(wait=True)

if __name__=='__main__':main()
