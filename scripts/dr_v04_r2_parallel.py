"""Four isolated GPU collectors, global FIFO dispatch, six CPU auditors. No fitting."""
import argparse,concurrent.futures,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from dr_v04_r2_batch import audit_one,load,write,sha,stamp
from mobiwam.dr_v04_parallel import group_output,completed_groups,choose_gpu,batches


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--binding',type=Path,required=True);p.add_argument('--manual-start-receipt',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();root=Path(__file__).resolve().parent.parent;b=load(a.binding);freeze=Path(b['parent_freeze']);order=b['allowed_group_ids'];rows={x['group_id']:x for x in b['scientific_rows']}
    permit=load(a.manual_start_receipt)
    if permit.get('run_id')!=b['run_id'] or permit.get('binding_sha256')!=sha(a.binding) or permit.get('requested_gpus')!=[0,1,2,3]:raise ValueError('manual launch receipt differs')
    if b.get('parallel_gpu_indices')!=[0,1,2,3] or b.get('new_route_budget')!=105 or b.get('training_authorized') is not False:raise ValueError('parallel boundary differs')
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=b['execution_code_commit'] or subprocess.check_output(['git','-C',str(root),'status','--porcelain']).strip():raise ValueError('runtime must be frozen and clean')
    lock=(run/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if load(run/'preflight/first-batch-runtime-review.json')['runtime_checks_pass'] is not True:raise ValueError('pilot closure missing')
    done=completed_groups(run,order,rows,b['run_id']);counts={};free={0,1,2,3};active={};futures={};pool=concurrent.futures.ThreadPoolExecutor(max_workers=6)
    status=dict(stage='remainder',mode='four_gpu',state='running',started_at=stamp(),pid=os.getpid(),run_id=b['run_id'],binding=str(a.binding),binding_sha256=sha(a.binding),groups_completed=done[:],active_collectors=[],worker_count=4,cpu_audit_workers=6,training=0,sealed_test=0,human_review='pending')
    def save():
        status['active_collectors']=[dict(group_id=g,pid=x['proc'].pid,gpu=x['gpu'],output=str(x['out'])) for g,x in active.items()];write(run/'status.json',status)
    def submit(group):
        out=group_output(run,order.index(group)+1,group)
        for route in rows[group]['route_order']:
            key=(group,route);rp=out/f'route-{route}-replay.json'
            if key in futures or not rp.exists():continue
            v=load(rp)
            if v['reproducible'] is not True or sha(v['result'])!=v['sha256']:raise ValueError('replay qualification changed')
            attempt=Path(load(out/f'route-{route}-dispatched.json')['path'])
            if load(attempt/'task-video-manifest.json')['binding']['run_id']!=b['run_id']:raise ValueError('native run differs')
            futures[key]=pool.submit(audit_one,root,freeze,group,route,attempt,run/'audit'/f'{group}-{route}.json')
    def poll():
        failed=[]
        for group,x in list(active.items()):
            submit(group);rc=x['proc'].poll()
            if rc is None:continue
            x['log'].close();write(run/'batches'/f'group-{order.index(group)+1:02d}-exit.json',dict(returncode=rc,at=stamp(),pid=x['proc'].pid,gpu=x['gpu'],mode='four_gpu'))
            if rc or not (x['out']/'completed.json').exists():failed.append(group)
            else:
                camera=load(x['out']/'camera-preview/camera.json')
                if not camera['visibility_pass'] or not camera['source_integration_unchanged']:failed.append(group)
                else:done.append(group)
            free.add(x['gpu']);del active[group];status['groups_completed']=sorted(done,key=order.index)
        save()
        if failed:raise RuntimeError('collector failed; preserve active routes, stop new dispatch: '+repr(failed))
    def close_batch(batch):
        results=[]
        for group in batch:
            submit(group)
            for route in rows[group]['route_order']:
                result=futures[group,route].result()
                if not result['replay_reproducible']:raise ValueError('audit replay failure')
                results.append(result);attempt=Path(result['attempt']);target=run/'videos'/f'{group}-{route}-native-target-side.mp4'
                if not target.exists():os.link(attempt/'original.mp4',target)
                write(run/'videos'/f'{group}-{route}.json',dict(group_id=group,task=rows[group]['task'],split=rows[group]['split'],route=route,path=str(target),original=str(attempt/'original.mp4'),sha256=result['original_video_sha256'],duration_s=result['completion_time_s'],frames=result['original_video_frames'],checker_success=result['checker_success'],reason=result['raw_executor_reason'],human_review='pending',machine_eligible_for_gate=result['machine_eligible_for_gate']))
        cp=run/'batches'/f'checkpoint-{order.index(batch[-1])+1:02d}.json'
        result=dict(completed_at=stamp(),groups=batch,routes=len(results),machine_eligible_routes=sum(x['machine_eligible_for_gate'] for x in results),task_successes=sum(x['checker_success'] for x in results),human_review='pending',formal_train_ready=False,mode='four_gpu')
        if not cp.exists():write(cp,result)
        status['latest_checkpoint']=load(cp);save();print(stamp(),'checkpoint',cp.name,flush=True)
    try:
        save()
        for batch in batches(order):
            if all(g in done for g in batch) and (run/'batches'/f'checkpoint-{order.index(batch[-1])+1:02d}.json').exists():continue
            for group in batch:
                if group in done:submit(group);continue
                while not free:poll();time.sleep(3)
                if shutil.disk_usage(run).free<30*1024**3:raise RuntimeError('capacity under 30GiB')
                index=order.index(group)+1;gpu=choose_gpu(free,rows[group]['task'],counts);free.remove(gpu);counts[gpu,rows[group]['task']]=counts.get((gpu,rows[group]['task']),0)+1
                out=group_output(run,index,group);ledger=run/'batches'/f'group-{index:02d}-dispatch.json'
                if out.exists() or ledger.exists():raise ValueError('duplicate/partial scientific dispatch')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONPATH=str(root/'src')+':'+str(root/'scripts')+':'+os.environ['R2_ROBOCASA_PATH'])
                temp=run/'preflight'/f'gpu-{gpu}-tmp';temp.mkdir(exist_ok=True);env['TMPDIR']=str(temp)
                cmd=[sys.executable,str(root/'scripts/dr_v04_formal_collect.py'),'--freeze',str(freeze),'--execution-binding',str(a.binding),'--run-id',b['run_id'],'--group-id',group,'--output',str(out)]
                with ledger.open('x') as f:json.dump(dict(group_id=group,index=index,created_at=stamp(),command=cmd,binding_sha256=sha(a.binding),allocation_gpu_index=gpu,logical_gpu_index=0,mode='four_gpu',dispatch_policy='frozen FIFO; outcome-blind task-balanced free GPU'),f,indent=2)
                log=(run/'batches'/f'group-{index:02d}.log').open('a');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);active[group]=dict(proc=proc,gpu=gpu,out=out,log=log);save();print(stamp(),'group_started',index,group,'GPU',gpu,'PID',proc.pid,flush=True)
                poll()
            while active:poll();time.sleep(3)
            close_batch(batch)
        status.update(state='collection_and_machine_audits_complete',ended_at=stamp());save()
    except BaseException as exc:
        status.update(state='hold_engineering_diagnosis',error=repr(exc),held_at=stamp());save()
        # Let already-dispatched unique scientific routes finish, never terminate them.
        for group,x in list(active.items()):
            rc=x['proc'].wait();x['log'].close();write(run/'batches'/f'group-{order.index(group)+1:02d}-exit.json',dict(returncode=rc,at=stamp(),pid=x['proc'].pid,gpu=x['gpu'],mode='four_gpu_draining_after_hold'))
            try:submit(group)
            except Exception as audit_error:print('held_route_audit',group,repr(audit_error),flush=True)
        raise
    finally:pool.shutdown(wait=True)

if __name__=='__main__':main()
