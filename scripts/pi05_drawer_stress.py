"""One recoverable coordinator, frozen six-method blocks and incremental audits."""
import argparse, csv, fcntl, hashlib, json, os, socket, subprocess, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from pi05_drawer_pipeline import R, GPUS, EGL, CP, simenv, read, write, now, outcome
from pi05_drawer_stress_prepare import definitions

PORTS=[8970,8971]
lock=threading.RLock()
running={};audit_futures=[];services=[];handles=[]

def deadline(r,key):return datetime.fromisoformat(read(r/'deadline-config.json')[key])
def seconds_until(r,key):return (deadline(r,key)-datetime.now(timezone.utc)).total_seconds()
def event(r,kind,**fields):
    with lock:
        with (r/'evidence/events.jsonl').open('a') as f:f.write(json.dumps(dict(at=now(),kind=kind,**fields))+'\n')
def patch_job(r,key,**updates):
    with lock:
        plan=read(r/'design/selected-plan.json');job=next(x for x in plan['jobs'] if x['key']==key);job.update(updates);write(r/'design/selected-plan.json',plan);return dict(job)
def rows(r):
    with lock:
        result=[outcome(j) for j in read(r/'design/selected-plan.json')['jobs']]
        for x in result:
            failure=Path(x['receipt']).parent/'failure.json'
            if failure.exists() and read(failure).get('status')=='engineering_supervisor_interrupted':
                f=read(failure);x.update(status=f['status'],executed=True,valid=False,unknown=True,
                    steps=f['recorded_prefix_steps'],policy_queries=f['recorded_prefix_queries'],
                    head_values=[None]*5,head_masks=[False]*5,native_success=None,safety_pass=None,safe_success=None,
                    target_video=f.get('target_video'),panoramic_video=f.get('panoramic_video'),
                    audit=None,terminal_duration_s=None,prefix_only=True,unlogged_suffix_unknown=True)
            m=r/'audits'/('media-'+x['key'].replace('/','-')+'.json');x['media_pass']=read(m).get('passed') if m.exists() else None
        write(r/'evidence/slot-ledger.json',dict(at=now(),rows=result));return result
def counters(rs):
    return dict(plan=len(rs),admitted=sum(x['admitted'] for x in rs),eligible=sum(x['hard_valid'] is True for x in rs),
                executed=sum(x['executed'] for x in rs),valid=sum(x['valid'] for x in rs),
                native_success=sum(x['native_success'] is True for x in rs),safe_success=sum(x['safe_success'] is True for x in rs),
                X=sum(x['status'].startswith('X') for x in rs),unknown=sum(x['unknown'] or x['status']=='engineering_unavailable' for x in rs),
                not_admitted=sum(not x['admitted'] for x in rs),unrun=sum(x['admitted'] and not x['executed'] and not x['status'].startswith('X') for x in rs),
                audited=sum(x['audit'] is not None for x in rs),media_checked=sum(x['media_pass'] is not None for x in rs))
def tables(r):
    rs=rows(r);summary=dict(at=now(),counters=counters(rs),panels={p:counters([x for x in rs if x['panel']==p]) for p in ['core','main_completion','extension']},
                          paper_evidence_ready=all(x['audit'] and x['media_pass'] is True for x in rs if x['executed']) and any(x['executed'] for x in rs),
                          selected_plan_evidence_complete=all(not x['admitted'] or x['valid'] and x['audit'] and x['media_pass'] is True or x['admitted'] and x['status'].startswith('X') for x in rs),
                          independent_parents=2,known_development=True,post_v1=True,human_review='pending',formal_train_ready=False)
    write(r/'tables/summary.json',summary)
    fields=['key','panel','environment_seed','offset_id','method','route','admitted','hard_valid','status','executed','valid','native_success','safety_pass','safe_success','collision','progress','path_m','terminal_duration_s','steps','policy_queries','audit','media_pass','receipt','target_video','panoramic_video']
    write_csv(r/'tables/T1-real-closed-loop.csv',rs,fields)
    write_csv(r/'tables/raw-video-index.csv',[x for x in rs if x['executed']],['key','method','route','status','target_video','panoramic_video','receipt','audit','media_pass'])
    states=[];prediction_rows=[]
    for state in read(r/'design/definitions-freeze.json')['states']:
        p=r/'policy/predictions'/f"{state['config_id']}.json";pred=read(p) if p.exists() else {};actual=[x for x in rs if x['config_id']==state['config_id']];binding=pred.get('state',{})
        states.append(dict(state,status=binding.get('status','not_prepared'),eligible_routes=json.dumps(binding.get('hard_valid_routes')),**counters(actual)))
        for model in ['MLP','Linear','ridge']:
            for route in 'EDA':
                value=pred.get('predictions',{}).get(model,{}).get(route);primary=next(x for x in actual if x['method']=='Fixed-'+route)
                prediction_rows.append(dict(config_id=state['config_id'],panel=state['panel'],offset_id=state['offset_id'],model=model,route=route,
                     model_status=pred.get('models',{}).get(model,{}).get('status','unavailable'),hard_valid=pred.get('valid_routes',{}).get(route),
                     selected_route=pred.get('choices',{}).get(model),predictions=json.dumps(value),observed_values=json.dumps(primary['head_values']),observed_masks=json.dumps(primary['head_masks']),primary_receipt=primary['receipt'],freeze=str(p)))
    write_csv(r/'tables/T2-frozen-predictions-lookup.csv',prediction_rows,list(prediction_rows[0]) if prediction_rows else [])
    write_csv(r/'tables/T3-start-coverage.csv',states,['config_id','panel','environment_seed','offset_id','axis','magnitude','status','eligible_routes','admitted','executed','valid','X','unknown','not_admitted','audited','media_checked'])
    # Costs use only the common observed safety-success intersection.
    comparisons=[]
    for panel in ['main','extension','all']:
        subset=[x for x in rs if panel=='all' or (x['panel']=='extension')==(panel=='extension')]
        for method in ['Fixed-D','Fixed-A','Geometry','Train-best-fixed','OBC-MLP']:
            pairs=[]
            for x in subset:
                if x['method']!=method or x['safe_success'] is not True:continue
                y=next((y for y in subset if y['config_id']==x['config_id'] and y['method']=='Fixed-E' and y['safe_success'] is True),None)
                if y and all(z.get('path_m') is not None and z.get('terminal_duration_s') is not None for z in [x,y]):pairs.append((x,y))
            comparisons.append(dict(panel=panel,method=method,reference='Fixed-E',common_safety_success_states=len(pairs),path_delta_m=float(np.mean([x['path_m']-y['path_m'] for x,y in pairs])) if pairs else None,time_delta_s=float(np.mean([x['terminal_duration_s']-y['terminal_duration_s'] for x,y in pairs])) if pairs else None))
    write_csv(r/'tables/common-safe-costs.csv',comparisons,list(comparisons[0]))
    return summary
def write_csv(path,data,fields):
    temp=path.with_suffix('.tmp');temp.parent.mkdir(parents=True,exist_ok=True)
    with temp.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(data)
    temp.replace(path)

def media(r,job,q):
    p=r/'audits'/('media-'+job['key'].replace('/','-')+'.json')
    if p.exists():return
    import h5py
    attempt=Path(q['attempt']);checks={};files=[]
    try:
        manifest=read(attempt/'task-video-manifest.json');binding=manifest['binding'];initial=read(Path(job['receipt']).parent/'initialization.json')['native']
        checks.update(run=binding['run_id']==r.name,group=binding['group_id']==job['config_id'],route=binding['route']==job['route'],attempt=binding['attempt']==str(attempt),
                      native_identity=all(binding['native'][k]==initial[k] for k in ['environment_class','fixture_name','fixture_class','joint_name','checker','checker_target','native_geometry_sha256']),
                      recorded_frame_binding=len(manifest.get('decoded_frames_sha256',[]))==q['steps'],frozen_scope=q['declared_sim_horizon_seconds']==120 and q['declared_wall_limit_seconds']==2700 and q['adapter_version']=='v6',
                      experiment_identity=q['experiment_id']=='MMWAM-OBC-002-PI05-DRAWER-STRESS-v1')
        for name in ['original.mp4','panoramic.mp4']:
            path=attempt/name
            if q['steps']<=0:files.append(dict(path=str(path),status='zero complete steps; previews and native partial retained'));continue
            h=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(path)],text=True));stream=probe['streams'][0]
            decoded=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-f','null','-'],capture_output=True,text=True,timeout=180)
            checks[name+'-frames']=int(stream['nb_read_frames'])==q['steps'];checks[name+'-decode']=decoded.returncode==0
            files.append(dict(path=str(path),sha256=h.hexdigest(),bytes=path.stat().st_size,stream=stream,decoder_stderr=decoded.stderr[-1000:]))
        if q['steps']:
            checks['distinct_views']=files[0]['sha256']!=files[1]['sha256']
            with h5py.File(attempt/'demo.hdf5','r') as h:
                g=h['data/demo_0'];checks['actions_steps']=len(g['actions'])==q['steps'];checks['states_steps_plus_initial']=len(g['states'])==q['steps']+1
        result=dict(at=now(),receipt=job['receipt'],checks=checks,files=files,passed=all(checks.values()),status='passed' if all(checks.values()) else 'failed',human_review='pending')
    except Exception:result=dict(at=now(),receipt=job['receipt'],status='failed',passed=False,checks=checks,files=files,reason=traceback.format_exc())
    write(p,result)

def owned_pid(record):
    """Reject PID reuse by exact argv, kernel start time and the registered command."""
    pid=record.get('pid');proc=Path(f'/proc/{pid}')
    if not proc.exists():return None
    try:
        argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        if argv!=record['command']:return None
        ticks=int((proc/'stat').read_text().split()[21])
        if record.get('starttime_ticks') is not None and ticks!=int(record['starttime_ticks']):return None
        boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
        if abs(boot+ticks/os.sysconf('SC_CLK_TCK')-datetime.fromisoformat(record['at']).timestamp())>15:return None
        return pid
    except (OSError,ValueError,KeyError):return None

def audit(r,job):
    receipt=Path(job['receipt'])
    if not receipt.exists():return
    q=read(receipt);attempt=Path(q['attempt']);key=job['key'].replace('/','-');cache=r/'audits'/('physical-'+key+'.json')
    af=attempt/'sprint-safety-audit.json';prior=read(cache) if cache.exists() else None
    if af.exists():
        write(cache,dict(at=now(),status='passed',exit_code=0,receipt=str(receipt),native_audit=str(af),safety_pass=read(af).get('all_safety_pass'),completed_receipt_reused=True))
    elif prior and prior.get('status')=='failed':
        event(r,'failed_audit_retained_without_repeat',key=job['key'])
    else:
        process_file=r/'launch'/('audit-'+key+'-process.json')
        record=read(process_file) if process_file.exists() else {}
        existing=owned_pid(record)
        if existing:
            event(r,'existing_owned_audit_followed',key=job['key'],pid=existing)
            while owned_pid(record) and not af.exists() and seconds_until(r,'audit_freeze')>0:time.sleep(5)
            if owned_pid(record) and not af.exists():os.kill(existing,15)
            rc=0 if af.exists() else -15
        elif (attempt/'formal-native-substeps.npz').exists():
            if prior:
                # Only an incomplete interrupted audit can resume; never repeat a passed/failed audit.
                if prior.get('resume_attempt',0)>=1:
                    event(r,'audit_resume_budget_exhausted',key=job['key']);return
                write(cache.with_name(cache.stem+'-interrupted-supervisor.json'),prior)
            write(cache,dict(at=now(),status='running',receipt=str(receipt),resume_attempt=1 if prior else 0))
            cmd=[str(R/'env/bin/python'),'-u',str(r/'runtime/mobipi/scripts/sim_sprint_safety.py'),'--receipt',str(receipt)]
            with (r/'logs'/('audit-'+key+'.log')).open('a') as f:
                proc=subprocess.Popen(cmd,env=simenv(r,0),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                write(process_file,dict(at=now(),pid=proc.pid,command=cmd,run_id=r.name,deadline_config=str(r/'deadline-config.json'),starttime_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21]))
                try:rc=proc.wait(timeout=max(1,seconds_until(r,'audit_freeze')))
                except subprocess.TimeoutExpired:
                    proc.terminate();proc.wait(timeout=30);rc=-15
        else:rc=-1
        write(cache,dict(at=now(),status='passed' if rc==0 and af.exists() else 'pending' if rc==-15 else 'failed',exit_code=rc,receipt=str(receipt),native_audit=str(af),safety_pass=read(af).get('all_safety_pass') if af.exists() else None,resume_attempt=1 if prior and not existing else 0))
    if seconds_until(r,'audit_freeze')>180:media(r,job,q)
    with lock:tables(r)
    event(r,'audit_complete',key=job['key'])

def launch(r,name,command,env):
    path=r/'logs'/f'{name}.log';f=path.open('a');proc=subprocess.Popen([str(x) for x in command],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(r/'launch'/f'{name}-process.json',dict(at=now(),pid=proc.pid,command=[str(x) for x in command],run_id=r.name,deadline_config=str(r/'deadline-config.json'),log=str(path),environment={k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID','PYTHONPATH']},starttime_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21]))
    return proc,f

def execute(r,job,worker,audits):
    if not job['hard_valid']:return
    if Path(job['receipt']).exists():
        q=read(job['receipt'])
        if q.get('usable_scientific_outcome'):
            audit_futures.append(audits.submit(audit,r,job));return
    parent=Path(job['receipt']).parent
    if parent.exists():
        failure=read(parent/'failure.json') if (parent/'failure.json').exists() else {}
        # Prefix or unresolved existing state stays unknown; no probing by rerun.
        patch_job(r,job['key'],status='engineering_unknown_prefix' if failure.get('usable_outcome') else 'engineering_unavailable',reason='existing incomplete attempt; mechanical retry requires recorded actual repair')
        event(r,'existing_attempt_isolated',key=job['key']);return
    if seconds_until(r,'last_episode_launch')<=0 or seconds_until(r,'execution_deadline')<2700:
        patch_job(r,job['key'],status='unrun_deadline');return
    cmd=[R/'env/bin/python','-u',r/'runtime/mobipi/scripts/pi05_harness_episode.py','--run',r,'--slot',str(job['slot']),'--route',job['route'],'--checkpoint-step','2000',
         '--port',str(PORTS[worker]),'--adapter-version','v6','--evaluation-tag',job['tag'],'--roster',r/'design/episode-roster.json','--purpose','paired' if job['source']=='primary' else 'online',
         '--A-private-version','A3N' if job['route']=='A' else 'A1','--sim-seconds','120','--wall-seconds','2700','--diagnostic-logging','--deadline-config',r/'deadline-config.json','--experiment-id','MMWAM-OBC-002-PI05-DRAWER-STRESS-v1']
    started=time.monotonic();name='episode-'+job['key'].replace('/','-');proc,f=launch(r,name,cmd,simenv(r,worker));running[job['key']]=dict(pid=proc.pid,started_at=now(),receipt=job['receipt'],log=str(r/'logs'/f'{name}.log'))
    patch_job(r,job['key'],status='running',worker=worker,pid=proc.pid,started_at=now());event(r,'episode_start',key=job['key'],pid=proc.pid,worker=worker)
    rc=proc.wait();f.close();running.pop(job['key'],None)
    patch_job(r,job['key'],status='completed' if Path(job['receipt']).exists() else 'engineering_unavailable',exit_code=rc,ended_at=now(),wall_elapsed_s=time.monotonic()-started)
    event(r,'episode_end',key=job['key'],exit_code=rc,wall_elapsed_s=time.monotonic()-started)
    if Path(job['receipt']).exists():audit_futures.append(audits.submit(audit,r,job))
    else:event(r,'mechanical_repair_required',key=job['key'],timebox_seconds=600,max_slot_retry=1,log=str(r/'logs'/f'{name}.log'))
    with lock:tables(r)

def estimate(r,jobs):
    observations=[]
    for j in read(r/'design/selected-plan.json')['jobs']:
        if Path(j['receipt']).exists():
            q=read(j['receipt']);duration=q.get('terminal_duration_s',0)
            if duration>1 and q.get('steps',0)>0:observations.append((datetime.fromisoformat(q['ended_at'])-datetime.fromisoformat(q['started_at'])).total_seconds()/duration)
    # Keep the full 120-s failure reserve even when observed outcomes are quick.
    slot=min(2700.,max(300.,max(observations)*120+90)) if observations else 2700.
    work=[sum(j['hard_valid'] is True for j in jobs if j['slot']==slotid) for slotid in sorted({j['slot'] for j in jobs})]
    pending=sum(not f.done() for f in audit_futures);audit_rates=[]
    for j in read(r/'design/selected-plan.json')['jobs']:
        p=Path(j['receipt'])
        if not p.exists():continue
        q=read(p);af=Path(q['attempt'])/'sprint-safety-audit.json'
        if af.exists() and q.get('terminal_duration_s',0)>1:
            audit_rates.append(read(af)['elapsed_seconds']/q['terminal_duration_s'])
    # Full 120-s risk is reserved for both scientific execution and native audit.
    audit_slot=max(audit_rates)*120+90 if audit_rates else 3600.
    new_count=sum(j['hard_valid'] is True and not Path(j['receipt']).exists() for j in jobs)
    audit_eta=(new_count+pending)*audit_slot/2*1.20+300
    return dict(slot_cap_s=slot,concurrency_discount_factor=1.20,execution_eta_s=max(work,default=0)*slot*1.20+120,
                latest_launch_eta_s=max(0,max(work,default=0)-1)*slot*1.20+120,
                audit_eta_s=audit_eta,audit_slot_full120_s=audit_slot,audit_workers=2,audit_concurrency_discount=1.20,
                audit_backlog=pending,observed_full_horizon_scaled=True,
                outcome_scores_used=False,unknown_eta_uses2700=not observations)

def heartbeats(r,stop):
    previous={};last_progress=time.monotonic();last_check=0
    while not stop.is_set():
        try:
            actual={}
            for key,job in list(running.items()):
                receipt=Path(job['receipt']);progress=receipt.parent/'query-action-feedback.jsonl';actual[key]=dict(pid=job['pid'],query_action_bytes=progress.stat().st_size if progress.exists() else 0,
                    stdout_bytes=Path(job['log']).stat().st_size if Path(job['log']).exists() else 0)
            closed=len([f for f in audit_futures if f.done()]);snapshot={'workers':actual,'audits_completed':closed,'receipts':len(list((r/'episodes').glob('*/*/engineering-attempt-*/completed.json')))}
            if snapshot!=previous:last_progress=time.monotonic();previous=snapshot
            with lock:
                q=read(r/'phase-state.json');q.update(updated_at=now(),job_heartbeat=now(),actual_progress=snapshot,seconds_without_progress=time.monotonic()-last_progress);write(r/'phase-state.json',q)
            if time.monotonic()-last_check>=60:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu','--format=csv,noheader'],text=True)
                write(r/'preflight/live-gpu-operational.json',dict(at=now(),sample=gpu,experiment_evidence=False));last_check=time.monotonic()
            if running and time.monotonic()-last_progress>=600:event(r,'progress_stall_requires_bounded_inspection',running=list(running),seconds=time.monotonic()-last_progress)
        except Exception:event(r,'heartbeat_error',reason=traceback.format_exc())
        stop.wait(60)

def main(r,v):
    lease=(r/'launch/coordinator.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(r/'launch/coordinator-process.json',dict(at=now(),pid=os.getpid(),run_id=r.name,argv=sys.argv,deadline_config=read(r/'deadline-config.json')))
    stop=threading.Event();thread=threading.Thread(target=heartbeats,args=(r,stop),daemon=True);thread.start()
    try:
        while not (r/'policy/prediction-freeze.json').exists():
            if seconds_until(r,'last_episode_launch')<=0:raise RuntimeError('preparation unavailable before launch deadline')
            time.sleep(5)
        assert subprocess.check_output(['git','-C',str(r/'runtime/mobipi'),'status','--porcelain'],text=True).strip()==''
        for w in range(2):
            ready=r/'policy'/f'service{w}/ready.json'
            if ready.exists():
                previous=read(r/'launch'/f'service{w}-process.json')
                assert owned_pid(previous) is None,'existing healthy own service requires explicit recovery binding; never duplicate'
                previous_output=ready.parent.with_name(ready.parent.name+'-preserved-supervisor0')
                assert not previous_output.exists(),'retain service lineage; never overwrite prior archive'
                ready.parent.rename(previous_output)
            op=r/'runtime/openpi';env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONPATH=f'{op}/src:{op}/scripts:{op}/packages/openpi-client/src',CUDA_VISIBLE_DEVICES=GPUS[w],JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(r/'policy'/f'jax-cache{w}'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            s=socket.socket();s.bind(('127.0.0.1',PORTS[w]));s.close()
            cmd=[R/'env/openpi/bin/python','-u',op/'scripts/pi05_fft_serve.py','--lora','--checkpoint',CP,'--output',r/'policy'/f'service{w}','--port',str(PORTS[w]),'--max-seconds',str(max(1,int(seconds_until(r,'execution_deadline')))),'--max-queries','100000']
            proc,f=launch(r,f'service{w}',cmd,env);services.append(proc);handles.append(f)
        for w in range(2):
            began=time.monotonic()
            while not (r/'policy'/f'service{w}/ready.json').exists() or read(r/'policy'/f'service{w}/ready.json').get('pid')!=services[w].pid:
                if services[w].poll() is not None or time.monotonic()-began>600:raise RuntimeError(f'isolated service{w} unavailable')
                time.sleep(5)
        with ThreadPoolExecutor(max_workers=2) as workers,ThreadPoolExecutor(max_workers=2) as audits:
            plan=read(r/'design/selected-plan.json')
            for j in plan['jobs']:
                if Path(j['receipt']).exists() and read(j['receipt']).get('usable_scientific_outcome'):
                    patch_job(r,j['key'],status='completed',recovered_existing_outcome=True)
            initial=estimate(r,plan['jobs'][:12])
            if not (r/'design/resource-freeze.json').exists():write(r/'design/resource-freeze.json',dict(at=now(),initial_estimate=initial,core36_eta_s=initial['execution_eta_s']*3,main72_eta_s=initial['execution_eta_s']*6,remaining_execution_s=seconds_until(r,'execution_deadline'),core36_not_hard_start_gate=True))
            else:event(r,'resource_estimate_repaired_after_observed_audit_cost',estimate=initial,original_freeze_preserved=True)
            with lock:q=read(r/'phase-state.json');q.update(phase='executing_blocks',status='running');write(r/'phase-state.json',q)
            for block in range(12):
                plan=read(r/'design/selected-plan.json');jobs=[x for x in plan['jobs'] if x['block']==block]
                if all(x['admitted'] for x in jobs) and all(Path(x['receipt']).exists() or not x['hard_valid'] for x in jobs):
                    for j in jobs:
                        if Path(j['receipt']).exists():audit_futures.append(audits.submit(audit,r,j))
                    continue
                eta=estimate(r,jobs);remain=seconds_until(r,'execution_deadline');delivery=seconds_until(r,'audit_freeze')
                # Recovery completes already admitted comparison slots; it never resets admission/attempts.
                already_admitted=all(x['admitted'] for x in jobs)
                allowed=already_admitted or (eta['execution_eta_s']<=remain and eta['execution_eta_s']+eta['audit_eta_s']<=delivery and seconds_until(r,'last_episode_launch')>=eta['latest_launch_eta_s'] and os.statvfs(r).f_bavail*os.statvfs(r).f_frsize>20*1024**3)
                event(r,'block_resource_decision',block=block,admitted=allowed,remaining_execution_s=remain,remaining_audit_s=delivery,eta=eta,recovery_of_already_admitted_block=already_admitted)
                if not allowed:break
                for j in jobs:patch_job(r,j['key'],admitted=True,admitted_at=now())
                if not any(j['hard_valid'] is True for j in jobs):continue
                def state_work(state_jobs,w):
                    for job in state_jobs:execute(r,job,w,audits)
                slots=sorted({j['slot'] for j in jobs});futures=[workers.submit(state_work,[j for j in jobs if j['slot']==slot],w) for w,slot in enumerate(slots)]
                for f in futures:f.result()
                with lock:tables(r)
                # A bounded audit queue prevents uncontrolled accumulation.
                while sum(not f.done() for f in audit_futures)>12 and seconds_until(r,'audit_freeze')>300:time.sleep(5)
            event(r,'scientific_queue_closed',reason='ordered resource gate or all blocks disposed',running=list(running))
            # Services stop after every own episode has saved; audit queue remains CPU-only.
            assert not running
            for p in services:
                if p.poll() is None:p.terminate();p.wait(timeout=30)
            for f in audit_futures:
                try:f.result(timeout=max(1,seconds_until(r,'audit_freeze')))
                except Exception:event(r,'audit_pending_or_failed',reason=traceback.format_exc())
        summary=tables(r);write(r/'delivery/coordinator-completed.json',dict(at=now(),status='science_and_bounded_audit_closed_pending_Git_delivery',summary=summary))
        with lock:q=read(r/'phase-state.json');q.update(phase='artifacts_ready',status='awaiting_agent_delivery',ended_at=now(),counters=summary['counters']);write(r/'phase-state.json',q)
    except Exception:
        event(r,'coordinator_failure',reason=traceback.format_exc());write(r/'delivery/coordinator-failure.json',dict(at=now(),reason=traceback.format_exc()));raise
    finally:
        assert not running,'active own episodes require provenance-bound recovery; do not interfere'
        for p in services:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
        for f in handles:f.close()
        stop.set();thread.join(timeout=2)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--old-run',type=Path,required=True);a=ap.parse_args();main(a.run,a.old_run)
