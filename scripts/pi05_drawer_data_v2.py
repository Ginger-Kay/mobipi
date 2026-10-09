"""Two isolated resident services, atomic state claims, minimal outcome labels."""
import argparse, csv, fcntl, http.client, json, os, socket, subprocess, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from pi05_drawer_pipeline import R, GPUS, EGL, CP, simenv, read, write, now

PORTS=[8980,8981];lock=threading.RLock();stop=threading.Event();heartbeat_done=threading.Event();running={}
SCIENTIFIC={'checker_success_10_steps','qp_protective_stop','joint_margin_stop','native_forbidden_contact_stop','policy_budget_stop_120s'}

def terminal(q):
    # A wall timeout or Python exception is never a scientific zero.
    if (q.get('status') or '').startswith('X_'):return 'X',None
    if q.get('status') not in SCIENTIFIC:return 'unknown',None
    if 'native_success' not in q:return 'unknown',None
    return 'completed',int(bool(q['native_success']))

def owned(rec):
    p=Path('/proc')/str(rec.get('pid',0))
    try:return p.exists() and [x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]==rec['command'] and (p/'stat').read_text().split()[21]==rec['starttime_ticks']
    except (OSError,KeyError):return False

def event(r,name,**data):
    with lock:
        with (r/'events.jsonl').open('a') as f:f.write(json.dumps(dict(at=now(),event=name,**data))+'\n')

def change(r,sid=None,jid=None,**values):
    with lock:
        d=read(r/'slot-ledger.json')
        if sid is not None:d['states'][sid-1].update(values)
        if jid is not None:d['slots'][jid-1].update(values)
        d['updated_at']=now();write(r/'slot-ledger.json',d)

def summary(r):
    with lock:
        d=read(r/'slot-ledger.json');rows=d['slots'];counts={s:sum(x['status']==s for x in rows) for s in ['completed','unknown','X','unrun','running']}
        counts.update(success=sum(x['success']==1 for x in rows),failure=sum(x['success']==0 for x in rows),plan=288)
        same=diff=0
        for s in d['states']:
            group=[x for x in rows if x['state_order']==s['state_order']]
            if all(x['status']=='completed' for x in group):
                if len({x['success'] for x in group})==1:same+=1
                else:diff+=1
        value=dict(updated_at=now(),counts=counts,complete_EDA_same_labels=same,complete_EDA_different_labels=diff,offline_safety_audit=False,safety_qualified=False,training=False)
        write(r/'tables/summary.json',value)
        def csvwrite(name,items,fields):
            path=r/'tables'/name;tmp=path.with_suffix('.tmp')
            with tmp.open('w') as f:
                w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(items)
            tmp.replace(path)
        csvwrite('labels.csv',rows,['slot_order','state_order','parent','split','offset','route','success'])
        csvwrite('status-uris.csv',rows,['slot_order','state_order','status','attempt','receipt','initial_state','initial_RGB','target_video','panoramic_video','trajectory','media_closed'])
        coverage=[]
        for split in ['train','dev']:
            for offset in read(r/'design/definitions-freeze.json')['params']:
                for route in 'EDA':
                    a=[x for x in rows if (x['split'],x['offset'],x['route'])==(split,offset,route)]
                    coverage.append(dict(split=split,offset=offset,route=route,success=sum(x['success']==1 for x in a),failure=sum(x['success']==0 for x in a),reliable=sum(x['status']=='completed' for x in a),unknown=sum(x['status']=='unknown' for x in a),X=sum(x['status']=='X' for x in a),unrun=sum(x['status']=='unrun' for x in a),running=sum(x['status']=='running' for x in a),plan=len(a)))
        csvwrite('coverage.csv',coverage,list(coverage[0]));return value

def launch(r,name,cmd,env):
    cmd=list(map(str,cmd));record=r/'launch'/(name+'-process.json');log=r/'logs'/(name+'.log')
    if record.exists() and owned(read(record)):return None,read(record),None
    f=log.open('a');p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    rec=dict(at=now(),pid=p.pid,command=cmd,starttime_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],run_id=r.name,log=str(log),environment={k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID','PYTHONPATH']})
    write(record,rec);return p,rec,f

def await_process(p,rec,f):
    if p is not None:rc=p.wait()
    else:
        while owned(rec):time.sleep(2)
        rc=None
    if f:f.close()
    return rc

def claim(r,w):
    with lock:
        d=read(r/'slot-ledger.json')
        pending=[s for s in d['states'] if s['status'] in ['unrun','claimed'] and (s['claimed_by'] is None or s['claimed_by']==w)]
        for s in pending:
            if s['stage']==2 and any(x['stage']==1 and x['status'] not in ['disposed','X','unknown'] for x in d['states']):return 'wait'
            s.update(status='claimed',claimed_by=w,claimed_at=s.get('claimed_at',now()));d['updated_at']=now();write(r/'slot-ledger.json',d)
            event(r,'atomic_state_claim',state_order=s['state_order'],worker=w);return s.copy()
        return None

def eta(r):
    # Historical full-horizon scaling, no offline audit cost and no test episode.
    base=read(r/'preflight/ETA.json')['full120_execution_eta_s'];ratios=[]
    for j in read(r/'slot-ledger.json')['slots']:
        if j['status']!='completed' or not j.get('receipt'):continue
        q=read(j['receipt']);sim=q.get('terminal_duration_s',0)
        if sim>1:ratios.append((datetime.fromisoformat(q['ended_at'])-datetime.fromisoformat(q['started_at'])).total_seconds()/sim)
    return min(2700,max(base,max(ratios,default=0)*120+90))

def execute(r,s,j,w):
    if j['status'] in ['completed','X','unknown']:return True
    deadlines=read(r/'deadline-config.json');remaining=(datetime.fromisoformat(deadlines['execution_deadline'])-datetime.now(timezone.utc)).total_seconds()
    budget=remaining-deadlines['save_reserve_seconds'];estimate=eta(r)
    if budget<estimate:
        event(r,'ordered_slot_time_admission_closed',slot_order=j['slot_order'],remaining_s=remaining,full120_eta_s=estimate);stop.set();return False
    if os.statvfs(r).f_bavail*os.statvfs(r).f_frsize<10*1024**3:event(r,'storage_admission_closed');stop.set();return False
    attempt=j['attempt'];out=r/'episodes'/('data-'+j['route'])/f"slot-{s['state_order']:02d}-{s['config_id']}"/f'engineering-attempt-{attempt}'
    receipt=out/'completed.json'
    if receipt.exists():q=read(receipt)
    else:
        cap=min(2700,int(budget));derived=dict(deadlines,wall_seconds=cap,canonical_config=str(r/'deadline-config.json'))
        dp=r/'launch'/f"deadline-slot-{j['slot_order']}-attempt-{attempt}.json";write(dp,derived)
        cmd=[R/'env/bin/python','-u',r/'runtime/mobipi/scripts/pi05_harness_episode.py','--run',r,'--slot',s['state_order'],'--route',j['route'],'--checkpoint-step',2000,'--port',PORTS[w],'--adapter-version','v6','--evaluation-tag','data-'+j['route'],'--roster',r/'design/episode-roster.json','--purpose','paired','--A-private-version','A3N' if j['route']=='A' else 'A1','--sim-seconds',120,'--wall-seconds',cap,'--attempt',attempt,'--deadline-config',dp,'--experiment-id','MMWAM-OBC-002-PI05-DRAWER-DATA-v2']
        if attempt:cmd+=['--mechanical-repair-receipt',r/'launch'/f"repair-slot-{j['slot_order']}-attempt-{attempt}.json"]
        p,rec,f=launch(r,f"episode-{j['slot_order']}-attempt-{attempt}",cmd,simenv(r,w));running[w]=rec
        change(r,jid=j['slot_order'],status='running',receipt=str(receipt),pid=rec['pid'],worker=w,started_at=now(),initial_state=s['initial_state'],initial_RGB=s['initial_RGB'])
        event(r,'slot_start',slot_order=j['slot_order'],route=j['route'],worker=w,pid=rec['pid'],wall_cap=cap,ETA=estimate)
        rc=await_process(p,rec,f);running.pop(w,None)
        event(r,'slot_process_closed',slot_order=j['slot_order'],exit_code=rc)
        q=read(receipt) if receipt.exists() else {}
    status,label=terminal(q)
    # Preserve checker-confirmed scientific terminal even if later media close failed.
    if status=='unknown':
        results=list(out.glob('attempt-*/result.json'))
        if len(results)==1:
            native=read(results[0]);reason=native.get('reason')
            if reason in SCIENTIFIC:
                status,label='completed',int(bool(native['checker_success']));q=dict(q,attempt=str(results[0].parent))
    identity=read(out/'process.json') if (out/'process.json').exists() else {}
    if identity.get('config_id')!=s['config_id'] or identity.get('route')!=j['route'] or identity.get('environment_seed')!=s['environment_seed']:status,label='unknown',None
    if status=='completed':
        raw=Path(q['attempt']);refs={k:str(raw/n) for k,n in [('target_video','original.mp4'),('panoramic_video','panoramic.mp4'),('trajectory','demo.hdf5')]}
        media_closed=all(Path(v).exists() for v in refs.values())
        change(r,jid=j['slot_order'],status=status,success=label,receipt=str(receipt) if receipt.exists() else str(raw/'result.json'),ended_at=now(),media_closed=media_closed,**refs)
    else:change(r,jid=j['slot_order'],status=status,success=label,ended_at=now())
    summary(r);event(r,'slot_saved',slot_order=j['slot_order'],status=status,success=label)
    if status=='unknown' and not stop.is_set():
        # The agent may provide a real repair; unchanged conditions never retry.
        wait_until=min(time.monotonic()+600,time.monotonic()+max(0,remaining-estimate-90))
        while time.monotonic()<wait_until and not stop.is_set():
            path=r/'launch'/f"repair-slot-{j['slot_order']}-attempt-{attempt+1}.json"
            if path.exists():
                repair=read(path);assert repair.get('actual_repair_commit') and repair.get('issue_signature') and repair.get('description')
                with lock:
                    d=read(r/'slot-ledger.json');issue=repair['issue_signature'];used=d['issue_repairs'].get(issue,0)
                    if attempt>=10 or used>=10:break
                    d['issue_repairs'][issue]=used+1;d['attempts'].append(dict(at=now(),slot=j['slot_order'],attempt=attempt+1,repair=str(path)))
                    d['slots'][j['slot_order']-1].update(status='unrun',attempt=attempt+1);write(r/'slot-ledger.json',d)
                return execute(r,s,read(r/'slot-ledger.json')['slots'][j['slot_order']-1],w)
            time.sleep(5)
    return True

def worker(r,w):
    while not stop.is_set():
        s=claim(r,w)
        if s=='wait':time.sleep(2);continue
        if s is None:return
        try:
            bp=r/'design/anchors'/s['config_id']/'binding.json'
            if not bp.exists():
                cmd=[R/'env/bin/python','-u',r/'runtime/mobipi/scripts/pi05_drawer_data_prepare.py','--run',r,'--state',s['state_order']]
                p,rec,f=launch(r,'prepare-'+str(s['state_order']),cmd,simenv(r,w));running[w]=rec;rc=await_process(p,rec,f);running.pop(w,None)
                if not bp.exists():raise RuntimeError('start construction unavailable; inspect task-owned preparation log')
            s=read(bp)
            with lock:
                roster=read(r/'design/episode-roster.json');roster['slots'][s['state_order']-1]=s;write(r/'design/episode-roster.json',roster)
            jobs=[j for j in read(r/'slot-ledger.json')['slots'] if j['state_order']==s['state_order']]
            for j in jobs:
                if stop.is_set():return
                if not s['hard_valid_routes'][j['route']]:change(r,jid=j['slot_order'],status='X',success=None);summary(r);continue
                if not execute(r,s,j,w):return
            change(r,sid=s['state_order'],status='disposed',disposed_at=now())
        except Exception:
            event(r,'mechanical_state_unavailable',state_order=s['state_order'],error=traceback.format_exc())
            for j in read(r/'slot-ledger.json')['slots']:
                if j['state_order']==s['state_order'] and j['status']=='unrun':change(r,jid=j['slot_order'],status='unknown',success=None)
            change(r,sid=s['state_order'],status='unknown');summary(r)

def heartbeat(r):
    previous=None;last=time.monotonic()
    while not heartbeat_done.wait(60):
        progress={}
        for w,rec in list(running.items()):
            log=Path(rec['log']);progress[str(w)]=dict(pid=rec['pid'],log_bytes=log.stat().st_size if log.exists() else 0,owned_alive=owned(rec))
        if progress!=previous:previous=progress;last=time.monotonic()
        write(r/'phase-state.json',dict(updated_at=now(),heartbeat=now(),phase='collecting',actual_progress=progress,seconds_without_progress=time.monotonic()-last,summary=summary(r)))
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
        contexts=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
        write(r/'operations/live-gpu.json',dict(at=now(),experiment_evidence=False,sample=gpu.stdout,contexts=contexts.stdout))
        if time.monotonic()-last>=600:event(r,'bounded_stall_inspection_required',workers=progress)
        if datetime.now(timezone.utc)>=datetime.fromisoformat(read(r/'deadline-config.json')['execution_deadline']):stop.set()

def main(r):
    lease=(r/'launch/coordinator.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(r/'launch/coordinator-process.json',dict(at=now(),pid=os.getpid(),command=sys.argv,run_id=r.name))
    services=[];thread=threading.Thread(target=heartbeat,args=(r,),daemon=True);thread.start()
    try:
        for w in range(2):
            op=r/'runtime/openpi';env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONPATH=f'{op}/src:{op}/scripts:{op}/packages/openpi-client/src',CUDA_VISIBLE_DEVICES=GPUS[w],JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(r/'policy'/f'jax-cache{w}'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            port=PORTS[w];existing=r/'launch'/f'service{w}-process.json'
            if not (existing.exists() and owned(read(existing))):
                with socket.socket() as sock:sock.bind(('127.0.0.1',port))
            remaining=int((datetime.fromisoformat(read(r/'deadline-config.json')['execution_deadline'])-datetime.now(timezone.utc)).total_seconds())
            cmd=[R/'env/openpi/bin/python','-u',op/'scripts/pi05_fft_serve.py','--lora','--checkpoint',CP,'--output',r/'policy'/f'service{w}','--port',port,'--max-seconds',remaining,'--max-queries',100000]
            p,rec,f=launch(r,f'service{w}',cmd,env);services.append((p,rec,f))
        for w,(p,rec,f) in enumerate(services):
            start=time.monotonic()
            while not (r/'policy'/f'service{w}/ready.json').exists():
                if not owned(rec) or time.monotonic()-start>600:raise RuntimeError(f'service{w} unavailable')
                time.sleep(2)
        event(r,'services_ready',workers=2)
        m=read(r/'run-manifest.json');m.update(updated_at=now(),status='running');write(r/'run-manifest.json',m)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,r,w) for w in range(2)]
            for f in futures:f.result()
        event(r,'ordered_scientific_intake_closed')
    finally:
        # Scientific subprocesses have isolated sessions and must finish before services.
        if running:event(r,'healthy_episodes_preserved_on_coordinator_error',workers=list(running))
        else:
            for w,(p,rec,f) in enumerate(services):
                if owned(rec):
                    try:
                        c=http.client.HTTPConnection('127.0.0.1',PORTS[w],timeout=10);c.request('POST','/finish-declared-units',body=b'{}');c.getresponse().read();c.close()
                        start=time.monotonic()
                        while owned(rec) and time.monotonic()-start<30:time.sleep(1)
                    except Exception:event(r,'service_close_interface_error',worker=w,error=traceback.format_exc())
                    if owned(rec):os.kill(rec['pid'],15)
                if f:f.close()
        stop.set();heartbeat_done.set();thread.join(timeout=2)
        final=summary(r);write(r/'phase-state.json',dict(updated_at=now(),phase='execution_closed',status='awaiting_agent_delivery',summary=final,ended_at=now()))
        m=read(r/'run-manifest.json');m.update(updated_at=now(),ended_at=now(),status='execution_closed_pending_delivery',actual_counts=final);write(r/'run-manifest.json',m)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
