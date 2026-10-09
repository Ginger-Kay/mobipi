"""Fixed48 drawer OBC v2 coordinator, persistent atomic ordered state ledger."""
import argparse, csv, fcntl, http.client, json, os, socket, subprocess, sys, threading, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pi05_drawer_pipeline import R, GPUS, CP, simenv, read, write, now
from pi05_drawer_data_v2 import launch, owned, await_process, terminal, SCIENTIFIC
from pi05_drawer_obc_fit_v2 import dataset, fit, available, freeze_state
PORTS=[8990,8991]
lock=threading.RLock();stop=threading.Event();done=threading.Event();running={};storage_reservations={}


def event(r,name,**kw):
    with lock:
        with (r/'events.jsonl').open('a') as f:f.write(json.dumps(dict(at=now(),event=name,**kw))+'\n')


def update(r,slot=None,state=None,**values):
    with lock:
        d=read(r/'slot-ledger.json')
        if slot is not None:d['slots'][slot-1].update(values)
        if state is not None:d['states'][state-1].update(values)
        d['updated_at']=now();write(r/'slot-ledger.json',d)


def snapshot(r):
    with lock:
        d=read(r/'slot-ledger.json');rows=d['slots'];statuses=['completed','unknown','X','unavailable','unrun','running'];counts={k:sum(a['status']==k for a in rows) for k in statuses};assert sum(counts.values())==48
        counts.update(plan=48,success=sum(a['success']==1 for a in rows),failure=sum(a['success']==0 for a in rows));assert counts['completed']==counts['success']+counts['failure']
        methods=[]
        for m in ['Fixed-D','OBC-MLP','Fixed-E','Fixed-A']:
            rs=[a for a in rows if a['method']==m];n=sum(a['status']=='completed' for a in rs);s=sum(a['success']==1 for a in rs)
            methods.append(dict(method=m,plan=12,completed=n,success=s,failure=n-s,success_per_completed=s/n if n else None,coverage_per12=n/12,**{k:sum(a['status']==k for a in rs) for k in statuses if k!='completed'},fallback=sum(bool(a.get('selector_fallback')) for a in rs if a['status']=='completed')))
        pairs=[]
        for st in d['states']:
            rr={a['method']:a for a in rows if a['state_order']==st['state_order']}
            a,b=rr['OBC-MLP'],rr['Fixed-D']
            if a['status']=='completed' and b['status']=='completed':
                pair=dict(state_order=st['state_order'],parent=st['environment_seed'],offset=st['offset_id'],OBC=a['success'],D=b['success'],success_difference=a['success']-b['success'],both_success=a['success']==b['success']==1)
                for key in ['path_m','terminal_duration_s']:pair['delta_'+key]=a.get(key)-b.get(key) if pair['both_success'] and a.get(key) is not None and b.get(key) is not None else None
                pairs.append(pair)
        perparent=[]
        for p in [142,143]:
            rr=[a for a in pairs if a['parent']==p]
            if rr:perparent.append(dict(parent=p,n=len(rr),success_difference=sum(a['success_difference'] for a in rr)/len(rr)))
        paired=dict(n=len(pairs),independent_parents=len(perparent),parent_equal_success_difference=sum(a['success_difference'] for a in perparent)/len(perparent) if perparent else None,per_parent=perparent,OBC_win=sum(a['OBC']==1 and a['D']==0 for a in pairs),D_win=sum(a['OBC']==0 and a['D']==1 for a in pairs),both_success=sum(a['OBC']==a['D']==1 for a in pairs),both_failure=sum(a['OBC']==a['D']==0 for a in pairs))
        value=dict(at=now(),counts=counts,methods=methods,OBC_D=paired,safety_qualified=False,review_status='pending')
        write(r/'tables/summary.json',value)
        for name,rs,fields in [('T1-methods.csv',methods,list(methods[0])),('T1-OBC-D-pairs.csv',pairs,['state_order','parent','offset','OBC','D','success_difference','both_success','delta_path_m','delta_terminal_duration_s']),('T3-state-method-matrix.csv',rows,['slot_order','state_order','parent','offset','method','route','status','success','attempt','selector_fallback','path_m','terminal_duration_s','receipt','target_video','panoramic_video','trajectory','initial_state','initial_RGB','media_closed'])]:
            path=r/'tables'/name;tmp=path.with_suffix('.tmp')
            with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rs)
            tmp.replace(path)
        return value


def eta(r):
    q=read(r/'preflight/ETA.json');base=q['full120_execution_eta_s']
    for a in read(r/'slot-ledger.json')['slots']:
        if a['status']!='completed' or not a.get('receipt'):continue
        p=read(a['receipt'])
        if p.get('terminal_duration_s',0)>=119.9:base=max(base,(datetime.fromisoformat(p['ended_at'])-datetime.fromisoformat(p['started_at'])).total_seconds()+60)
    return base


def claim(r,w):
    with lock:
        d=read(r/'slot-ledger.json')
        for s in d['states']:
            if s['status']=='mechanical_blocked':
                repair=r/'launch'/f"state-{s['state_order']}-repair.json"
                if repair.exists():
                    q=read(repair)
                    if q.get('actual_repair_commit') and q.get('issue_signature') and q.get('description') and not s.get('repair_applied'):
                        used=d['issue_repairs'].get(q['issue_signature'],0)
                        if used<10:
                            d['issue_repairs'][q['issue_signature']]=used+1;s.update(status='unrun',claimed_by=None,repair_applied=str(repair))
            if s['status']=='unrun':
                s.update(status='claimed',claimed_by=w,claimed_at=now());write(r/'slot-ledger.json',d);event(r,'atomic_state_claim',state_order=s['state_order'],worker=w);return dict(s)
        return None


def command(r,name,cmd,w):
    p,rec,f=launch(r,name,cmd,simenv(r,w));running[w]=rec
    try:return await_process(p,rec,f)
    finally:running.pop(w,None)


def prepare_state(r,s,w):
    bp=r/'design/anchors'/s['config_id']/'binding.json'
    if not bp.exists():
        rc=command(r,'prepare-'+str(s['state_order']),[R/'env/bin/python','-B','-u',r/'runtime/mobipi/scripts/pi05_drawer_data_prepare.py','--run',r,'--state',s['state_order']],w)
        if not bp.exists():raise RuntimeError(f'prepare state failed rc={rc}; no scientific outcome; inspect task log')
    s=read(bp)
    if s['status']=='X':return s,None
    fp=r/'design/inputs'/s['config_id']/'features.json'
    if not fp.exists():
        roster=r/'design'/f"input-roster-{s['state_order']}.json";write(roster,dict(selected=[dict(s,tier=1)]))
        rc=command(r,'features-'+str(s['state_order']),[R/'env/bin/python','-B','-u',r/'runtime/mobipi/scripts/pi05_drawer_features.py','--run',r,'--roster',roster,'--freeze-output',f"design/features-state-{s['state_order']}.json"],w)
        if not fp.exists():raise RuntimeError(f'state feature unavailable rc={rc}; no dynamics used')
    with lock:
        roster=read(r/'design/episode-roster.json');roster['slots'][s['state_order']-1]=s;write(r/'design/episode-roster.json',roster)
    return s,freeze_state(r,s)


def execute(r,s,j,pred,w):
    if j['status'] in ['completed','unknown','X','unavailable']:return True
    route=pred['choices']['MLP']['route'] if j['method']=='OBC-MLP' else j['route']
    fallback=pred['choices']['MLP'].get('selector_fallback',False) if j['method']=='OBC-MLP' else False
    if route not in ('E','D','A') or not pred['valid_routes'].get(route,False):
        update(r,slot=j['slot_order'],status='unavailable',success=None,unavailable_reason=pred['choices']['MLP']['status'] if j['method']=='OBC-MLP' else 'fixed_route_hard_invalid');snapshot(r);return True
    with lock:
        deadline=read(r/'deadline-config.json');remaining=(datetime.fromisoformat(deadline['execution_deadline'])-datetime.now(timezone.utc)).total_seconds();estimate=eta(r);margin=deadline['save_reserve_seconds'];peak=read(r/'preflight/ETA.json')['full_episode_storage_peak_bytes'];free=os.statvfs(r).f_bavail*os.statvfs(r).f_frsize;reserved=sum(storage_reservations.values());reason=None
        if remaining<estimate+margin:reason='full120ETA_plus_save_exceeds_remaining'
        elif free-reserved<peak+128*1024**2:reason='storage_peak_exceeds_unreserved_space'
        if reason:
            event(r,'ordered_slot_admission_closed',next_slot=j['slot_order'],method=j['method'],decision_time=now(),ETA_s=estimate,remaining_s=remaining,save_margin_s=margin,free_bytes=free,reserved_bytes=reserved,episode_peak_bytes=peak,reason=reason);stop.set();return False
        storage_reservations[w]=peak
    attempt=j['attempt'];tag='obc-v2-'+j['method'];out=r/'episodes'/tag/f"slot-{s['state_order']:02d}-{s['config_id']}"/f'engineering-attempt-{attempt}';receipt=out/'completed.json';cap=min(2700,int(remaining-margin));dp=r/'launch'/f"deadline-slot-{j['slot_order']}-attempt-{attempt}.json";write(dp,dict(deadline,wall_seconds=cap,canonical_config=str(r/'deadline-config.json')))
    cmd=[R/'env/bin/python','-B','-u',r/'runtime/mobipi/scripts/pi05_harness_episode.py','--run',r,'--slot',s['state_order'],'--route',route,'--checkpoint-step',2000,'--port',PORTS[w],'--adapter-version','v6','--evaluation-tag',tag,'--roster',r/'design/episode-roster.json','--purpose','online' if j['method']=='OBC-MLP' else 'paired','--A-private-version','A3N' if route=='A' else 'A1','--sim-seconds',120,'--wall-seconds',cap,'--attempt',attempt,'--deadline-config',dp,'--experiment-id','MMWAM-OBC-002-PI05-DRAWER-OBC-v2']
    try:
        if not receipt.exists():
            p,rec,f=launch(r,f"episode-{j['slot_order']}-attempt-{attempt}",cmd,simenv(r,w));running[w]=rec;update(r,slot=j['slot_order'],status='running',worker=w,pid=rec['pid'],started_at=now(),route=route,selector_fallback=fallback,receipt=str(receipt),initial_state=s['initial_state'],initial_RGB=s['initial_RGB'],prediction_freeze=str(r/'design/predictions'/s['config_id']/'freeze.json'));event(r,'slot_start',slot=j['slot_order'],method=j['method'],route=route,ETA_s=estimate,wall_cap_s=cap,worker=w);rc=await_process(p,rec,f);running.pop(w,None);event(r,'slot_process_closed',slot=j['slot_order'],exit_code=rc)
        q=read(receipt) if receipt.exists() else {};status,label=terminal(q)
        if status=='unknown':
            results=list(out.glob('*/'+route+'/attempt-*/result.json'))
            if len(results)==1:
                raw_result=read(results[0])
                if raw_result.get('reason') in SCIENTIFIC and 'checker_success' in raw_result:
                    q=dict(raw_result,status=raw_result['reason'],native_success=raw_result['checker_success'],attempt=str(results[0].parent),terminal_receipt_recovered_without_reexecution=True)
                    receipt=out/'recovered-terminal.json';write(receipt,q);status,label=terminal(q);update(r,slot=j['slot_order'],receipt=str(receipt))
        identity=read(out/'process.json') if (out/'process.json').exists() else {}
        if identity.get('config_id')!=s['config_id'] or identity.get('route')!=route or identity.get('environment_seed')!=s['environment_seed']:status,label='unknown',None
        if status=='completed':
            raw=Path(q['attempt']);refs={k:str(raw/n) for k,n in [('target_video','original.mp4'),('panoramic_video','panoramic.mp4'),('trajectory','demo.hdf5')]};update(r,slot=j['slot_order'],status='completed',success=label,ended_at=now(),media_closed=all(Path(v).exists() and Path(v).stat().st_size>0 for v in refs.values()),path_m=q.get('actual_base_path_m'),terminal_duration_s=q.get('terminal_duration_s'),progress=q.get('native_progress'),**refs)
        else:update(r,slot=j['slot_order'],status='unknown',success=None,ended_at=now(),mechanical_condition='no reliable bound scientific terminal; actual repair required')
        snapshot(r);event(r,'slot_saved',slot=j['slot_order'],status=status,success=label)
        if status=='unknown':raise RuntimeError(f'slot{j["slot_order"]} mechanical unknown; preserve prefix and require actual repair')
        return True
    finally:
        with lock:storage_reservations.pop(w,None)


def worker(r,w):
    while not stop.is_set():
        s=claim(r,w)
        if s is None:return
        try:
            s,pred=prepare_state(r,s,w);jobs=[j for j in read(r/'slot-ledger.json')['slots'] if j['state_order']==s['state_order']]
            if s['status']=='X':
                for j in jobs:update(r,slot=j['slot_order'],status='X',success=None)
                update(r,state=s['state_order'],status='X');snapshot(r);continue
            for j in jobs:
                if stop.is_set():return
                if not execute(r,s,j,pred,w):return
            update(r,state=s['state_order'],status='disposed',disposed_at=now())
        except Exception:
            error=traceback.format_exc();update(r,state=s['state_order'],status='mechanical_blocked',claimed_by=None,mechanical_error=error,recovery_condition='actual code/environment fix commit and launch/state-N-repair.json; original priority retained');event(r,'state_mechanical_isolated',state_order=s['state_order'],error=error);snapshot(r)


def heartbeat(r):
    previous=None;last=time.monotonic()
    while not done.wait(60):
        progress={str(w):dict(pid=a['pid'],alive=owned(a),log_bytes=Path(a['log']).stat().st_size if Path(a['log']).exists() else 0) for w,a in list(running.items())}
        if progress!=previous:previous=progress;last=time.monotonic()
        write(r/'phase-state.json',dict(at=now(),phase='evaluating',heartbeat=now(),workers=progress,seconds_without_progress=time.monotonic()-last,summary=snapshot(r)))
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu','--format=csv,noheader'],text=True);contexts=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True);write(r/'operations/live-gpu.json',dict(at=now(),GPU=gpu,contexts=contexts,experiment_evidence=False))
        if time.monotonic()-last>=600:event(r,'bounded_stall_inspection_required',workers=progress)
        if datetime.now(timezone.utc)>=datetime.fromisoformat(read(r/'deadline-config.json')['execution_deadline']):stop.set()


def main(r):
    lease=(r/'launch/coordinator.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(r/'launch/coordinator-process.json',dict(at=now(),pid=os.getpid(),command=sys.argv,run_id=r.name))
    if not (r/'training/recipe.json').exists():dataset(r)
    for kind in ['MLP','Linear','ridge']:
        if not (r/'training'/kind/'completed.json').exists():
            if (r/'training'/kind).exists():raise RuntimeError('partial fit retained; actual implementation repair needed before another fit')
            try:fit(r,kind)
            except Exception:
                event(r,'fit_unavailable',model=kind,error=traceback.format_exc())
                write(r/'training'/f'{kind}-unavailable.json',dict(at=now(),error=traceback.format_exc(),status='unavailable_before_first_eval'))
    if not (r/'training/model-freeze.json').exists():available(r)
    services=[];thread=threading.Thread(target=heartbeat,args=(r,),daemon=True);thread.start()
    try:
        for w in range(2):
            op=r/'runtime/openpi';env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=f'{op}/src:{op}/scripts:{op}/packages/openpi-client/src',CUDA_VISIBLE_DEVICES=GPUS[w],JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(r/'policy'/f'jax-cache{w}'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1');remaining=int((datetime.fromisoformat(read(r/'deadline-config.json')['execution_deadline'])-datetime.now(timezone.utc)).total_seconds())
            with socket.socket() as sock:sock.bind(('127.0.0.1',PORTS[w]))
            cmd=[R/'env/openpi/bin/python','-B','-u',op/'scripts/pi05_fft_serve.py','--lora','--checkpoint',CP,'--output',r/'policy'/f'service{w}','--port',PORTS[w],'--max-seconds',remaining,'--max-queries',100000];services.append(launch(r,f'service{w}',cmd,env))
        for w,(p,rec,f) in enumerate(services):
            start=time.monotonic()
            while not (r/'policy'/f'service{w}/ready.json').exists():
                if not owned(rec) or time.monotonic()-start>600:raise RuntimeError(f'service{w} unavailable; actual repair required')
                time.sleep(2)
        event(r,'services_ready',workers=2);write(r/'phase-state.json',dict(at=now(),phase='evaluating',model_freeze=str(r/'training/model-freeze.json')))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,r,w) for w in range(2)]
            for f in futures:f.result()
    finally:
        if running:event(r,'healthy_workers_preserved',workers=list(running))
        else:
            for w,(p,rec,f) in enumerate(services):
                if owned(rec):
                    try:
                        conn=http.client.HTTPConnection('127.0.0.1',PORTS[w],timeout=10);conn.request('POST','/finish-declared-units',body=b'{}');conn.getresponse().read();conn.close()
                        start=time.monotonic()
                        while owned(rec) and time.monotonic()-start<30:time.sleep(1)
                    except Exception:event(r,'service_close_interface_error',worker=w,error=traceback.format_exc())
                    if owned(rec):os.kill(rec['pid'],15)
                if f:f.close()
        done.set();thread.join(timeout=2);write(r/'phase-state.json',dict(at=now(),phase='execution_closed',summary=snapshot(r)));event(r,'scientific_execution_closed')


def guard(r):
    write(r/'launch/deadline-guard-process.json',dict(at=now(),pid=os.getpid(),command=sys.argv,run_id=r.name))
    while True:
        deadline=datetime.fromisoformat(read(r/'deadline-config.json')['execution_deadline'])
        if (r/'phase-state.json').exists() and read(r/'phase-state.json').get('phase')=='execution_closed':return
        if datetime.now(timezone.utc)>=deadline-timedelta(seconds=30):break
        time.sleep(2)
    records=[]
    for path in (r/'launch').glob('*-process.json'):
        rec=read(path)
        if path.name.startswith(('episode-','prepare-','features-','service')) and rec.get('run_id')==r.name and owned(rec) and any(str(r) in a for a in rec['command']):records.append(rec)
    for rec in records:
        if owned(rec):os.kill(rec['pid'],2)
    event(r,'deadline_save_signal',PIDs=[q['pid'] for q in records],deadline=deadline.isoformat())
    until=deadline-timedelta(seconds=5)
    while datetime.now(timezone.utc)<until and any(owned(q) for q in records):time.sleep(1)
    for rec in records:
        if owned(rec):os.kill(rec['pid'],15)
    write(r/'delivery/deadline-guard-receipt.json',dict(at=now(),deadline=deadline.isoformat(),only_verified_owned_PIDs=[q['pid'] for q in records],unknown_not_zero=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--guard',action='store_true');a=p.parse_args();(guard if a.guard else main)(a.run)
