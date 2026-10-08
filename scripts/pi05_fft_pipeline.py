"""Manual-launched S2 -> bounded baseline -> one main fit -> dev selection.

S4 onward is resumed by the agent from the concrete frozen policy receipt.
No old run queue is recovered and no valid failure is retried.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
import http.client
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import threading

R=Path('/share/personal/chensiyu/haokaijiang/MobiWAM')
GPUS=['GPU-e8903903-cd05-0169-aead-361aba1b99d2','GPU-d1bd8de6-c581-df57-451d-10e3870d66fc',
      'GPU-aabcbd67-5c6a-0e96-3f9d-0881d087c276','GPU-a22aaeb2-6c91-39e3-a9f8-fcf86bacf63e']
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    tmp=p.with_name(p.name+'.tmp-'+str(threading.get_ident()));tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--fsdp-devices',choices=[2,4],type=int,required=True);a=ap.parse_args()
    run=a.run;op=run/'runtime/openpi';sim=run/'runtime/mobipi'
    manifest=read(run/'run-manifest.json');phase=read(run/'phase-state.json')
    deadline=datetime.fromisoformat(manifest['policy_deadline'])
    stop_starting=deadline-timedelta(seconds=2700)
    receipt=run/'launch/manual-launch-receipt.json'
    if receipt.exists():raise RuntimeError('this run already has a manual launch receipt; use stage recovery, never duplicate launch')
    if datetime.now(timezone.utc)>=stop_starting:raise RuntimeError('dev scheduling cutoff expired; no new S2/S3 unit')
    write(receipt,dict(at=now(),researcher_manual_shell_launch=True,fsdp_devices=a.fsdp_devices,pid=os.getpid(),
        argv=sys.argv,GPU_uuids=GPUS[:a.fsdp_devices],not_codex_auto_launch=True))
    cache=Path(manifest['approved_roots']['cache'])
    common=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        XDG_CACHE_HOME=str(cache/'xdg'),HF_HOME=str(cache/'huggingface'),TORCH_HOME=str(cache/'torch'),
        OPENPI_DATA_HOME=str(cache/'assets'),TMPDIR=str(cache/'tmp'))
    envop=dict(common,PYTHONPATH=f'{op}/src:{op}/packages/openpi-client/src',CUDA_VISIBLE_DEVICES=','.join(GPUS[:a.fsdp_devices]),
        XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(cache/'jax'))
    envsim=dict(common,PYTHONPATH=f'{sim}/src:{sim}/scripts:{sim}:{R}/Mobipi/external/robocasa:{R}/Mobipi/external/robomimic:{R}/Mobipi/external/mimicgen',
        MUJOCO_GL='egl',LD_LIBRARY_PATH=str(R/'env/lib'))
    jobs=[]
    def event(status,next_action,**kw):
        phase.update(read(run/'phase-state.json'))
        phase.update(updated_at=now(),agent_status='awaiting_agent_resume',job_status=status,next_action=next_action,**kw)
        write(run/'phase-state.json',phase);print(json.dumps(dict(at=now(),status=status,next_action=next_action)),flush=True)
    def start(cmd,env,log_name):
        log=run/'logs'/log_name;f=log.open('x');p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        rec=dict(at=now(),pid=p.pid,command=cmd,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH') if k in env},log=str(log))
        jobs.append(rec);write(run/'launch/owned-jobs.json',jobs);return p,f,rec
    def run_job(cmd,env,name):
        p,f,rec=start(cmd,env,name);code=p.wait();f.close();rec.update(ended_at=now(),exit_code=code)
        write(run/'launch/owned-jobs.json',jobs)
        if code:raise RuntimeError(f'own job failed exit={code}: {name}; inspect preserved failure; no unchanged retry')
    def train(fit_id,batch,steps,diagnostic,train_deadline,save_interval=1000):
        cmd=[str(R/'env/openpi/bin/python'),'-u',str(op/'scripts/pi05_fft_train.py'),'--run',str(run),'--fit-id',fit_id,
            '--fsdp-devices',str(a.fsdp_devices),'--batch',str(batch),'--steps',str(steps),'--deadline',train_deadline,
            '--save-interval',str(save_interval)]
        if diagnostic:cmd.append('--diagnostic')
        else:cmd+=['--candidate-steps',str(steps)]
        run_job(cmd,envop,fit_id+'.log');return read(run/'policy'/fit_id/'result.json')
    def develop(tag,checkpoint,lora,slots):
        if datetime.now(timezone.utc)>=stop_starting:raise RuntimeError('no new development after cutoff')
        roster=run/'policy'/(tag+'-roster.json');write(roster,dict(at=now(),slots=slots,policy_sampling_seed=20261008))
        configs=sorted({s['config_id'] for s in slots});all_results=[]
        workers=[]
        for i,cid in enumerate(configs):
            card=0 if i==0 else 3;port=8920+i
            jobs_here=[]
            for k,s in enumerate(slots):
                if s['config_id']!=cid:continue
                job=dict(slot=k+1,config_id=cid,parent_group=s['parent_group'],task=s['task'],route=s['route'],
                    purpose='policy-dev',evaluation_tag=tag+'-'+s['route'])
                if s.get('status')=='X':job['predeclared_X']='X_static_start_binding'
                jobs_here.append(job)
            queue=run/'policy'/(tag+f'-queue{i}.json')
            write(queue,dict(at=now(),jobs=jobs_here,roster=str(roster),expected_checkpoint=str(checkpoint),
                stop_starting_at=stop_starting.isoformat(),finish_own_service=True))
            service_env=dict(envop,CUDA_VISIBLE_DEVICES=GPUS[card])
            cmd=[str(R/'env/openpi/bin/python'),'-u',str(op/'scripts/pi05_fft_serve.py'),'--checkpoint',str(checkpoint),
                '--output',str(run/'policy'/(tag+f'-service{i}')),'--port',str(port),'--max-seconds',str(max(1,int((deadline-datetime.now(timezone.utc)).total_seconds())))]
            if lora:cmd.append('--lora')
            service=start(cmd,service_env,tag+f'-service{i}.log')
            collector_env=dict(envsim,CUDA_VISIBLE_DEVICES=GPUS[card],MUJOCO_EGL_DEVICE_ID=str(card))
            cmd=[str(R/'env/bin/python'),'-u',str(sim/'scripts/pi05_fft_collect.py'),'--run',str(run),'--queue',str(queue),
                '--port',str(port),'--checkpoint-step',checkpoint.name,'--status-name',tag+f'-worker{i}']
            collector=start(cmd,collector_env,tag+f'-worker{i}.log');workers.append((collector,service))
        for (p,f,rec),(sp,sf,srec) in workers:
            code=p.wait();f.close();rec.update(ended_at=now(),exit_code=code)
            # Service normally receives finish-declared-units from its own collector.
            # Failure exits are bounded by the service's own deadline; unknown PIDs untouched.
            if code:raise RuntimeError('development worker mechanical failure; saved unknown; no automatic repeat')
            sp.wait();sf.close();srec.update(ended_at=now(),exit_code=sp.returncode)
        write(run/'launch/owned-jobs.json',jobs)
        receipts=[]
        for k,s in enumerate(slots):
            folder=run/'episodes'/(tag+'-'+s['route'])/f'slot-{k+1:02d}-{s["config_id"]}'/'engineering-attempt-0'
            cp=folder/'completed.json'
            if cp.exists():receipts.append(cp)
            else:all_results.append(dict(config_id=s['config_id'],route=s['route'],status='X' if s.get('status')=='X' else 'unknown_or_unrun'))
        def audit(cp):
            q=read(cp);target=Path(q['attempt'])/'sprint-safety-audit.json'
            if not target.exists():run_job([str(R/'env/bin/python'),'-u',str(sim/'scripts/sim_sprint_safety.py'),'--receipt',str(cp)],
                dict(envsim,MUJOCO_EGL_DEVICE_ID='8'),tag+'-'+str(cp.parent.parent.name)+'-'+str(cp.parent.parent.parent.name)+'-audit.log')
            return dict(receipt=str(cp),config_id=q['config_id'],task=q['task'],route=q['route'],status=q['status'],
                native_success=q['native_success'],safety_pass=read(target).get('all_safety_pass'),started_at=q['started_at'],ended_at=q['ended_at'])
        with ThreadPoolExecutor(max_workers=2) as pool:all_results+=list(pool.map(audit,receipts))
        write(run/'policy'/(tag+'-development-results.json'),dict(at=now(),slots=all_results,complete=len(all_results)==6,
            safety_complete=all(x.get('safety_pass') is not None or x['status']=='X' for x in all_results)))
        phase.update(read(run/'phase-state.json'))
        phase['counters']['development']=sum(1 for p in (run/'episodes').glob('*/slot-*/engineering-attempt-*/completed.json')
            if not read(p)['status'].startswith('X_'))
        write(run/'phase-state.json',phase)
        return all_results
    try:
        event('S2_diagnostic','Run selected topology100-update diagnostic; do not count as main fit',phase='S2')
        diagnostic_deadline=min(deadline-timedelta(seconds=19800),datetime.now(timezone.utc)+timedelta(seconds=5400))
        if diagnostic_deadline<=datetime.now(timezone.utc):raise RuntimeError('not enough time for diagnostic and complete candidate development reserve')
        diagnostic_used=0
        for batch in (32,16,8):
            fit_id='diagnostic100' if batch==32 else f'diagnostic-repair-b{batch}'
            try:
                result=train(fit_id,batch,100-diagnostic_used,True,diagnostic_deadline.isoformat(),100)
                diagnostic_used+=result['actual_updates']
                break
            except RuntimeError:
                failure_path=run/'policy'/fit_id/'failure.json'
                if not failure_path.exists():raise
                failure=read(failure_path);diagnostic_used+=failure['actual_updates']
                text=failure['traceback'].lower()
                if not ('out of memory' in text or 'resource_exhausted' in text):raise
                if not (run/'policy'/fit_id/'trainable-and-fsdp.json').exists():raise RuntimeError('OOM before real FSDP/optimizer audit; needs targeted repair, no blind batch retry')
                if batch==8 or diagnostic_used>=100:raise
                write(run/'policy'/(fit_id+'-oom-repair.json'),dict(at=now(),actual_updates=diagnostic_used,
                    next_batch=batch//2,FSDP_optimizer_sharding='verified before failure',remat='official Gemma/SigLIP nn.remat present',
                    LR_unchanged=True,same_error_retry=False,main_fit_started=False))
        phase['counters']['diagnostic_updates']=diagnostic_used
        if result['status']!='passed':raise RuntimeError('diagnostic incomplete; no main fit')
        event('S2_old_baseline','Execute exactly6 frozen120s old-fit2 development slots and audit them')
        slots=read(run/'policy/policy-dev-roster.json')['slots']
        baseline=develop('old-fit2',R/'checkpoints/obc-pi05-v1/20261006T181000Z-query-relative-fit2/2000',True,slots)
        if not all('receipt' in x or x['status']=='X' for x in baseline):raise RuntimeError('baseline unknown; input resource measurement incomplete')
        binding=read(run/'inventory/dataset-binding.json');N=binding['valid_windows'];batch=result['batch_global']
        # One candidate is a resource decision before mainfit, never a score decision.
        #12 remaining episodes reserve the2700s cap across two verified workers.
        reserve_seconds=12*2700/2+1800
        remaining=(deadline-datetime.now(timezone.utc)).total_seconds()-reserve_seconds
        stable=max(result['p95_step_seconds'] or 1,result['median_step_seconds'] or 1)+(result['p95_loader_seconds'] or 0)
        main_seconds=min(remaining,6600.)
        S=min(int(max(0,main_seconds-600)/(stable*1.15)),int(np_ceil(20*N/batch)))
        if S<2:raise RuntimeError('conservative reserve leaves no complete main-fit candidate budget')
        save_interval=max(1,min(S,int(1800/stable),int(N/batch)))
        freeze=dict(at=now(),N=N,batch_global=batch,S=S,target_exposure=S*batch/N,target20_exposures=20,
            main_fit_count=1,official_base_reset=True,candidate_steps=[S],candidate_count=1,
            measured_step_plus_loader_seconds=stable,main_compile_reload_reserve_seconds=600,
            candidate_rationale='before main fit, limited to one complete six-slot candidate by absolute deadline; no dev score used',
            warmup=max(1,int(.05*S)),lr_peak=2e-5,lr_end=2e-6,save_interval=save_interval,
            remaining_development_reserve_seconds=reserve_seconds,unit_wall_cap_seconds=2700,workers=2,
            main_deadline=(datetime.now(timezone.utc)+timedelta(seconds=main_seconds)).isoformat(),policy_deadline=deadline.isoformat(),
            diagnostic_updates_excluded=result['actual_updates'])
        write(run/'policy/training-freeze.json',freeze)
        event('S3_main_fit','Run the unique official-base reset full fit then its complete6 candidate slots',phase='S3')
        fitted=train('main-fit1',batch,S,False,freeze['main_deadline'],save_interval)
        phase['counters']['main_fit_updates']=fitted['actual_updates']
        if fitted['actual_updates']!=S:raise RuntimeError('predeclared candidate not reached; no fallback toold fit2')
        candidate=develop('full-final',Path(fitted['checkpoint']),False,slots)
        complete=len(candidate)==6 and all(('receipt' in x and x.get('safety_pass') is not None and x['status']!='compute-timeout') or x['status']=='X' for x in candidate)
        if not complete:raise RuntimeError('selection_incomplete: current candidate lacks same-version complete6 terminal/safety slots')
        selected=dict(at=now(),checkpoint=fitted['checkpoint'],candidate='full-final',candidate_count=1,
            selection_status='fixed_single_complete_candidate',winner_claim=False,common_six=candidate,
            tasks={task:dict(ready=any(x.get('native_success') and x.get('safety_pass') for x in candidate if x.get('task')==task))
                for task in ['CloseDrawer','CloseSingleDoor']})
        write(run/'policy/selection-receipt.json',selected)
        design=read(run/'design/start-design.json')['selected'];fast={s['parent_group'] for s in slots}
        complement=[dict(r,route=route) for r in design if r['role']=='development' and r['parent_group'] not in fast for route in 'EDA']
        additional=develop('selected-complement',Path(fitted['checkpoint']),False,complement)
        selected['tasks']={task:dict(ready=any(x.get('native_success') and x.get('safety_pass') and x['status']!='compute-timeout'
            for x in candidate+additional if x.get('task')==task)) for task in ['CloseDrawer','CloseSingleDoor']}
        selected.update(frozen_at=now(),component_code_commit=subprocess.check_output(['git','-C',str(sim),'rev-parse','HEAD'],text=True).strip(),
            openpi_code_commit=subprocess.check_output(['git','-C',str(op),'rev-parse','HEAD'],text=True).strip())
        write(run/'policy/component-freeze.json',selected)
        event('S3_frozen','Agent resumes S4 only for task with safe development success; preserve72 denominator',phase='S3_complete',
            tasks=selected['tasks'])
    except Exception:
        write(run/'launch/pipeline-failure.json',dict(at=now(),traceback=traceback.format_exc(),preserve_valid_failures=True))
        event('stopped_with_retained_failure','Agent inspects exact failure; only authorized implementation repair or independent downstream preparation')
        traceback.print_exc();raise


def np_ceil(x):
    import math
    return math.ceil(x)


if __name__=='__main__':main()
