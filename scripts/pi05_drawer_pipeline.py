"""Contract-bound drawer dispatch: frozen sources, two isolated collectors, fixed heads."""
import argparse, csv, hashlib, http.client, json, os, socket, subprocess, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
R=Path('/share/personal/chensiyu/haokaijiang/MobiWAM')
GPUS=['GPU-e8903903-cd05-0169-aead-361aba1b99d2','GPU-a22aaeb2-6c91-39e3-a9f8-fcf86bacf63e']
EGL=[0,3];PORTS=[8960,8961]
CP=R/'checkpoints/obc-pi05-v1/20261006T181000Z-query-relative-fit2/2000'
DEV=datetime.fromisoformat('2026-10-08T16:00:00+00:00');OBC=datetime.fromisoformat('2026-10-09T00:00:00+00:00');FINAL=datetime.fromisoformat('2026-10-09T04:00:00+00:00')
lock=threading.RLock()
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp-'+str(threading.get_ident()));t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def clean_json(d):
    if isinstance(d,dict): return {k:clean_json(v) for k,v in d.items()}
    if isinstance(d,(list,tuple)):return [clean_json(x) for x in d]
    if isinstance(d,np.generic):return clean_json(d.item())
    if isinstance(d,float) and not np.isfinite(d):return None
    return d

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def phase(r,name,**extra):
    with lock:
        q=read(r/'phase-state.json');q.update(updated_at=now(),phase=name,**extra);write(r/'phase-state.json',q)

def simenv(r,worker=0):
    c=r/'runtime/mobipi'
    return dict(os.environ,PYTHONNOUSERSITE='1',PYTHONPATH=f'{c}/src:{c}/scripts:{c}:{R}/Mobipi/external/robocasa:{R}/Mobipi/external/robomimic:{R}/Mobipi/external/mimicgen',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(EGL[worker]),CUDA_VISIBLE_DEVICES=GPUS[worker],LD_LIBRARY_PATH=str(R/'env/lib'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')

def run_command(r,name,cmd,env,cwd=None):
    log=r/'logs'/f'{name}.log'
    with log.open('x') as f:
        p=subprocess.Popen([str(x) for x in cmd],env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        write(r/'launch'/f'{name}-process.json',dict(at=now(),pid=p.pid,command=[str(x) for x in cmd],log=str(log),python=cmd[0],environment={k:env.get(k) for k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID']},status='running'))
        rc=p.wait()
    write(r/'launch'/f'{name}-exit.json',dict(at=now(),exit_code=rc));return rc

def outcome(job):
    p=Path(job.get('receipt','/nonexistent'));q=read(p) if p.exists() else {};af=Path(q['attempt'])/'sprint-safety-audit.json' if q else Path('/nonexistent');audit=read(af) if af.exists() else {}
    native=Path(q['attempt'])/'formal-native-substeps-receipt.json' if q else Path('/nonexistent')
    collision=audit.get('native_collision')
    if collision is None and native.exists(): collision=read(native).get('forbidden_contact') is not None
    failure=read(p.parent/'failure.json') if (p.parent/'failure.json').exists() else {}
    executed=bool((q and q.get('usable_scientific_outcome') and not q['status'].startswith('X_')) or failure.get('usable_outcome'))
    known=bool(q) and executed and q['status']!='compute-timeout' and not q['status'].startswith('engineering_')
    y=[None]*5;mask=[False]*5
    if known:y[0]=float(q['native_success']);mask[0]=True
    if executed and q:
        for j,key in [(2,'terminal_native_opening'),(3,'actual_base_path_m'),(4,'terminal_duration_s')]:
            v=q.get(key)
            if v is not None and np.isfinite(v):y[j]=float(np.clip(1-v,0,1)) if j==2 else float(v);mask[j]=True
    if isinstance(collision,bool) and (executed or collision):y[1]=float(collision);mask[1]=True
    return dict(job,status=q.get('status','engineering_unknown_prefix' if failure.get('usable_outcome') else job.get('status','unrun')),executed=executed,valid=known,unknown=bool(executed and not known),native_success=bool(q['native_success']) if known else None,safety_pass=audit.get('all_safety_pass'),safe_success=bool(q['native_success'] and audit['all_safety_pass']) if known and audit.get('all_safety_pass') is not None else False if known and not q['native_success'] else None,collision=collision,initial_opening=q.get('initial_native_opening'),terminal_opening=q.get('terminal_native_opening'),progress=y[2],path_m=y[3],terminal_duration_s=y[4],censored=q.get('censored'),head_values=y,head_masks=mask,route_semantics=q.get('route_semantics_pass'),audit=str(af) if af.exists() else None,steps=q.get('steps',0),policy_queries=q.get('policy_queries',0),target_video=str(Path(q['attempt'])/'original.mp4') if q else None,panoramic_video=str(Path(q['attempt'])/'panoramic.mp4') if q else None)

def all_jobs(r):return read(r/'design/selected-plan.json')['jobs']
def ledger(r):
    rows=[outcome(j) for j in all_jobs(r)];write(r/'evidence/slot-ledger.json',dict(at=now(),rows=rows));return rows

def freeze_resource(r):
    design=read(r/'design/start-design.json');features=read(r/'design/features-freeze.json');byid={x['config_id']:x for x in features['records']}
    configs=design['selected']
    for c in configs:
        f=byid.get(c['config_id'],{'routes':[]});v={x['route_family']:bool(x['hard_valid']) for x in f['routes']};c['hard_valid_routes']={t:v.get(t,False) for t in 'EDA'}
        if c['status']=='bound':assert c['hard_valid_routes']==read(Path(c['source']).parent/'binding.json')['hard_valid_routes']
    old=R/'artifacts/MMWAM-OBC-002-PI05-FFT/v1/20261008T070908Z-pi05-fft-anchor120';walls=[];starts=[];ends=[]
    for p in (old/'logs').glob('old-fit2-worker*.log'):
        for line in p.read_text().splitlines():
            d=json.loads(line);st=datetime.fromisoformat(d['started_at']);en=datetime.fromisoformat(d['ended_at']);walls.append((en-st).total_seconds());starts.append(st);ends.append(en)
    speedup=min(1.35,sum(walls)/(max(ends)-min(starts)).total_seconds());upper=max(walls);nowdt=datetime.now(timezone.utc);available=(FINAL-nowdt).total_seconds()
    # Actual six-slot concurrency informs a conservative rate, never assume2x.
    estimates={str(k):dict(new_slots=21+12*k,collection_s=(21+12*k)*upper/speedup,train_dev_s=21*upper/speedup,setup_obc_and_storage_s=1800,final_audit_reserve_s=7200,total_s=(21+12*k)*upper/speedup+9000) for k in [1,2,3]}
    eligible=[k for k in [1,2,3] if estimates[str(k)]['total_s']<=available and estimates[str(k)]['train_dev_s']+900<(OBC-nowdt).total_seconds()]
    if not eligible:
        write(r/'design/resource-freeze.json',dict(at=now(),status='k1_infeasible_no_science',available_seconds=available,estimates=estimates,old_wall_seconds=walls,measured_speedup_used=speedup));raise RuntimeError('k1 cannot fit complete core comparison; contract prohibits launch')
    k=max(eligible);selected=[c for c in configs if c['role']!='evaluation' or c['start_index']<k]
    jobs=[];reuse=read(r/'preflight/dev-reuse-receipt.json')
    for c in selected:
        for route in 'EDA':
            tag=('dev' if c['role']=='development' else 'train' if c['role']=='train' else 'eval-primary')+'-'+route
            j=dict(config_id=c['config_id'],parent_group=c['parent_group'],parent_config_id=c.get('parent_config_id',c['config_id']),environment_seed=c['environment_seed'],split=c['role'],start_index=c['start_index'],start_type=c['start_type'],route=route,method='Fixed-'+route,source='primary',slot=c['slot'],tag=tag,hard_valid=c['hard_valid_routes'][route],status='unrun' if c['hard_valid_routes'][route] else 'X_static_or_route_invalid')
            j['receipt']=str(r/'episodes'/tag/f'slot-{c["slot"]:02d}-{c["config_id"]}'/'engineering-attempt-0/completed.json')
            if c['role']=='development' and c['environment_seed']==109:
                reused=next(x for x in reuse['rows'] if x['route']==route);assert reused['compatible'];j.update(receipt=reused['receipt'],status='reused',source='inherited_dev',reused=True)
            jobs.append(j)
        if c['role']=='evaluation':
            for method in ['MLP','geometry','train-best-fixed']:
                tag='eval-online-'+method
                jobs.append(dict(config_id=c['config_id'],parent_group=c['parent_group'],environment_seed=c['environment_seed'],split='evaluation',start_index=c['start_index'],start_type=c['start_type'],route=None,method=method,source='fresh_online',slot=c['slot'],tag=tag,hard_valid=None,status='unrun_prediction_pending',receipt=str(r/'episodes'/tag/f'slot-{c["slot"]:02d}-{c["config_id"]}'/'engineering-attempt-0/completed.json')))
    write(r/'design/selected-plan.json',dict(at=now(),k=k,target_plan=60,selected_plan=24+12*k,new_science_upper=21+12*k,configurations=selected,jobs=jobs,schedule='train parent order E/D/A interleaved newdev; eval s0 both parents, then s1 both, then s2 both; parent cyclic primary/online order'))
    write(r/'design/episode-roster.json',dict(at=now(),slots=configs))
    write(r/'design/resource-freeze.json',dict(at=now(),status='frozen_before_any_new_science',k=k,available_seconds=available,estimates=estimates,old_wall_seconds=walls,measured_speedup_used=speedup,wall_estimate_uses_both_tasks=True,live_sample=str(r/'preflight/live-allocation-samples.json'),absolute_deadlines_unchanged=True,all_online_reserved=True,static_X_not_replaced=True,outcome_used_for_k=False))
    files=['scripts/pi05_harness_episode.py','scripts/reference_executor.py','src/mobiwam/pi05_adapter.py','src/mobiwam/pi05_motion.py','src/mobiwam/pi05_route.py','src/mobiwam/pi05_A3.py','src/mobiwam/pi05_natural_A.py','src/mobiwam/reference_collision.py','src/mobiwam/reference_prefix_safety.py','src/mobiwam/reference_formal_substep.py','scripts/teleop_reference.py','scripts/human_scene_pilot.py','scripts/pi05_candidate_features.py','src/mobiwam/task_video_identity.py']
    code=r/'runtime/mobipi';assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
    write(r/'policy/main-component-freeze.json',dict(at=now(),checkpoint=str(CP),auxiliary={str(CP/n):sha(CP/n) for n in ['custom-norm-stats.json','policy-adapter-config.json']},code_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),openpi_commit=subprocess.check_output(['git','-C',str(r/'runtime/openpi'),'rev-parse','HEAD'],text=True).strip(),behavior={n:sha(code/n) for n in files},input_files={str(p):sha(p) for f in features['records'] for p in [r/'design/inputs'/f['config_id']/'features.json',r/'design/inputs'/f['config_id']/'pre-outcome-RGB-sensors.npz',Path(f['source'])/'integration.npy']},installed_controller={str(p):sha(p) for p in (R/'env/lib/python3.10/site-packages/robosuite/controllers').glob('parts/mobile_base/*.py')},native_checker={str(p):sha(p) for p in [R/'Mobipi/external/robocasa/robocasa/environments/kitchen/single_stage/kitchen_drawer.py',R/'Mobipi/external/robocasa/robocasa/models/fixtures/cabinets.py']},sampling_seed=20261008,adapter='v6',horizon10_execute5_flow10_Hz20=True,sim_seconds=120,wall_seconds=2700,old_dev_reuse=str(r/'preflight/dev-reuse-receipt.json')))
    ledger(r);return selected

def verify_frozen(r):
    q=read(r/'policy/main-component-freeze.json')
    for n,v in q['behavior'].items():assert sha(r/'runtime/mobipi'/n)==v
    for key in ['auxiliary','installed_controller','native_checker']:
        for n,v in q[key].items():assert sha(n)==v

def execute_job(r,job,worker):
    if job.get('reused') or not job.get('hard_valid'):return job
    limit=DEV if job['split']=='development' else OBC if job['split']=='train' else FINAL
    reserve=300 if job['split']=='development' else 1800 if job['split']=='train' else 600
    if datetime.now(timezone.utc)>=limit-timedelta(seconds=2700+reserve):
        job['status']='unrun_deadline'
        with lock:
            plan=read(r/'design/selected-plan.json');next(x for x in plan['jobs'] if x['config_id']==job['config_id'] and x['tag']==job['tag']).update(job);write(r/'design/selected-plan.json',plan)
        return job
    verify_frozen(r);p=Path(job['receipt']);assert not p.parent.exists(),'scientific slot already exists; retry prohibited'
    code=r/'runtime/mobipi';cmd=[R/'env/bin/python','-u',code/'scripts/pi05_harness_episode.py','--run',r,'--slot',str(job['slot']),'--route',job['route'],'--checkpoint-step','2000','--port',str(PORTS[worker]),'--adapter-version','v6','--evaluation-tag',job['tag'],'--roster',r/'design/episode-roster.json','--purpose','online' if job['source']=='fresh_online' else 'policy-dev' if job['split']=='development' else 'paired','--A-private-version','A3N' if job['route']=='A' else 'A1','--sim-seconds','120','--wall-seconds','2700','--diagnostic-logging']
    name=f'{job["tag"]}-slot-{job["slot"]:02d}';rc=run_command(r,name,cmd,simenv(r,worker),code);job.update(exit_code=rc,status='completed' if p.exists() else 'mechanical_failure_unrepeated')
    with lock:
        plan=read(r/'design/selected-plan.json');target=next(x for x in plan['jobs'] if x['config_id']==job['config_id'] and x['tag']==job['tag']);target.update(job);write(r/'design/selected-plan.json',plan);rows=ledger(r)
        q=read(r/'phase-state.json');q['updated_at']=now();q['counters'].update(dev_reused=sum(x.get('reused',False) for x in rows),dev_new=sum(x['executed'] and x['split']=='development' and not x.get('reused') for x in rows),train=sum(x['executed'] and x['split']=='train' for x in rows),eval_primary=sum(x['executed'] and x['split']=='evaluation' and x['source']=='primary' for x in rows),online=sum(x['executed'] and x['source']=='fresh_online' for x in rows));write(r/'phase-state.json',q)
    print(json.dumps(dict(at=now(),config=job['config_id'],route=job['route'],method=job['method'],rc=rc,receipt=job['receipt'])),flush=True);return job

def audit_job(r,job):
    p=Path(job['receipt'])
    if not p.exists() or job.get('reused'):return
    q=read(p);af=Path(q['attempt'])/'sprint-safety-audit.json'
    if af.exists():return
    if not (Path(q['attempt'])/'formal-native-substeps.npz').exists():return
    return run_command(r,'audit-'+job['tag']+f'-slot-{job["slot"]:02d}',[R/'env/bin/python','-u',r/'runtime/mobipi/scripts/sim_sprint_safety.py','--receipt',p],simenv(r,0))

def dataset(r):
    import torch
    from mobiwam.pi05_data_learning import HEADS,SCALES,Head,fit_scaler,scale,support,hierarchy_weights
    from mobiwam.scene004 import CANDIDATE_FEATURE_FIELDS
    rows=ledger(r);train=[x for x in rows if x['split']=='train'];selected=[x for x in train if x['hard_valid']]
    participating={x['parent_group'] for x in selected if any(x['head_masks'])}
    data=r/'training/tier-1';data.mkdir(parents=True,exist_ok=True)
    (data/'labels.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows if x['split']!='evaluation'))
    if not participating:write(data/'unavailable.json',dict(at=now(),reason='no_reliable_supervision'));return False
    X=np.stack([np.load(r/'design/inputs'/x['config_id']/(x['route']+'-X.npy')) for x in selected]);y=np.array([[np.nan if v is None else v for v in x['head_values']] for x in selected],np.float32);mask=np.array([x['head_masks'] for x in selected],bool)
    p=np.array([x['parent_group'] for x in selected]);c=np.array([x['config_id'] for x in selected]);routes=np.array([x['route'] for x in selected]);mean,std=fit_scaler(X,p,c,routes,np.array([v in participating for v in p]));heads=support(y,mask)
    np.savez_compressed(data/'train-only.npz',X=X,y=y,mask=mask,parent_group=p,config_id=c,route=routes,hard_valid=np.ones(len(X),bool),role=np.array(['train']*len(X)))
    write(data/'dataset-binding.json',dict(at=now(),training_cutoff=now(),samples=len(selected),planned18=True,independent_parents=len(set(p)),train_only=True,reserved_development_evaluation_ancestors=read(r/'inventory/source-split.json')['reserved_development_evaluation_ancestors'],head_support=heads,late_audits_not_fed_back=True))
    torch.manual_seed(17);m=Head('MLP');opt=torch.optim.AdamW([dict(params=[v for n,v in m.named_parameters() if not n.endswith('bias')],weight_decay=.05),dict(params=[v for n,v in m.named_parameters() if n.endswith('bias')],weight_decay=0.)],lr=3e-4)
    defaults={k:v for k,v in opt.defaults.items() if k!='params'}
    recipe=dict(at=now(),code_commit=read(r/'policy/main-component-freeze.json')['code_commit'],implementation='scripts/pi05_data_obc_train.py; src/mobiwam/pi05_data_learning.py; scripts/pi05_drawer_pipeline.py dataset',CLIP=dict(path=str(R/'cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'),weights='frozen local only',processing='actual renderer256 uint8 /255 -> bicubic224 align_cornersFalse antialiasTrue -> registeredCLIP mean/std; pooler_output; CUDA bf16 autocast -> FP32',cameras=['robot0_agentview_left_image','robot0_agentview_right_image','robot0_eye_in_hand_image'],fusion='L2-normalize each1024 visual token; fourth tanh explicit time-zero proprio token padded/truncated1024; mean4'),geometry=dict(fields=list(CANDIDATE_FEATURE_FIELDS),width=21,planned_time_normalized='existing prefix proxy seconds/120; no terminal state',route_condition='actual pre-outcome frozen geometric prefix'),input=dict(blocks=[1024,21,3],one_hot_order=list('EDA'),scaler='same hierarchy train-only; all1048 dimensions including route onehot; std<1e-6 transforms0',mean=mean.tolist(),std=std.tolist(),sample_keys=[(x['config_id'],x['route']) for x in selected]),heads=heads,output_scales=SCALES.tolist(),output_transform='sigmoid stable logits binary; progress clipped[0,1]; path/time nonnegative; multiply physical scales; constant overrides physical value; unknown nullable',loss='binary BCEWithLogits; continuous masked MSE on normalized units; per-head route withinconfig -> config withinparent -> parent -> learnable heads equal; constants/unsupported excluded',architecture=dict(MLP='1048->32 ReLU->5',Linear='1048->5',initialization='separate seed17 torch.nn.Linear reset_parameters; uniform kaiming weights and uniform bias; FP32 independent optimizer'),optimizer=dict(algorithm='AdamW',defaults=defaults,weights_decay=.05,bias_decay=0.,clip=1.,fullbatch=True,updates=2000,scheduler='CosineAnnealingLR T_max2000 eta_min1e-5; optimizer.step then scheduler.step'),ridge=dict(actual_features='scaled1048 slice1024:1045; route flags occur inside geometry21; extra trailing onehot excluded',alpha=1.,intercept='weighted centered fit; unregularized intercept',regularization='SVD of sqrt(weight)*(X-weighted_mean); alpha1',targets='same five normalized outputs, only learnable heads fitted; binary closed-form regression then clipped; no ridge BCE optimizer',weighting='same head-specific hierarchy',prediction='clip first3[0,1], last2 nonnegative; x2m/x120s; constant/unsupported override'),labels=dict(native_success='observed checker at complete terminal horizon including normal120s budget failure=0; compute timeout unknown',collision='actual native forbidden-contact monitor/audit; protective stop not collision',censoring='budget failure censors time-to-success; actual120s terminal duration/native horizon failure observed',late_labels='archived only after frozen cutoff'),selected_checkpoint='2000 only;500/1000/1500 provenance only',geometry_selector='unchanged scene004.geometry_rule_select',selection='hard-valid -> success .05 -> supported collision .05 -> progress .05 -> path .02m -> time1s -> E<D<A')
    write(r/'training/obc-recipe.json',clean_json(recipe));return True

def predictions(r):
    import torch
    from mobiwam.pi05_data_learning import Head,scale,decode,choose,SCALES
    if datetime.now(timezone.utc)>=OBC:raise RuntimeError('prediction deadline expired; no evaluation without prior frozen predictions')
    rows=ledger(r);train=[x for x in rows if x['split']=='train'];parents=list(dict.fromkeys(x['parent_group'] for x in train));bounds={}
    for route in 'EDA':
        rs=[x for x in train if x['route']==route];vals=[next(x for x in rs if x['parent_group']==p)['safe_success'] for p in parents]
        bounds[route]=dict(lower=sum(x is True for x in vals)/len(parents),upper=sum(x is not False for x in vals)/len(parents))
    decisive=[t for t in 'EDA' if all(bounds[t]['lower']>=bounds[o]['upper'] for o in 'EDA' if o!=t)]
    common=[p for p in parents if decisive and all(next(x for x in train if x['parent_group']==p and x['route']==t)['safe_success'] is True for t in decisive)]
    costs={t:dict(path_m=float(np.mean([next(x for x in train if x['parent_group']==p and x['route']==t)['path_m'] for p in common])) if common else 0.,time_s=float(np.mean([next(x for x in train if x['parent_group']==p and x['route']==t)['terminal_duration_s'] for p in common])) if common else 0.) for t in decisive}
    trainbest=min(decisive,key=lambda t:(-bounds[t]['lower'],costs[t]['path_m'],costs[t]['time_s'],'EDA'.index(t))) if decisive else 'E'
    models={};methods={};data=r/'training/tier-1'
    for kind in ['MLP','Linear']:
        cp=data/kind/'step2000.pt'
        if cp.exists() and read(data/kind/'completed.json').get('steps')==2000:
            state=torch.load(cp,map_location='cpu',weights_only=False);m=Head(kind);m.load_state_dict(state['model']);m.eval();models[kind]=(m,state);methods[kind]=dict(status='available_final2000',checkpoint=str(cp),sha256=sha(cp),heads=state['heads'])
        else: methods[kind]=dict(status='model_unavailable',heads=[])
    ridge=read(data/'ridge/ridge.json') if (data/'ridge/ridge.json').exists() else None;methods['ridge']=dict(status='available_once_alpha1' if ridge else 'model_unavailable',checkpoint=str(data/'ridge/ridge.json') if ridge else None)
    predictions=[];plan=read(r/'design/selected-plan.json')
    for c in plan['configurations']:
        if c['role']!='evaluation':continue
        f=read(r/'design/inputs'/c['config_id']/'features.json') if c['status']=='bound' else dict(routes=[],geometry_selection='X')
        valid=c['hard_valid_routes'];xs={t:np.load(r/'design/inputs'/c['config_id']/(t+'-X.npy')) for t in 'EDA' if valid[t]};values={};choices={}
        for kind,(m,state) in models.items():
            per={}
            for t,X in xs.items():
                with torch.inference_mode():raw=m(torch.from_numpy(scale(X[None],state['scaler']['mean'],state['scaler']['std']))).numpy()
                per[t]=decode(raw,state['heads'])[0].tolist()
            values[kind]=per;choices[kind]=choose(per,valid,state['heads'])
        for kind in ['MLP','Linear']:
            if kind not in choices:choices[kind]='model_unavailable'
        if ridge:
            per={};rc=ridge['recipe'];heads=rc['heads']
            for t,X in xs.items():
                x=scale(X[None],np.array(rc['mean']),np.array(rc['std']))[0];p=np.full(5,np.nan)
                for j,h in enumerate(heads):
                    if h['status']=='constant':p[j]=h['constant']
                    elif h['status']=='learnable':
                        fit=ridge['fits'][str(j)];v=float(x[1024:1045]@np.array(fit['coef'])+fit['intercept']);p[j]=(np.clip(v,0,1) if j<3 else max(v,0))*SCALES[j]
                per[t]=p.tolist()
            values['ridge']=per;choices['ridge']=choose(per,valid,heads)
        else:choices['ridge']='model_unavailable'
        g=f.get('geometry_selection','X');choices['geometry']={x['candidate_id']:x['route_family'] for x in f['routes']}.get(g,'X');choices['train-best-fixed']=trainbest if valid.get(trainbest,False) else 'X'
        predictions.append(dict(config=c,valid_routes=valid,at_least_two_hard_valid=sum(valid.values())>=2,predictions=values,choices=choices))
        for job in plan['jobs']:
            if job['config_id']==c['config_id'] and job['source']=='fresh_online':
                chosen=choices[job['method']];job.update(route=chosen if chosen in 'EDA' and len(chosen)==1 else None,hard_valid=chosen in ('E','D','A'),status='unrun' if chosen in ('E','D','A') else 'unrun_model_unavailable' if chosen=='model_unavailable' else 'X_static_or_route_invalid')
    write(r/'policy/final-evaluation-freeze.json',clean_json(dict(at=now(),status='frozen_before_first_eval',methods=methods,train_best_fixed=trainbest,train_best_bounds=bounds,train_best_selection_supported=bool(decisive),common_safe_train_parents=common,train_best_common_cost=costs,configurations=predictions,scaler_train_only=True,evaluation_outcomes_accessed=False,online_methods=['MLP','geometry','train-best-fixed'])))
    write(r/'design/selected-plan.json',plan);return plan

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run;services=[];handles=[];auditf=[]
    assert not (r/'launch/science-launch-receipt.json').exists(),'no duplicate dispatch'
    write(r/'launch/pipeline-process.json',dict(at=now(),pid=os.getpid(),argv=sys.argv,host=socket.gethostname(),single_agent=True,max_collectors=2))
    try:
        while not (r/'design/start-design.json').exists():
            if datetime.now(timezone.utc)>=DEV-timedelta(seconds=3600):raise RuntimeError('static qualification not complete before dev cutoff')
            time.sleep(10)
        phase(r,'S0_features',status='running')
        rc=run_command(r,'static-features',[R/'env/bin/python','-u',r/'runtime/mobipi/scripts/pi05_drawer_features.py','--run',r],simenv(r,0),r/'runtime/mobipi');assert rc==0,'static features mechanically failed; no science launched'
        configs=freeze_resource(r);write(r/'launch/science-launch-receipt.json',dict(at=now(),pid=os.getpid(),contract_revision=2,GPUs=GPUS,EGL_devices=EGL,ports=PORTS,policy_training=False,resource_freeze=str(r/'design/resource-freeze.json')))
        phase(r,'S1_train_dev',status='running')
        op=r/'runtime/openpi'
        for w in range(2):
            env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONPATH=f'{op}/src:{op}/scripts:{op}/packages/openpi-client/src',CUDA_VISIBLE_DEVICES=GPUS[w],JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(r/'policy/jax-cache'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            s=socket.socket();s.bind(('127.0.0.1',PORTS[w]));s.close()
            cmd=[str(R/'env/openpi/bin/python'),'-u',str(op/'scripts/pi05_fft_serve.py'),'--lora','--checkpoint',str(CP),'--output',str(r/'policy'/f'service{w}'),'--port',str(PORTS[w]),'--max-seconds',str(max(1,int((FINAL-datetime.now(timezone.utc)).total_seconds()))),'--max-queries','15000']
            f=(r/'logs'/f'service{w}.log').open('x');p=subprocess.Popen(cmd,env=env,cwd=op,stdout=f,stderr=subprocess.STDOUT);services.append(p);handles.append(f);write(r/'launch'/f'service{w}-process.json',dict(at=now(),pid=p.pid,command=cmd,CUDA_VISIBLE_DEVICES=GPUS[w],allocation_index=0 if w==0 else 3,logical_cuda_index=0,EGL_index=EGL[w]))
        for w in range(2):
            limit=time.monotonic()+600
            while not (r/'policy'/f'service{w}/ready.json').exists():
                if services[w].poll() is not None:raise RuntimeError(f'own policy service{w} startup failed')
                if time.monotonic()>limit:raise RuntimeError('own policy service readiness600s exceeded')
                time.sleep(5)
        with ThreadPoolExecutor(max_workers=2) as pool,ThreadPoolExecutor(max_workers=2) as audits:
            plan=read(r/'design/selected-plan.json');train=[x for x in plan['jobs'] if x['split']=='train'];dev=[x for x in plan['jobs'] if x['split']=='development' and not x.get('reused')];schedule=[]
            for i in range(6):
                schedule.extend(train[3*i:3*i+3])
                if i<3:schedule.append(dev[i])
            queues=[schedule[::2],schedule[1::2]]
            def collect(queue,w):
                for j in queue:
                    execute_job(r,j,w)
                    if Path(j['receipt']).exists():auditf.append((j,audits.submit(audit_job,r,j)))
            fs=[pool.submit(collect,queues[w],w) for w in range(2)]
            for f in fs:f.result()
            # Prioritize closure of the training audit fields before first update.
            for j,f in list(auditf):
                if j['split']=='train':
                    remaining=max(0.,(OBC-timedelta(seconds=1200)-datetime.now(timezone.utc)).total_seconds())
                    try:f.result(timeout=remaining)
                    except TimeoutError:print('Late train audit archived; reliable native heads proceed',j['receipt'],flush=True)
            phase(r,'S2_train_cutoff',status='running')
            hasdata=dataset(r)
            if hasdata:
                for kind in ['MLP','Linear','ridge']:
                    if datetime.now(timezone.utc)>=OBC-timedelta(seconds=300):break
                    rc=run_command(r,'obc-'+kind,[R/'env/bin/python','-u',r/'runtime/mobipi/scripts/pi05_data_obc_train.py','--run',r,'--tier','1','--model',kind,'--device','cpu'],simenv(r,0),r/'runtime/mobipi')
                    if rc:print('OBC model unavailable',kind,rc,flush=True)
            plan=predictions(r);phase(r,'S3_evaluation',status='running')
            for layer in range(plan['k']):
                states=[c for c in plan['configurations'] if c['role']=='evaluation' and c['start_index']==layer]
                def evalstate(c,w):
                    rr=list('EDA');mm=['MLP','geometry','train-best-fixed'];rr=rr[w:]+rr[:w];mm=mm[w:]+mm[:w]
                    for route,method in zip(rr,mm):
                        for j in [next(x for x in plan['jobs'] if x['config_id']==c['config_id'] and x['source']=='primary' and x['route']==route),next(x for x in plan['jobs'] if x['config_id']==c['config_id'] and x['source']=='fresh_online' and x['method']==method)]:
                            execute_job(r,j,w)
                            if Path(j['receipt']).exists():auditf.append((j,audits.submit(audit_job,r,j)))
                fs=[pool.submit(evalstate,c,w) for w,c in enumerate(states)]
                for f in fs:f.result()
            phase(r,'S4_audits',status='running')
            for j,f in auditf:f.result()
        rows=ledger(r);phase(r,'S4_execution_finished',status='awaiting_delivery',ended_at=now())
        write(r/'delivery/pipeline-completed.json',dict(at=now(),status='execution_and_bounded_audits_closed',new_executed=sum(x['executed'] and not x.get('reused') for x in rows),paper_evidence_ready=False,report_still_required=True))
    except BaseException:
        failure=dict(at=now(),traceback=traceback.format_exc(),scientific_results_preserved=True,automatic_retry=False);write(r/'delivery/pipeline-failure.json',failure);phase(r,'paused_affected_pipeline',status='failure_preserved');traceback.print_exc();raise
    finally:
        for w,p in enumerate(services):
            if p.poll() is None:
                try:
                    c=http.client.HTTPConnection('127.0.0.1',PORTS[w],timeout=10);c.request('POST','/finish-declared-units',body=b'{}');c.getresponse().read();c.close()
                except OSError:pass
        for h in handles:h.close()

if __name__=='__main__':main()
