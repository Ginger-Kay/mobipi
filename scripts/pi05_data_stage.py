"""Prepare frozen queues and isolated launchers without reading task outcomes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

GPUS=[('GPU-e8903903-cd05-0169-aead-361aba1b99d2',0,8892),('GPU-a22aaeb2-6c91-39e3-a9f8-fcf86bacf63e',3,8893)]
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--stage',choices=['train1','train2','evaluation'],required=True);a=ap.parse_args();r=a.run;root=r.parents[3]
 freeze=read(r/'policy/main-component-freeze.json');cp=Path(freeze['policy_checkpoint']);code=r/'runtime/mobipi';op=r/'runtime/openpi'
 for name,digest in freeze['behavioral_components'].items():assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest
 configs=read(r/'design/primary-roster-plan.json')['configurations']
 slots=[dict(source=c['source'],task=c['task'],parent_group=c['parent_group'],config_id=c['config_id'],family_id=c['family_id'],environment_seed=c['parent_environment_seed']) for c in configs]
 target=r/'design/frozen-harness-roster.json'
 if target.exists():assert read(target)['slots']==slots
 else:write(target,dict(at=datetime.now(timezone.utc).isoformat(),slots=slots,predeclared_roster=str(r/'design/primary-roster-plan.json')))
 jobs=[[],[]];policy=r/'policy';out=policy/('stage-'+a.stage);out.mkdir(exist_ok=False)
 if a.stage=='evaluation':
  final=read(policy/'final-evaluation-freeze.json');selected={x['config']['config_id']:x for x in final['configurations']};tier=final['selected_tier']
  configs=[c for c in configs if c['role']=='evaluation' and c['tier']<=tier]
 else:
  tier=1 if a.stage=='train1' else 2;configs=[c for c in configs if c['role']!='evaluation' and c['tier']==tier]
 if a.stage=='train2':assert read(r/'design/tier-decision.json')['selected_tier']==2
 deadline=read(r/'run-manifest.json')['hard_science_deadline']
 # Conservative start cap leaves45min per last unit plus at least2h closeout.
 from datetime import timedelta
 stop=(datetime.fromisoformat(deadline)-timedelta(seconds=9900)).isoformat()
 if (r/'design/capacity-plan.json').exists() and a.stage!='evaluation':stop=read(r/'design/capacity-plan.json').get('train_dev_stop_starting_at',stop)
 for c in configs:
  worker=c['worker_assignment'];folder=r/'design/repaired-inputs-v1'/c['config_id'];features=read(folder/'features.json') if (folder/'features.json').exists() else {'routes':[]}
  valid={route:bool(next((f['hard_valid'] for f in features['routes'] if f['route_family']==route),False)) for route in 'EDA'}
  if c['static_config_legal']:
   assert Path(features['source']).resolve()==Path(c['source']).resolve()
   assert features['source_integration_sha256']==hashlib.sha256((Path(c['source'])/'integration.npy').read_bytes()).hexdigest()
  route_jobs=[(route,f'primary-tier{c["tier"]}-{c["role"]}-{route}','paired',None) for route in c['route_order']]
  if a.stage=='evaluation':
   methods=final['online_methods'];offset=c['global_config_index']%3;methods=methods[offset:]+methods[:offset]
   route_jobs += [(selected[c['config_id']]['choices'][method],f'online-tier{tier}-{method}','online',method) for method in methods]
  for route,tag,purpose,method in route_jobs:
   job=dict(slot=c['global_config_index']+1,route=route,config_id=c['config_id'],parent_group=c['parent_group'],task=c['task'],evaluation_tag=tag,purpose=purpose,method=method)
   if not c['static_config_legal']:job.update(predeclared_X='X_static_start_rejected',static_rejection_receipt=str(r/'design/start-design.json'))
   elif not valid.get(route,False):job.update(predeclared_X='X_no_legal_candidate',static_rejection_receipt=str(folder/'features.json'))
   jobs[worker].append(job)
 for worker,(uuid,egl,port) in enumerate(GPUS):
  name=a.stage+'-worker'+str(worker);queue=out/(name+'-queue.json');write(queue,dict(at=datetime.now(timezone.utc).isoformat(),stage=a.stage,jobs=jobs[worker],roster=str(target),expected_checkpoint=str(cp),stop_starting_at=stop,finish_own_service=True,component_freeze=str(policy/'main-component-freeze.json'),outcomes_used_for_schedule=False))
  common='#!/usr/bin/env bash\nset -euo pipefail\nsource /share/personal/chensiyu/haokaijiang/dev-jhk.sh\nexport PYTHONNOUSERSITE=1\n'
  shell=common+f'export PYTHONPATH={code}/src:{code}/scripts:{code}:{root}/Mobipi/external/robocasa:{root}/Mobipi/external/robomimic:{root}/Mobipi/external/mimicgen\nexport MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID={egl} CUDA_VISIBLE_DEVICES={uuid} LD_LIBRARY_PATH={root}/env/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XDG_CACHE_HOME={root}/cache/openpi\ncd {code}\n'
  shell+=f'printf "%s\\n" "$$" > {out}/{name}-shell.pid\n{root}/env/bin/python -u scripts/pi05_data_collect.py --run {r} --queue {queue} --port {port} --checkpoint-step {cp.name} --status-name {name} 2>&1 | tee {r}/logs/{name}.log\n'
  (out/('run-'+name+'.sh')).write_text(shell)
  service=a.stage+'-service'+str(worker)
  maxqueries=1200*sum(not j.get('predeclared_X') for j in jobs[worker])+10
  shell=common+f'source {root}/env/openpi/activate-project.sh\nexport PYTHONPATH={op}/src:{op}/packages/openpi-client/src\nexport CUDA_VISIBLE_DEVICES={uuid} XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR={root}/cache/openpi/jax OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1\ncd {op}\n'
  shell+=f'printf "%s\\n" "$$" > {out}/{service}-shell.pid\n{root}/env/openpi/bin/python -u scripts/pi05_harness_serve.py --checkpoint {cp} --output {policy}/{service} --port {port} --max-seconds 172800 --max-queries {maxqueries} 2>&1 | tee {r}/logs/{service}.log\n'
  (out/('run-'+service+'.sh')).write_text(shell)
 write(out/'stage-binding.json',dict(at=datetime.now(timezone.utc).isoformat(),stage=a.stage,tier=tier,configs=len(configs),planned_jobs=sum(map(len,jobs)),static_X=sum(bool(j.get('predeclared_X')) for js in jobs for j in js),workers=[dict(index=i,GPU_uuid=u,egl_index=e,port=p,jobs=len(jobs[i])) for i,(u,e,p) in enumerate(GPUS)],stop_starting_at=stop,outcomes_used=False))
 print((out/'stage-binding.json').read_text(),flush=True)
if __name__=='__main__':main()
