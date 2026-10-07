"""Bounded frozen-stage pipeline; no retrials, tuning, or score-driven tier."""
import argparse
from datetime import datetime,timezone
import json,os
from pathlib import Path
import subprocess,sys,time

def read(p):return json.loads(Path(p).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run;code=Path(__file__).resolve().parents[1];root=r.parents[3];out=r/'delivery';out.mkdir(exist_ok=True)
 assert not (out/'pipeline-process.json').exists(),'no duplicate pipeline'
 (out/'pipeline-process.json').write_text(json.dumps(dict(at=stamp(),pid=os.getpid(),argv=sys.argv,python=sys.executable,mode='single agent declared execution queues; no retry or post-eval fit'),indent=2)+'\n')
 def status(stage,**kw):
  (out/'pipeline-status.json').write_text(json.dumps(dict(at=stamp(),pid=os.getpid(),stage=stage,**kw),indent=2)+'\n');print(stage,stamp(),kw,flush=True)
 def run(name,*args,env=None):
  command=[sys.executable,'-u',str(code/'scripts'/name),'--run',str(r),*map(str,args)];log=r/'logs'/('pipeline-'+name.removesuffix('.py')+'-'+str(time.time_ns())+'.log');status('subcommand',command=command,log=str(log))
  with log.open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
 def await_stage(stage):
  nextcapacity=0.
  while True:
   closed=all((r/'policy'/f'{stage}-worker{i}/completed.json').exists() for i in [0,1])
   if stage in ['train1','train2'] and time.monotonic()>=nextcapacity:
    run('pi05_data_capacity.py');nextcapacity=time.monotonic()+300
   if closed:
    # Own server shutdown is graceful and must be observed, not guessed.
    for i in [0,1]:
     q=read(r/'policy'/f'{stage}-worker{i}/completed.json');assert q['all_declared_jobs_closed']
    if all((r/'policy'/f'{stage}-service{i}/completed.json').exists() for i in [0,1]):
     failures=[x for i in [0,1] for x in read(r/'policy'/f'{stage}-worker{i}/completed.json')['results'] if x.get('exit_code',0)!=0]
     if failures:
      status('mechanical-failure-paused-for-agent',stage_name=stage,failed_units=failures)
      raise RuntimeError('mechanical failure retained; no automatic resume or fit/eval dispatch')
     break
   status('await-'+stage,workers_closed=closed)
   time.sleep(30)
 def stage(name):
  run('pi05_data_stage.py','--stage',name);run('pi05_data_launch_stage.py','--stage',name);await_stage(name)
 status('await-main-policy-freeze')
 freeze_deadline=datetime.fromisoformat(read(r/'run-manifest.json')['policy_freeze_deadline'])
 while not (r/'policy/main-component-freeze.json').exists():
  if datetime.now(timezone.utc)>=freeze_deadline:raise RuntimeError('policy freeze deadline reached; no primary launched')
  time.sleep(10)
 assert not list((r/'episodes').glob('primary-*/*/engineering-attempt-0/completed.json'))
 stage('train1');run('pi05_data_dataset.py','--tier',1);run('pi05_data_capacity.py','--decide-tier');tier=read(r/'design/tier-decision.json')['selected_tier']
 if tier==2:run('pi05_data_capacity.py');stage('train2');run('pi05_data_dataset.py','--tier',2)
 # Only the selected tier is fitted; skipping the optional first-tier fit
 # conserves the fixed recipe without changing a final model based on scores.
 samples=[]
 for i in range(6):samples.append(dict(at=stamp(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.free','--format=csv,noheader'],text=True)));time.sleep(2)
 (r/'preflight/gpu-selection-obc-final-12s.json').write_text(json.dumps(samples,indent=2)+'\n')
 for row in samples:
  card=[x for x in row['gpu'].splitlines() if x.startswith('1,')][0];assert int(card.split(',')[-1].strip().split()[0])>=2000
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-d1bd8de6-c581-df57-451d-10e3870d66fc',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 for kind in ['MLP','Linear','ridge']:run('pi05_data_obc_train.py','--tier',tier,'--model',kind,'--device','cuda:0',env=env)
 run('pi05_data_predict_freeze.py','--tier',tier);stage('evaluation');run('pi05_data_results.py');status('science-queues-closed-await-evidence',selected_tier=tier)
 (out/'pipeline-completed.json').write_text(json.dumps(dict(at=stamp(),selected_tier=tier,science_queues_closed=True,required_evidence_closeout_pending=True,new_science_after_handoff=False),indent=2)+'\n')
if __name__=='__main__':main()
