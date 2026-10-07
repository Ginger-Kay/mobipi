"""Launch only declared isolated workers after exact own service closure."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import socket,subprocess,time

def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--stage',choices=['train1','train2','evaluation'],required=True);a=ap.parse_args();r=a.run;plan=r/'policy'/('stage-'+a.stage)
 assert (plan/'stage-binding.json').exists() and not (plan/'launch.json').exists()
 freeze=read(r/'policy/main-component-freeze.json')
 for name,digest in freeze['behavioral_components'].items():assert hashlib.sha256((r/'runtime/mobipi'/name).read_bytes()).hexdigest()==digest
 if a.stage=='train1':previous='dev-corrective2000'
 elif a.stage=='train2':previous='train1'
 else:previous='train2' if read(r/'design/tier-decision.json')['selected_tier']==2 else 'train1'
 for i in [0,1]:
  q=read(r/'policy'/f'{previous}-service{i}/completed.json');assert q['declared_units_finished']
  with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8892+i))!=0,'previous exact service still closing or unknown listener; do not duplicate'
 samples=[]
 for i in range(6):
  sample=dict(at=datetime.now(timezone.utc).isoformat(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.free','--format=csv,noheader'],text=True),contexts=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True));samples.append(sample);time.sleep(2)
 for row in samples:
  gpu={int(line.split(',')[0]):int(line.split(',')[-1].strip().split()[0]) for line in row['gpu'].splitlines()}
  assert len(gpu)==4 and gpu[0]>=18000 and gpu[3]>=18000,'not enough free memory for own isolated policy+EGL contexts'
 samplepath=r/'preflight'/('gpu-selection-'+a.stage+'-12s.json');samplepath.write_text(json.dumps(samples,indent=2)+'\n')
 live=__import__('shutil').disk_usage(r);assert live.free>250*1024**3,'insufficient conservative first batch storage headroom'
 launches=[]
 for worker in [0,1]:
  for kind in ['service','worker']:
   name=a.stage+'-'+kind+str(worker);path=plan/('run-'+name+'.sh');command='bash '+str(path)
   subprocess.run(['tmux','new-window','-t','jhk-obc-pi05-data-v1','-n',name,command],check=True);launches.append(dict(name=name,command=command,worker=worker,GPU_uuid=read(plan/'stage-binding.json')['workers'][worker]['GPU_uuid']))
 receipt=dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stage=a.stage,session='jhk-obc-pi05-data-v1',socket=os.environ.get('JHK_TMUX_SOCKET'),shell_initialization='source /share/personal/chensiyu/haokaijiang/dev-jhk.sh',previous_services=previous,resource_audit=str(samplepath),free_storage_bytes=live.free,launches=launches,isolated_models_envs_RNG_outputs=True,unknown_jobs_untouched=True)
 (plan/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
