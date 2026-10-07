"""Claim remaining terminal saved-state CPU audits, preserving existing jobs."""
import argparse
from datetime import datetime,timezone
import json,os
from pathlib import Path
import subprocess,sys,time

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--workers',type=int,default=12);a=ap.parse_args();r=a.run;jobs={};deadline=datetime.fromisoformat(json.loads((r/'run-manifest.json').read_text())['hard_science_deadline'])
 while datetime.now(timezone.utc)<deadline:
  for key,(p,f,q) in list(jobs.items()):
   rc=p.poll()
   if rc is not None:
    f.close();q.update(ended_at=datetime.now(timezone.utc).isoformat(),exit_code=rc);Path(key).write_text(json.dumps(q,indent=2)+'\n');del jobs[key]
  receipts=[]
  for file in (r/'episodes').glob('*/slot-*/engineering-attempt-0/completed.json'):
   q=json.loads(file.read_text())
   if q['steps']>0:receipts.append((not q['native_success'],str(file),file,q))
  for _,_,file,q in sorted(receipts):
   if len(jobs)>=a.workers:break
   attempt=Path(q['attempt']);marker=file.parent/'audit-process.json'
   if (attempt/'sprint-safety-audit.json').exists():continue
   try:fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
   except FileExistsError:continue
   command=[sys.executable,'-u',str(Path(__file__).parent/'sim_sprint_safety.py'),'--receipt',str(file)]
   f=(file.parent/'safety-audit.log').open('x');p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
   record=dict(at=datetime.now(timezone.utc).isoformat(),pid=p.pid,command=command,mode='CPU saved-native original0.5mm audit; no physics replay',owner='auxiliary-positive-first-CPU-auditor')
   with os.fdopen(fd,'w') as m:m.write(json.dumps(record,indent=2)+'\n')
   jobs[str(marker)]=(p,f,record);print('AUDIT',p.pid,file,flush=True)
  (r/'paper-evidence/aux-audit-watch-state.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),active_audits={k:p.pid for k,(p,_,_) in jobs.items()},workers=a.workers),indent=2)+'\n')
  if (r/'delivery/stop-audit-watch.json').exists() and not jobs:break
  time.sleep(10)
 for key,(p,f,q) in jobs.items():
  rc=p.wait();f.close();q.update(ended_at=datetime.now(timezone.utc).isoformat(),exit_code=rc);Path(key).write_text(json.dumps(q,indent=2)+'\n')
if __name__=='__main__':main()
