"""Bounded CPU saved-state audits. No replay, policy, or outcome-based retry."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();jobs={}
    deadline=datetime.fromisoformat(json.loads((a.run/'run-manifest.json').read_text())['hard_science_deadline'])
    while datetime.now(timezone.utc)<deadline:
        for key,(p,f,record) in list(jobs.items()):
            rc=p.poll()
            if rc is not None:
                f.close();record.update(ended_at=datetime.now(timezone.utc).isoformat(),exit_code=rc)
                (Path(key).parent/'audit-process.json').write_text(json.dumps(record,indent=2)+'\n');del jobs[key]
        for receipt in sorted((a.run/'episodes').glob('*/slot-*/engineering-attempt-*/completed.json')):
            q=json.loads(receipt.read_text());attempt=Path(q['attempt']);key=str(receipt)
            if key in jobs or (attempt/'sprint-safety-audit.json').exists() or (receipt.parent/'audit-process.json').exists():continue
            if len(jobs)>=a.workers:break
            f=(receipt.parent/'safety-audit.log').open('x');command=[sys.executable,'-u',str(Path(__file__).parent/'sim_sprint_safety.py'),'--receipt',str(receipt)]
            p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT);record=dict(at=datetime.now(timezone.utc).isoformat(),pid=p.pid,command=command,mode='CPU saved-native conservative0.5mm sweep; no physics replay')
            jobs[key]=(p,f,record);(receipt.parent/'audit-process.json').write_text(json.dumps(record,indent=2)+'\n');print('AUDIT',p.pid,receipt,flush=True)
        state=dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),active_audits={k:p.pid for k,(p,_,_) in jobs.items()},native_interval_threshold_m=.0005)
        (a.run/'paper-evidence/audit-watch-state.json').write_text(json.dumps(state,indent=2)+'\n')
        if (a.run/'delivery/stop-audit-watch.json').exists() and not jobs:break
        time.sleep(10)
    for key,(p,f,record) in jobs.items():
        # Already started bounded CPU audit completes; no new science is opened.
        rc=p.wait();f.close();record.update(ended_at=datetime.now(timezone.utc).isoformat(),exit_code=rc)
        (Path(key).parent/'audit-process.json').write_text(json.dumps(record,indent=2)+'\n')
    print('AUDIT WATCH CLOSED',flush=True)

if __name__=='__main__':main()
