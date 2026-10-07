"""Execute one predeclared, isolated worker queue; no selection by outcomes."""
import argparse
from datetime import datetime,timezone
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--queue',type=Path,required=True);ap.add_argument('--port',type=int,required=True);ap.add_argument('--checkpoint-step',type=int,required=True);ap.add_argument('--status-name',required=True);a=ap.parse_args()
    plan=json.loads(a.queue.read_text());out=a.run/'policy'/a.status_name;out.mkdir(exist_ok=False)
    ready_deadline=time.monotonic()+600
    while True:
        try:
            c=http.client.HTTPConnection('127.0.0.1',a.port,timeout=5);c.request('GET','/status');response=c.getresponse();binding=json.loads(response.read());c.close()
            if response.status==200:break
        except (OSError,ValueError):pass
        if time.monotonic()>ready_deadline:raise RuntimeError('own declared service failed to become ready in600s')
        time.sleep(5)
    write(out/'process.json',dict(at=now(),pid=os.getpid(),argv=sys.argv,queue=str(a.queue),policy_binding=binding))
    results=[]
    for job in plan['jobs']:
        if datetime.now(timezone.utc)>=datetime.fromisoformat(plan['stop_starting_at']):
            results.append(dict(job=job,status='unrun_deadline'));continue
        old=a.run/'episodes'/job['evaluation_tag']/f'slot-{job["slot"]:02d}-{job["config_id"]}'/'engineering-attempt-0'
        if old.exists():raise ValueError('predeclared slot already exists; no unmodified repeat')
        cmd=[sys.executable,'-u',str(Path(__file__).parent/'pi05_harness_episode.py'),'--run',str(a.run),'--slot',str(job['slot']),'--route',job['route'],
            '--checkpoint-step',str(a.checkpoint_step),'--port',str(a.port),'--adapter-version','v6','--evaluation-tag',job['evaluation_tag'],
            '--roster',plan['roster'],'--purpose',job['purpose'],'--A-private-version','A3N' if job['route']=='A' else 'A1','--sim-seconds','300','--wall-seconds','2700','--diagnostic-logging']
        started=now();log=a.run/'logs'/f'{job["evaluation_tag"]}-slot-{job["slot"]:03d}-{job["route"]}.log'
        with log.open('x') as f:
            p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            write(out/'status.json',dict(at=now(),started_at=started,pid=p.pid,job=job,command=cmd,log=str(log),status='running'))
            code=p.wait()
        result=dict(job=job,exit_code=code,started_at=started,ended_at=now(),command=cmd,log=str(log),completed_receipt=str(old/'completed.json'))
        results.append(result);write(out/'results.json',results);print(json.dumps(result),flush=True)
        # Pure mechanical failures remain in place. This scheduler never retries
        # or changes code, policy, source, roster or result labels on its own.
    write(out/'completed.json',dict(at=now(),planned=len(plan['jobs']),results=results,all_declared_jobs_closed=True))
    if plan.get('finish_own_service',True):
        c=http.client.HTTPConnection('127.0.0.1',a.port,timeout=5);c.request('POST','/finish-declared-units',body=b'{}');c.getresponse().read();c.close()

if __name__=='__main__':main()
