"""One GPU, six fixed slots in sequence; no outcome-driven replacement/retry."""
import argparse,json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone

def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+"\n")
def now():return datetime.now(timezone.utc).isoformat()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();r=a.run
    assert len(json.loads((r/"frozen-six/roster.json").read_text())["slots"])==6
    write(r/"manifest/queue-process.json",dict(started_at=now(),pid=os.getpid(),command=__import__("sys").argv))
    rows=[]
    for slot in range(1,7):
        cmd=[__import__("sys").executable,"-u",str(Path(__file__).with_name("dr_v04_r4_episode.py")),"--run",str(r),"--slot",str(slot)]
        log=r/"episodes"/f"slot-{slot:02d}.log"
        with log.open("x") as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            write(r/"manifest/status.json",dict(updated_at=now(),status="running",slot=slot,child_pid=proc.pid,command=cmd,finished=rows))
            code=proc.wait()
        row=dict(slot=slot,exit_code=code,log=str(log),ended_at=now())
        rows.append(row);write(r/"manifest/queue-results.json",rows)
        print(json.dumps(row),flush=True)
        # Stop unreached model-semantic failures; normal recorded stops exit0 and continue.
        if code:
            write(r/"manifest/status.json",dict(updated_at=now(),status="engineering_hold",failed_slot=slot,finished=rows))
            return code
    write(r/"manifest/status.json",dict(updated_at=now(),status="six_primary_outcomes_completed",finished=rows))
    return 0
if __name__=="__main__":raise SystemExit(main())
