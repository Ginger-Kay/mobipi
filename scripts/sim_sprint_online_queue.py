"""One frozen learned/geometry attempt per validation group; retain all failures."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    out=r/'online-queue';out.mkdir(exist_ok=False)
    freeze=json.loads((r/'freeze.json').read_text());groups=sorted(set(x['group_id'] for x in freeze['main_slots'] if x['split']=='development-validation'))
    roster=[dict(group_id=g,methods=['learned','geometry'] if i%2==0 else ['geometry','learned']) for i,g in enumerate(groups)]
    write(out/'freeze.json',dict(created_at=now(),groups=roster,episodes_cap=16,Source_config_monitor_from=str(r/'freeze.json'),
        checkpoint='fixed training/MLP/step2000.pt after comparison',policy='reference fallback only; frozen BC readiness failed',
        group_replacement=False,scientific_outcome_retries=0,method_order='sorted group index odd/even alternating',
        transfer_difference='MW repaired closed-progress preflight; separate from inherited training table',sealed_test_used=False))
    write(out/'process.json',dict(started_at=now(),pid=os.getpid(),argv=sys.argv,code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()))
    results=[];wait_started=time.monotonic()
    while not (r/'comparison/final-prediction-receipt.json').exists():
        write(out/'status.json',dict(updated_at=now(),status='waiting_fixed_final_comparison',wait_seconds=time.monotonic()-wait_started))
        if time.monotonic()-wait_started>7200:raise TimeoutError('fixed final comparison still absent after bounded2h wait')
        time.sleep(10)
    bindings=json.loads((r/'reference-plans/current-bindings.json').read_text())
    for entry in roster:
        group=entry['group_id'];plan=Path(bindings[group]);t0=time.monotonic()
        while not (plan/'completed.json').exists() and not (plan/'failure.json').exists():
            write(out/'status.json',dict(updated_at=now(),status='waiting_preflight',group_id=group,wait_seconds=time.monotonic()-t0,completed=results))
            if time.monotonic()-t0>3600:
                write(out/(group+'-preflight-unavailable.json'),dict(at=now(),reason='bounded_wait_expired_no_outcome',plan=str(plan)));break
            time.sleep(10)
        for method in entry['methods']:
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('sim_sprint_online.py')),'--run',str(r),'--group',group,'--method',method]
            with (out/f'{group}-{method}.log').open('x') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                write(out/'status.json',dict(updated_at=now(),status='running',group_id=group,method=method,child_pid=proc.pid,command=cmd,completed=results))
                exitcode=proc.wait()
            row=dict(group_id=group,method=method,exit_code=exitcode,finished_at=now(),output=str(r/'episodes'/group/method))
            results.append(row);write(out/'results.json',results);print(json.dumps(row),flush=True)
            # Every valid result is final. Mechanical failures are retained, not
            # silently rerun or replaced by another Source in this queue.
    write(out/'status.json',dict(updated_at=now(),status='completed_all_frozen_slots',completed=results))

if __name__=='__main__':main()
