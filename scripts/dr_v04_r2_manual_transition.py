"""Researcher-run transition: stop this queue's dispatch, drain children, then launch.

Only SIGINT is sent, to the exact registered R2 CPU coordinator. No signal is
sent to a collector, auditor, unknown process, GPU worker or occupancy program.
"""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from dr_v04_r2_batch import load,write,stamp,sha
from mobiwam.dr_v04_gpu_inventory import inventory,require_occupancy_only


def command(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
    except FileNotFoundError:return ''


def live_collectors(run):
    found=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:text=path.read_bytes().replace(b'\0',b' ').decode()
        except (OSError,UnicodeError):continue
        if str(run) in text and '/scripts/dr_v04_formal_collect.py ' in text:found.append(int(path.parent.name))
    return found


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--binding',type=Path,required=True);a=p.parse_args();run=a.run.resolve();b=load(a.binding)
    lock=(run/'four-gpu-transition.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=run/'preflight/manual-four-gpu-start.json'
    if marker.exists():raise ValueError('manual transition already started; inspect existing provenance before any retry')
    status=load(run/'status.json');root=Path(__file__).resolve().parent.parent
    oldpid=status['pid'];expected=str(run/'runtime-stop-repair/scripts/dr_v04_r2_batch.py')
    if status['state']!='running' or expected not in command(oldpid) or '--run '+str(run) not in command(oldpid):raise ValueError('old coordinator identity/state differs; no process intervention')
    if status.get('mode')=='four_gpu':raise ValueError('four GPU queue already running')
    write(marker,dict(at=stamp(),run_id=b['run_id'],binding_sha256=sha(a.binding),requested_gpus=[0,1,2,3],manual_entrypoint=str(Path(__file__).resolve()),old_coordinator_pid=oldpid,old_coordinator_command=command(oldpid),old_status=status,scope='user-authorized four-card data collection; researcher invoked manual launch; no training'))
    print(stamp(),'DRAIN exact CPU coordinator',oldpid,'; collectors/auditors continue',flush=True)
    os.kill(oldpid,signal.SIGINT)
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        # The wrapper writes exit only after original CPU audits drained.
        if (run/'terminal-resume.exit').exists() and (run/'terminal-resume-observer.exit').exists() and not command(oldpid) and not live_collectors(run):break
        time.sleep(5)
    else:raise TimeoutError('bounded drain did not close; no further signal or parallel launch')
    write(run/'preflight/four-gpu-drain-completed.json',dict(at=stamp(),prior_status=load(run/'status.json'),old_exit=(run/'terminal-resume.exit').read_text().strip(),old_observer_exit=(run/'terminal-resume-observer.exit').read_text().strip(),scientific_outcomes_repeated=0))
    observations=[]
    for i in range(6):
        inv=inventory();require_occupancy_only(inv,b['expected_occupancy']);observations.append(dict(at=stamp(),inventory=inv));time.sleep(2)
    write(run/'preflight/gpu-before-four-gpu.json',observations)
    # Do not touch unknown dirty Research work; observer will only publish if clean.
    m=load(run/'manifest.json');m.setdefault('binding_history',[]).append(dict(binding=m['binding'],code_commit=m['code_commit'],scope='single-card collection completed before manual parallel transition'))
    m.update(binding=str(a.binding.resolve()),execution_binding=str(a.binding.resolve()),binding_sha256=sha(a.binding),execution_binding_sha256=sha(a.binding),code_commit=b['execution_code_commit'],execution_code_commit=b['execution_code_commit'],runtime_code_root=str(root),worker_count=4,gpu_allocation_indices=[0,1,2,3],analysis_code_root=str(root),analysis_code_commit=b['execution_code_commit'],current_control_write_owner='Compute Execution',status='four_gpu_launching',source_code_delivery='origin/codex/obc-wam-dr-v04-r2-four-gpu')
    write(run/'manifest.json',m)
    session='jhk-obc-dr-v04-r2-batch'
    subprocess.run(['tmux','new-window','-t',session,'-n','four-gpu-queue','bash '+str(run/'run-four-gpu.sh')],check=True)
    for i in range(30):
        s=load(run/'status.json')
        if s.get('mode')=='four_gpu':break
        if (run/'four-gpu.exit').exists():raise RuntimeError('parallel coordinator failed at startup')
        time.sleep(2)
    else:raise TimeoutError('four GPU coordinator did not report startup')
    subprocess.run(['tmux','new-window','-t',session,'-n','four-gpu-checkpoints','bash '+str(run/'observe-four-gpu.sh')],check=True)
    write(run/'preflight/four-gpu-launch.json',dict(at=stamp(),status=s,transition_pid=os.getpid(),session=session,private_socket=os.environ.get('JHK_TMUX_SOCKET'),physics_and_A_speed_changed=False))
    print(stamp(),'four-GPU coordinator launched',flush=True)

if __name__=='__main__':main()
