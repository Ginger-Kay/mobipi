"""Independent16:40 export and bounded final delivery; never interrupts healthy eval."""
import argparse,json,os,signal,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from pi05_drawer_pipeline import R,simenv,read,write,now
from pi05_drawer_data_v2 import launch,owned
from pi05_drawer_obc_report_v2 import STEM

def generate(r,final=False):
    manifest=read(r/'run-manifest.json');manifest['code_commit']=subprocess.check_output(['git','-C',str(r/'runtime/mobipi'),'rev-parse','HEAD'],text=True).strip();write(r/'run-manifest.json',manifest)
    cmd=[R/'env/bin/python','-B','-u',r/'runtime/mobipi/scripts/pi05_drawer_obc_report_v2.py','--run',r]
    if final:cmd.append('--final')
    subprocess.run(list(map(str,cmd)),env=simenv(r,0),check=True,timeout=180)



def event(r,name,**kw):
    with (r/'delivery/closeout-events.jsonl').open('a') as f:f.write(json.dumps(dict(at=now(),event=name,**kw))+'\n')


def git(r):
    c=r/'runtime/control';m=r/'runtime/mobipi'
    def run(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.STDOUT,timeout=60).strip()
    rec=dict(at=now(),status='prepared_local_final_report')
    try:
        assert run(c,'remote','get-url','origin')=='git@github-jhk:Ginger-Kay/MM-WAM-Research.git'
        assert run(m,'remote','get-url','origin')=='git@github-jhk:Ginger-Kay/mobipi.git'
        run(c,'fetch','--prune','origin');run(m,'fetch','--prune','origin')
        paths=['docs/obc-wam-active-recovery.md','08-experiments/README.md','08-experiments/code-registry.md',f'08-experiments/reports/{STEM}.md',f'08-experiments/reports/{STEM}-tables',f'08-experiments/handoff/{STEM}.md']
        run(c,'add',*paths)
        staged=run(c,'diff','--cached','--name-only').splitlines()
        assert all(any(p==a or p.startswith(a+'/') for a in paths) for p in staged)
        if staged:run(c,'commit','-m','Deliver bounded drawer OBC v2 online results and common-support tables')
        rec['control_commit']=run(c,'rev-parse','HEAD');rec['mobipi_commit']=run(m,'rev-parse','HEAD')
        run(c,'push','origin','HEAD:main');run(m,'push','origin','HEAD');run(c,'fetch','origin');run(m,'fetch','origin')
        rec.update(status='pushed_control_and_mobipi',control_remote='origin/main',control_parity=run(c,'rev-list','--left-right','--count','HEAD...origin/main'),mobipi_remote='origin/codex/pi05-drawer-obc-v2-20261009',mobipi_parity=run(m,'rev-list','--left-right','--count','HEAD...@{upstream}'),control_clean=not bool(run(c,'status','--porcelain')),mobipi_clean=not bool(run(m,'status','--porcelain')))
    except Exception:rec.update(status='local_result_preserved_git_delivery_failed',error=traceback.format_exc())
    rec['ended_at']=now();write(r/'delivery/git-delivery.json',rec);event(r,'git_delivery',status=rec['status']);return rec


def main(r):
    write(r/'launch/closeout-process.json',dict(at=now(),pid=os.getpid(),argv=__import__('sys').argv,run_id=r.name))
    exported=False
    while True:
        d=read(r/'deadline-config.json');phase=read(r/'phase-state.json')
        if not exported and datetime.now(timezone.utc)>=datetime.fromisoformat(d['incremental_export']):
            try:generate(r,False);event(r,'16_40_incremental_export');exported=True
            except Exception:event(r,'incremental_export_error',error=traceback.format_exc());exported=True
        if phase.get('phase')=='execution_closed':break
        time.sleep(5)
    # Evaluation services/episodes are already closed; use one GPU for remaining
    # dev input caches only. Per-config files survive a deadline interruption.
    records=[read(p) for p in (r/'launch').glob('episode-*-process.json')]+[read(p) for p in (r/'launch').glob('service*-process.json')]
    assert not any(owned(q) for q in records),'healthy science process still alive; preserve it'
    seconds=(datetime.fromisoformat(d['execution_deadline'])-datetime.now(timezone.utc)).total_seconds()-40
    if seconds>60:
        cmd=[R/'env/bin/python','-B','-u',r/'runtime/mobipi/scripts/pi05_drawer_features.py','--run',r,'--roster',r/'design/dev-input-roster.json','--freeze-output','design/dev-features-freeze.json']
        p,rec,f=launch(r,'features-dev-final',cmd,simenv(r,0));event(r,'dev_features_after_online_close',seconds_available=seconds,pid=rec['pid'])
        if p is not None:
            try:rc=p.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                if owned(rec):p.send_signal(signal.SIGINT)
                try:rc=p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if owned(rec):p.terminate()
                    rc=p.wait(timeout=10)
                event(r,'dev_feature_deadline_partial_cached_inputs_retained')
            if f:f.close()
            event(r,'dev_features_closed',exit_code=rc)
    event(r,'all_scientific_inference_closed')
    generate(r,True);result=git(r);write(r/'delivery/closeout-completed.json',dict(at=now(),status='closed',git_status=result['status'],no_new_science=True,all_healthy_episodes_preserved=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
