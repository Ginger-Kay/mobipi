"""Single-use train/dev paired slots, after task and route release."""
import argparse,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--task',required=True)
    p.add_argument('--route',choices=['E','D','A'],required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args();r=a.run
    readiness=json.loads((r/'policy/task-readiness.json').read_text());assert readiness['tasks'][a.task]['passed']
    if a.route!='E':
        release=json.loads((r/'policy/harness-route-release.json').read_text());assert release['tasks'][a.task][a.route]['released']
    source=json.loads((r/'data/paired-source-roster.json').read_text());slots=[i+1 for i,x in enumerate(source['slots']) if x['task']==a.task and x['role'] in ('train','dev')]
    assert len(slots)==6
    evaluation='paired-v6-train-dev-'+a.route;manifest=r/'paired'/('freeze-'+a.task+'-'+a.route+'.json');manifest.parent.mkdir(exist_ok=True)
    assert not manifest.exists()
    manifest.write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),task=a.task,route=a.route,slots=slots,source_roster='data/paired-source-roster.json',
        policy_checkpoint=readiness['policy_checkpoint'],control_compatibility='policy/collection-compatibility.json',purpose='new frozen pi05 paired outcomes, no old labels',
        final_sources_excluded=True,route_order='E first, release D/A separately on same sealed Source; no common physical changes'),
        indent=2)+'\n')
    code=r/'runtime/mobipi-v6-collection/scripts/pi05_harness_episode.py'
    for slot in slots:
        log=r/f'logs/paired-{a.route}-{a.task}-slot-{slot}.log'
        with log.open('x') as f:
            subprocess.run([sys.executable,'-u',str(code),'--run',str(r),'--slot',str(slot),'--route',a.route,'--checkpoint-step','2000','--port',str(a.port),
                '--adapter-version','v6','--evaluation-tag',evaluation,'--roster',str(r/'data/paired-source-roster.json'),'--purpose','paired'],stdout=f,stderr=subprocess.STDOUT,check=False)
    receipts=list((r/'episodes'/evaluation).glob('slot-*/engineering-attempt-0/completed.json'))
    from concurrent.futures import ThreadPoolExecutor
    def audit(receipt):
        q=json.loads(receipt.read_text());done=Path(q['attempt'])/'sprint-safety-audit.json'
        if done.exists():return 0
        with (receipt.parent/'safety-audit.log').open('x') as f:
            return subprocess.run([sys.executable,str(r/'runtime/mobipi/scripts/sim_sprint_safety.py'),'--receipt',str(receipt)],stdout=f,stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=3) as pool:print('audits',list(pool.map(audit,receipts)),flush=True)
    from mobiwam.pi05_adapter import call
    try:call(a.port,'/finish-declared-units',b'{}')
    except ConnectionRefusedError:print('service already reached declared bound',flush=True)
if __name__=='__main__':main()
