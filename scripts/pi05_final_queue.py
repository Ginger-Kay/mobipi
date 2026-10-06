"""Actual prospective paired/online calls; no action cache or slot retry."""
import argparse,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--mode',choices=['paired','online'],required=True);a=p.parse_args();r=a.run
    freeze=json.loads((r/'evaluation/final-freeze.json').read_text());assert len(freeze['predictions'])*3<=24
    release=freeze['released_routes'];code=r/'runtime/mobipi-v6-collection/scripts/pi05_harness_episode.py';A2code=r/'runtime/mobipi-v6-A2/scripts/pi05_harness_episode.py';made=[]
    for group_index,g in enumerate(freeze['predictions']):
        if a.mode=='paired':
            order=list('EDA');order=order[group_index%3:]+order[:group_index%3];units=[(route,route) for route in order if release['tasks'][g['task']].get(route,{}).get('released')]
        else:
            methods=freeze['online_methods'];order=methods[group_index%3:]+methods[:group_index%3];units=[(method,g['selected'][method]) for method in order]
        for label,route in units:
            if route=='X':continue
            evaluation='paired-v6-final-'+label if a.mode=='paired' else 'online-v6-'+label
            log=r/f'logs/{evaluation}-slot-{g["slot"]}.log'
            args=[sys.executable,'-u',str(A2code if route=='A' else code),'--run',str(r),'--slot',str(g['slot']),'--route',route,'--checkpoint-step','2000','--port',str(a.port),
                '--adapter-version','v6','--evaluation-tag',evaluation,'--roster',str(r/'data/paired-source-roster.json'),'--purpose',a.mode]
            if route=='A':args+=['--A-private-version','A2']
            with log.open('x') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=False)
            path=next((r/'episodes'/evaluation).glob(f'slot-{g["slot"]:02d}*/engineering-attempt-0/completed.json'),None)
            if path:made.append(path)
    def audit(p):
        q=json.loads(p.read_text());done=Path(q['attempt'])/'sprint-safety-audit.json'
        if done.exists():return 0
        with (p.parent/'safety-audit.log').open('x') as f:return subprocess.run([sys.executable,str(r/'runtime/mobipi/scripts/sim_sprint_safety.py'),'--receipt',str(p)],stdout=f,stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=3) as pool:print('audits',list(pool.map(audit,made)),flush=True)
    from mobiwam.pi05_adapter import call
    try:call(a.port,'/finish-declared-units',b'{}')
    except ConnectionRefusedError:print('service reached declared bound',flush=True)
if __name__=='__main__':main()
