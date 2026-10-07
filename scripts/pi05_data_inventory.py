"""Read approved inventories and finalized captures; freeze source uses before fit.

No simulator, policy, sealed-test metadata, or replay is accessed here.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path('/share/personal/chensiyu/haokaijiang/MobiWAM')
HUMAN = ROOT/'artifacts/MMWAM-OBC-002-DR/development-human-scene-expansion/20261003T140621Z-two-day-preparation'
HARNESS = ROOT/'artifacts/MMWAM-OBC-002-PI05-HARNESS/v1/20261006T150500Z-pi05-harness'
SPRINT = ROOT/'artifacts/MMWAM-OBC-002-SIM-SPRINT/v1/20261006T072532Z-aamas-simulation-sprint'
SIM = ROOT/'artifacts/MMWAM-OBC-002-PI05-SIM/v2/20261007T030221Z-natural-a-mw300'

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); a=ap.parse_args()
    out=a.run/'inventory'; out.mkdir(exist_ok=True)
    if (out/'source-split.json').exists(): raise ValueError('source split is immutable; inspect existing result')
    rows=read(SPRINT/'data/inventory-v5.json')
    byid={x['record_id']:dict(x,inventory_receipt=str(SPRINT/'data/inventory-v5.json')) for x in rows}
    old_split=read(HARNESS/'data/lineage-split.json')['parent_groups']
    old_fit=read(HARNESS/'data/relative-dataset-binding.json')
    fit_ancestors={x['parent_group'] for x in old_fit['episodes']}
    groups=[]; rng=np.random.default_rng(17)
    for task in ('CloseDrawer','CloseSingleDoor'):
        for historical,role,n in [('train','train',6),('dev','development',2),('final','evaluation',2)]:
            pool=sorted([x for x in old_split if x['task']==task and x['role']==historical],key=lambda x:x['parent_group'])
            if role!='train': pool=[x for x in pool if x['parent_group'] not in fit_ancestors]
            for ix in rng.permutation(len(pool))[:n]:
                x=dict(pool[int(ix)]); x.update(historical_role=historical,role=role,
                    environment_seed=int(re.search(r'seed(\d+)$',x['config_id']).group(1)),
                    known_development=True,blind_evaluation=False,template_status='seen-template',
                    policy_fit_ancestor=x['parent_group'] in fit_ancestors)
                groups.append(x)
    assert len(groups)==20
    # Reserve all old development/final parents, including those outside core20.
    reserved={x['parent_group'] for x in old_split if x['role'] in ('dev','final')}
    train_pool={x['parent_group'] for x in old_split if x['role'] in ('train','unused')}
    assert not fit_ancestors & reserved
    split={'at':now(),'seed':17,'selection':'preserve historical train/dev/final roles; stable parent-ID sort then per-task numpy seed17 permutation; 6/2/2 each task; no new outcome used',
        'parent_groups':groups,'reserved_development_evaluation_ancestors':sorted(reserved),
        'old_fit2_policy_fit_ancestors':sorted(fit_ancestors),'additional_policy_fit_only_parents':sorted(train_pool-{x['parent_group'] for x in groups if x['role']=='train'}),
        'counts':dict(Counter(x['role'] for x in groups)),'old_sealed_test':'not opened; only delivered nonsealed inventory and existing exclusion boundary',
        'family_count':2,'shared_template_claim':'known development, seen-template; no blind or template-held-out claim'}
    write(out/'source-split.json',split)
    indexed=list(csv.DictReader((HUMAN/'attempt-index.csv').open()))
    known={x['attempt_id'] for x in indexed}
    filesystem_attempts=sorted(HUMAN.glob('scenes/*/*/source-*/*/attempt-*'))
    extra=[]
    for d in filesystem_attempts:
        if d.name in known: continue
        result=d/'result.json'
        extra.append({'attempt':str(d),'attempt_id':d.name,'status':'new' if result.exists() and read(result).get('ended_at') else 'incomplete','files':[{ 'path':str(p),'size':p.stat().st_size} for p in d.iterdir() if p.is_file()]})
    qualification={}
    # Small qualification receipts only; no replay or raw video decoding.
    for p in HUMAN.glob('qualification/**/qualification.json'):
        x=read(p)
        if x.get('attempt'): qualification[Path(x['attempt']).name]=(str(p),x)
    for x in indexed:
        ident=x['attempt_id']; p=Path(x['result_path']); result=read(p) if p.exists() else {}
        attempt=p.parent; meta=read(attempt/'metadata.json') if (attempt/'metadata.json').exists() else {}
        source=attempt.parents[1]; old=byid.get(ident,{})
        source_meta=read(source/'source.json') if (source/'source.json').exists() else {}
        task=old.get('task') or source_meta.get('task') or ('CloseDrawer' if 'DR-' in str(source) else 'CloseSingleDoor')
        qual=qualification.get(ident)
        q=qual[1] if qual else {}
        machine=bool(q.get('machine_qualification_pass',q.get('machine_qualified_success',False)))
        finalized=bool(result.get('ended_at'))
        row=dict(old,record_id=ident,view='B_human',human_or_auto='human',task=task,route=x['route'],record_type=x['record_type'],
            parent_source=str(source),parent_group=str(source),source_id=x['source_id'],config_id=old.get('config_id',source.parent.parent.name),
            family_id=task+'-layout1-style0',attempt=str(attempt),source=str(source),observations_path=x['trajectory_path'],video_path=x['original_video_path'],panoramic_path=x['panoramic_video_path'],
            success=result.get('checker_success'),stop_reason=result.get('reason'),finalized=finalized,machine_qualified_success=machine,
            qualification_receipt=qual[0] if qual else old.get('qualification_receipt'),human_continuity=old.get('human_grip'),
            historical_use='human development; not automatic outcome label',new_use='policy-demo-candidate' if finalized and machine and result.get('checker_success') else 'diagnostic-only',
            rejection_reason=None if finalized and machine and result.get('checker_success') else 'not a complete quality-qualified successful expert',inventory_receipt=str(HUMAN/'attempt-index.csv'),
            original_result_receipt=str(p),native_task_success=result.get('checker_success'),obc_label_admission=False)
        if ident=='attempt-20261006T165202212474Z':
            row.update(new_use='diagnostic-only',rejection_reason='operator reports slip and regrasp; whole episode rejected as expert',human_continuity='rejected slip/regrasp')
        byid[ident]=row
    for row in byid.values():
        if row['view'].startswith('A_'):
            parent=row.get('parent_source'); row['parent_group']=parent
            row['new_use']='historical executor outcomes; policy-demo-candidate' if parent in train_pool and row.get('success') and str(row.get('clearance_qualification')).startswith('pass') else 'historical/diagnostic-only'
            row['demonstrator_type']='reference_controller'
            row['rejection_reason']=None if 'candidate' in row['new_use'] else ('reserved development/evaluation ancestry' if parent in reserved else 'expert quality/success not admitted')
        row['obc_label_admission']=False
    for base,version in [(HARNESS,'PI05-HARNESS-v1'),(SIM,'PI05-SIM-v2')]:
        for p in sorted((base/'episodes').glob('*/*/engineering-attempt-*/completed.json')):
            x=read(p); ident=version+'/'+str(p.relative_to(base)); byid[ident]=dict(x,record_id=ident,view='PI05_historical',historical_use=version,
                raw_receipt=str(p),new_use='component-compatibility-pending; historical evidence',obc_label_admission=False,
                reason='old120s truncation or changed solver/policy/initial-state cannot substitute current300s prospective outcome')
    allrows=list(byid.values())
    (out/'inventory.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in allrows))
    supplement=[{'attempt_id':x['attempt_id'],'source_id':x['source_id'],'route':x['route'],'status':'supplement','reason':'already included in delivered10 supplement, not net-new10','result':x['result_path']} for x in indexed if x['record_type']=='reference_supplement']
    write(out/'new10-reconciliation.json',{'at':now(),'checked_paths':[str(HUMAN/'attempt-index.csv'),str(HUMAN/'scenes'),str(ROOT/'imports/chensiyu-20260901-paired-eda-pilot100-v2/import-manifest.json'),'/share/chensiyu/MobiWAM'],
        'donor_live_path_exists':(Path('/share/chensiyu/MobiWAM')).exists(),'method':'attempt/source/finalized result identity; delivered primary30 and supplement10 baseline; filesystem attempt reconciliation; mtime not identity',
        'existing_primary':sum(x['record_type']=='primary' for x in indexed),'existing_supplement':len(supplement),'indexed_all_records':len(indexed),'filesystem_attempt_count':len(filesystem_attempts),
        'existing_supplement_relationships':supplement,'additional_attempts':extra,'claimed_new10':[{'ordinal':i,'status':'not_found','reason':'no supplied new attempt/source/file identity and no unindexed finalized capture in checked scope'} for i in range(1,11)],
        'net_new_finalized':sum(x['status']=='new' for x in extra),'missing_specific_item':'manifest mapping claimed ten to attempt/source/raw URI; live authorized donor root absent',
        'not_global_absence_claim':True,'unknown_UI_processes_untouched':True})
    media=list(csv.DictReader((SIM/'paper-evidence/all-native-videos.csv').open()))
    mw=[x for x in media if x['task']=='CloseSingleDoor' and x['route']=='E' and x['evaluation']=='final-MW-v2']
    assert len(mw)==3 and all(Path(x['main']).is_file() for x in mw)
    write(out/'SIM-v2-three-MW-E-originals.json',{'at':now(),'index_receipt':str(SIM/'paper-evidence/all-native-videos.csv'),'videos':mw,'existence_verified':True,'checksum_decode_receipts_reused':True})
    summary={'at':now(),'inventory_records':len(allrows),'views':dict(Counter(x['view'] for x in allrows)),
        'policy_demo_candidates':dict(Counter(x['view'] for x in allrows if 'policy-demo-candidate' in x.get('new_use',''))),
        'core_parents':20,'core_split':split['counts'],'new10_net_new':sum(x['status']=='new' for x in extra),'zero_forward_step_replay':True}
    write(out/'summary.json',summary); print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
