"""New frozen pi05 train/dev labels only, with explicit missing masks."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from mobiwam.sim_sprint_learning import require_group_split,ROUTES

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
    frozen=json.loads((r/'policy/frozen-public-components.json').read_text());roster=json.loads((r/'data/paired-source-roster.json').read_text())
    records=[]
    for receipt in (r/'episodes').glob('paired-v6-train-dev-*/slot-*/engineering-attempt-0/completed.json'):
        q=json.loads(receipt.read_text());assert q['purpose']=='paired' and q['adapter_version']=='v6' and q['checkpoint_step']==2000
        slot=roster['slots'][q['slot']-1];assert slot['role'] in ('train','dev') and slot['parent_group']==q['parent_group']
        binding=json.loads((receipt.parent/'policy-binding.json').read_text());assert binding['checkpoint']==frozen['policy_checkpoint']
        attempt=Path(q['attempt']);monitor=json.loads((attempt/'formal-native-substeps-receipt.json').read_text())
        z=np.load(attempt/'formal-native-substeps.npz',allow_pickle=False)
        Xpath=r/'inputs/geometry-prefix-v2'/q['config_id']/(q['route']+'-X.npy')
        if not Xpath.exists():continue
        y=np.full(5,np.nan,np.float32);mask=np.zeros(5,bool)
        if q['status']!='X_no_legal_candidate' and q['usable_scientific_outcome']:
            y[0]=q['native_success'];y[1]=monitor.get('forbidden_contact') is not None;mask[:2]=True
            trace=attempt/'trace.jsonl'
            if q['steps'] and trace.exists():
                lines=trace.read_text().splitlines();first=json.loads(lines[0]);last=json.loads(lines[-1])
                before=float(first['before']['target']['door']);after=float(last['after']['target']['door'])
                if before>1e-6:y[2]=np.clip((before-after)/before,0,1);mask[2]=True
            elif len(z['phases'])==0:y[2]=0.;mask[2]=True
            # PandaOmron's two allocation-native base slide coordinates are
            # verified by the preflight/controller binding; no fixture coords.
            y[3]=float(np.linalg.norm(np.diff(z['qpos'][:,:2],axis=0),axis=1).sum())
            y[4]=float(z['sim_time'][-1]-z['sim_time'][0]);mask[3:]=True
        audit=attempt/'sprint-safety-audit.json';safe=json.loads(audit.read_text()) if audit.exists() else {}
        records.append(dict(slot=q['slot'],route=q['route'],role=slot['role'],parent_group=q['parent_group'],config_id=q['config_id'],family_id=q['family_id'],
            task=q['task'],status=q['status'],X=np.load(Xpath,allow_pickle=False),y=y,mask=mask,receipt=str(receipt),
            safety_qualified_success=safe.get('safety_qualified_success'),feature_sha256=hashlib.sha256(Xpath.read_bytes()).hexdigest()))
    present={(x['parent_group'],x['route']) for x in records}
    for i,slot in enumerate(roster['slots']):
        if slot['task'] not in frozen['task_scope'] or slot['role'] not in ('train','dev'):continue
        for route in ROUTES:
            if (slot['parent_group'],route) in present:continue
            path=r/'inputs/geometry-prefix-v2'/slot['config_id']/(route+'-X.npy')
            if not path.exists():continue
            records.append(dict(slot=i+1,route=route,role=slot['role'],parent_group=slot['parent_group'],config_id=slot['config_id'],family_id=slot['family_id'],
                task=slot['task'],status='unrun_no_label',X=np.load(path,allow_pickle=False),y=np.full(5,np.nan,np.float32),mask=np.zeros(5,bool),
                receipt=None,safety_qualified_success=None,feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    train=[x for x in records if x['role']=='train'];dev=[x for x in records if x['role']=='dev']
    groups={x['parent_group'] for x in train if x['mask'].any()};routes={x['route'] for x in train if x['mask'].any()}
    if len(groups)<4 or len(routes)<2:raise ValueError(f'insufficient new pi05 supervision: train parents{len(groups)}, routes{sorted(routes)}')
    require_group_split([x['parent_group'] for x in train],[x['parent_group'] for x in dev])
    out=r/'training';out.mkdir(exist_ok=True);assert not (out/'dataset-binding.json').exists()
    for name,rows in [('train-only',train),('development-validation',dev)]:
        if not rows:continue
        np.savez_compressed(out/(name+'.npz'),X=np.stack([x['X'] for x in rows]),y=np.stack([x['y'] for x in rows]),mask=np.stack([x['mask'] for x in rows]),
            group_id=np.array([x['parent_group'] for x in rows]),route=np.array([x['route'] for x in rows]),task=np.array([x['task'] for x in rows]))
    binding=dict(at=datetime.now(timezone.utc).isoformat(),controller='frozen_pi05',policy_checkpoint=frozen['policy_checkpoint'],public_components='policy/frozen-public-components.json',
        feature_version='geometry-prefix-v2',schema='1024 context+21 measured geometric prefix+3 route indicators',train_parent_groups=len(groups),supervised_train_routes=sorted(routes),
        train_records=len(train),dev_records=len(dev),final_labels_accessed=False,old_outcomes_used=False,unknown_labels='NaN and mask false; no outcome substitution',
        scaler_input='all three preoutcome routes of train-only participating parents; includes masked unrun rows, no outcome fill',
        records=[{k:v for k,v in x.items() if k not in ('X','y','mask')}|dict(y=[float(v) if m else None for v,m in zip(x['y'],x['mask'])],mask=x['mask'].tolist()) for x in records],
        train_sha256=hashlib.sha256((out/'train-only.npz').read_bytes()).hexdigest(),formal_train_ready=False)
    (out/'dataset-binding.json').write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps({k:v for k,v in binding.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
