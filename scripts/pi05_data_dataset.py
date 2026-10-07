"""Freeze reliable current-policy train/dev labels; never open evaluation."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from mobiwam.pi05_data_learning import HEADS,labels

def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--tier',type=int,choices=[1,2],required=True);a=ap.parse_args()
    roster=json.loads((a.run/'design/primary-roster-plan.json').read_text());split=json.loads((a.run/'inventory/source-split.json').read_text())
    out=a.run/'training'/f'tier-{a.tier}';out.mkdir(exist_ok=True)
    if (out/'dataset-binding.json').exists():raise ValueError('existing dataset binding is immutable')
    rows=[];eligibility=[];byrole={'train':[],'development':[]}
    for config in roster['configurations']:
        if config['tier']>a.tier or config['role']=='evaluation':continue
        inputs=a.run/'design/repaired-inputs-v1'/config['config_id'];features=json.loads((inputs/'features.json').read_text()) if (inputs/'features.json').exists() else {'routes':[]}
        candidates={x['route_family']:x for x in features['routes']}
        for route in 'EDA':
            valid=config['static_config_legal'] and bool(candidates.get(route,{}).get('hard_valid',False))
            tag=f'primary-tier{config["tier"]}-{config["role"]}-{route}'
            p=a.run/'episodes'/tag/f'slot-{config["global_config_index"]+1:02d}-{config["config_id"]}'/'engineering-attempt-0/completed.json'
            receipt=json.loads(p.read_text()) if p.exists() else {};audit={}
            if receipt:
                af=Path(receipt['attempt'])/'sprint-safety-audit.json'
                if af.exists():audit=json.loads(af.read_text())
                assert receipt['config_id']==config['config_id'] and receipt['parent_group']==config['parent_group']
                assert receipt['protocol']=='PI05-DATA-v1' and receipt['declared_sim_horizon_seconds']==300
                policy=json.loads((p.parent/'policy-binding.json').read_text())
                freeze=json.loads((a.run/'policy/main-component-freeze.json').read_text())
                assert Path(policy['checkpoint']).resolve()==Path(freeze['policy_checkpoint']).resolve()
                init=json.loads((p.parent/'initialization.json').read_text());assert init['restore']['max_abs_error']<=1e-6
            y,mask=labels(receipt,audit)
            X=np.load(inputs/(route+'-X.npy'),allow_pickle=False) if valid else None
            row=dict(parent_group=config['parent_group'],config_id=config['config_id'],family_id=config['family_id'],task=config['task'],role=config['role'],slot=config['slot'],tier=config['tier'],route=route,
                hard_valid=valid,planned=True,executed=bool(receipt),status=receipt.get('status','X_static_or_route_invalid' if not valid else 'unrun'),
                task_success=float(y[0]) if mask[0] else None,collision=float(y[1]) if mask[1] else None,progress=float(y[2]) if mask[2] else None,
                base_path_m=float(y[3]) if mask[3] else None,terminal_duration_s=float(y[4]) if mask[4] else None,
                initial_native_opening=receipt.get('initial_native_opening'),terminal_native_opening=receipt.get('terminal_native_opening'),
                opening_change=(receipt['initial_native_opening']-receipt['terminal_native_opening']) if 'terminal_native_opening' in receipt else None,
                observation_horizon_s=300,censored=receipt.get('censored'),safety_pass=audit.get('all_safety_pass'),
                safe_qualified_success=bool(receipt.get('native_success') and audit.get('all_safety_pass')) if audit else None,
                masks=mask.tolist(),raw_receipt=str(p) if receipt else None,safety_receipt=str(Path(receipt['attempt'])/'sprint-safety-audit.json') if receipt else None,
                input_receipt=str(inputs/'features.json'),old_reference_or_policy_auxiliary=False)
            rows.append(row)
            for j,name in enumerate(HEADS):eligibility.append(dict(parent_group=row['parent_group'],config_id=row['config_id'],route=route,head=name,admitted=bool(mask[j]),reason='reliable actual terminal field' if mask[j] else row['status']+'; missing/unknown field',label=row.get(['task_success','collision','progress','base_path_m','terminal_duration_s'][j])))
            if X is not None:byrole[config['role']].append((row,X,y,mask))
    for role,records in byrole.items():
        assert records
        np.savez(out/('train-only.npz' if role=='train' else 'development-only.npz'),X=np.stack([x[1] for x in records]),y=np.stack([x[2] for x in records]),mask=np.stack([x[3] for x in records]),
            parent_group=np.array([x[0]['parent_group'] for x in records]),config_id=np.array([x[0]['config_id'] for x in records]),route=np.array([x[0]['route'] for x in records]),
            role=np.array([role]*len(records)),hard_valid=np.array([x[0]['hard_valid'] for x in records]))
    (out/'labels.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with (out/'eligibility-by-head.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(eligibility[0]));w.writeheader();w.writerows(eligibility)
    binding=dict(at=now(),tier=a.tier,controller='frozen_pi05',rows=len(rows),planned_train_routes=sum(x['role']=='train' for x in rows),planned_dev_routes=sum(x['role']=='development' for x in rows),
        executed=sum(x['executed'] for x in rows),hard_valid=sum(x['hard_valid'] for x in rows),valid_label_by_head={h:sum(x['masks'][j] for x in rows) for j,h in enumerate(HEADS)},
        reserved_development_evaluation_ancestors=split['reserved_development_evaluation_ancestors'],source_split=str(a.run/'inventory/source-split.json'),
        main_component_freeze=str(a.run/'policy/main-component-freeze.json'),eval_outcomes_opened=False,old_incompatible_auxiliary_admitted=0,
        time_scale_seconds=300,path_scale_m=2,progress_definition='clip(1-native_opening,0,1); relative change separately reported',source_hierarchy='parent/config/route per-head equal')
    write(out/'dataset-binding.json',binding);print(json.dumps(binding),flush=True)

if __name__=='__main__':main()
