"""Freeze model predictions before any prospective final outcome."""
import argparse,hashlib,json,subprocess,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from mobiwam.sim_sprint_learning import ActiveHead,scale,transform,select,ridge
from mobiwam.scene004 import geometry_rule_select

def physical_transform(raw,heads):
    values=transform(raw,heads)
    for j,h in enumerate(heads):
        if h['status']!='learnable':values[:,j]=np.nan if h['mean'] is None else h['mean']
    return values

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;out=r/'evaluation';out.mkdir(exist_ok=True)
    assert not (out/'final-freeze.json').exists()
    assert not list((r/'episodes').glob('paired-v6-final-*/slot-*/engineering-attempt-*/completed.json'))
    assert not list((r/'episodes').glob('online-v6-*/slot-*/engineering-attempt-*/completed.json'))
    binding=json.loads((r/'training/dataset-binding.json').read_text());z=np.load(r/'training/train-only.npz',allow_pickle=False)
    release=json.loads((r/'policy/harness-route-release.json').read_text());roster=json.loads((r/'data/paired-source-roster.json').read_text())
    models={};configs={}
    for name in ('MLP','Linear'):
        complete=r/'training'/name/'completed.json';assert complete.exists()
        path=r/'training'/name/'step2000.pt';cp=torch.load(path,map_location='cpu');assert cp['step']==2000 and cp['config']['train_sha256']==binding['train_sha256']
        net=ActiveHead(name,cp['active']);net.load_state_dict(cp['model']);net.eval();models[name]=(net,cp);configs[name]=dict(checkpoint=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),active=cp['active'],heads=cp['head_support'])
    # Fixed-route choice uses the frozen train supervision only. Unknown safe
    # qualification on native successes must be resolved before this freeze.
    byroute={x:[] for x in 'EDA'}
    for rec in binding['records']:
        if rec['role']!='train' or rec['receipt'] is None:continue
        receipt=Path(rec['receipt']);q=json.loads(receipt.read_text());audit=Path(q['attempt'])/'sprint-safety-audit.json'
        if q['native_success'] and not audit.exists():raise ValueError('train-best-fixed needs completed success safety audit')
        safe=json.loads(audit.read_text())['safety_qualified_success'] if audit.exists() else False
        byroute[rec['route']].append(dict(safe=bool(safe),path=rec['y'][3],duration=rec['y'][4]))
    available=[x for x in 'EDA' if byroute[x]]
    fixed=min(available,key=lambda route:(-sum(x['safe'] for x in byroute[route]),
        np.mean([x['path'] for x in byroute[route] if x['safe']]) if any(x['safe'] for x in byroute[route]) else float('inf'),
        np.mean([x['duration'] for x in byroute[route] if x['safe']]) if any(x['safe'] for x in byroute[route]) else float('inf'),'EDA'.index(route)))
    final=[(i+1,x) for i,x in enumerate(roster['slots']) if x['role']=='final' and x['task'] in release['tasks'] and release['tasks'][x['task']]['E']['released']]
    predictions=[];ridge_fits=None
    for slot,g in final:
        d=r/'inputs/geometry-prefix-v2'/g['config_id'];features=json.loads((d/'features.json').read_text());rows={x['route_family']:x for x in features['routes']}
        X=np.stack([np.load(d/(route+'-X.npy'),allow_pickle=False) for route in 'EDA'])
        eligible=[bool(rows.get(route,{}).get('hard_valid') and release['tasks'][g['task']].get(route,{}).get('released')) for route in 'EDA']
        chosen={};predicted={}
        for name,(net,cp) in models.items():
            with torch.no_grad():raw=net(torch.from_numpy(scale(X,np.asarray(cp['scaler']['mean']),np.asarray(cp['scaler']['std'])))).numpy()
            values=physical_transform(raw,cp['head_support']);idx=select(values,eligible,cp['active']);chosen[name]='EDA'[idx] if idx is not None else 'X';predicted[name]=[[float(v) if np.isfinite(v) else None for v in row] for row in values]
        heads=models['MLP'][1]['head_support'];values,raw,fit=ridge(z['X'],z['y'],z['mask'],z['group_id'],X,heads)
        ridge_fits=fit
        for j,h in enumerate(heads):
            if h['status']!='learnable':values[:,j]=np.nan if h['mean'] is None else h['mean']
        active=[i for i,h in enumerate(heads) if h['status']=='learnable'];idx=select(values,eligible,active);chosen['ridge']='EDA'[idx] if idx is not None else 'X';predicted['ridge']=[[float(v) if np.isfinite(v) else None for v in row] for row in values]
        geometry=[dict(rows[route],stage_eligible=eligible[i]) for i,route in enumerate('EDA') if route in rows]
        choice=geometry_rule_select(geometry);chosen['geometry']='X' if choice=='X' else choice.split('-',1)[0];chosen['train-best-fixed']=fixed
        predictions.append(dict(slot=slot,config_id=g['config_id'],parent_group=g['parent_group'],family_id=g['family_id'],task=g['task'],eligible=eligible,selected=chosen,predictions=predicted,
            feature_sha256={route:hashlib.sha256((d/(route+'-X.npy')).read_bytes()).hexdigest() for route in 'EDA'}))
    bound_paths=['policy/frozen-public-components.json','policy/collection-compatibility.json','policy/harness-route-release.json','data/paired-source-roster.json','training/dataset-binding.json',
        'runtime/mobipi-v6-collection/scripts/pi05_harness_episode.py','runtime/mobipi-v6-collection/src/mobiwam/pi05_adapter.py','runtime/mobipi-v6-collection/src/mobiwam/pi05_grip_projection.py',
        'runtime/mobipi-v6-collection/src/mobiwam/pi05_motion.py','runtime/mobipi/src/mobiwam/sim_sprint_learning.py','runtime/mobipi/src/mobiwam/scene004.py',
        'runtime/mobipi/scripts/pi05_comparison_inputs.py','runtime/mobipi/scripts/pi05_final_queue.py','runtime/mobipi/scripts/sim_sprint_safety.py','runtime/mobipi/src/mobiwam/pi05_outcome_metrics.py']
    provenance=dict(file_sha256={path:hashlib.sha256((r/path).read_bytes()).hexdigest() for path in bound_paths},
        code_commits={name:subprocess.check_output(['git','-C',str(r/'runtime'/name),'rev-parse','HEAD'],text=True).strip() for name in ['mobipi','mobipi-v6','mobipi-v6-collection','openpi']},
        inference_weights=json.loads((r/'preflight/frozen-policy-files.json').read_text()),
        policy_components=json.loads((r/'policy/frozen-public-components.json').read_text()),
        seed=dict(environment='inherited exact physical Source',policy_sampling=20261006,evaluation=20261006,training=17))
    freeze=dict(at=datetime.now(timezone.utc).isoformat(),policy='policy/frozen-public-components.json',OBC_training_binding='training/dataset-binding.json',models=configs,
        released_routes=release,source_roster='data/paired-source-roster.json',final_slots=[x['slot'] for x in predictions],train_best_fixed=fixed,predictions=predictions,
        provenance=provenance,ridge=dict(alpha=1.,input_width=21,head_fits=ridge_fits,heads=models['MLP'][1]['head_support'],train_sha256=binding['train_sha256']),
        train_best_fixed_coverage={route:len(values) for route,values in byroute.items()},
        scope='prospective held-out known-development; partial task/routes declared',final_outcomes_accessed=False,
        maximum_online_episodes=24,online_methods=['MLP','geometry','train-best-fixed'],method_order='cyclic by Source index',formal_train_ready=False)
    pending=out/'.final-freeze.pending.json'
    with pending.open('x') as f:
        f.write(json.dumps(freeze,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(pending,out/'final-freeze.json')
    print(json.dumps(dict(final_slots=freeze['final_slots'],train_best_fixed=fixed,selections=[x['selected'] for x in predictions])),flush=True)
if __name__=='__main__':main()
