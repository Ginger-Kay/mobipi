"""Final-only predictions and complete paired-outcome lookup, never online claims."""
import argparse,csv,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from mobiwam.sim_sprint_learning import *

def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def csvwrite(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--reuse-neural',action='store_true');a=ap.parse_args();r=a.run
    cached=np.load(r/'comparison/final-predictions.npz',allow_pickle=False) if a.reuse_neural else None
    prior_receipt=json.loads((r/'comparison/final-prediction-receipt.json').read_text()) if a.reuse_neural else None
    out=r/'comparison' if not a.reuse_neural else r/'comparison/corrected-scaled-B3-v2'
    if a.reuse_neural:out.mkdir(exist_ok=False)
    train=np.load(r/'training/train-only.npz',allow_pickle=False)
    inherited=json.loads((r/'data/inventory.json').read_text());records={(x['config_id'],x['route']):x for x in inherited if x['view']=='A_reference_auto'}
    # Train-best-fixed is frozen before loading validation labels or predictions.
    common=[]
    for group in sorted(set(train['group_id'])):
        ids=np.flatnonzero(train['group_id']==group)
        if len(ids)==3 and bool(train['mask'][ids][:,[0,3,4]].all()):common.append(group)
    train_scores=[]
    for route in ROUTES:
        ids=np.flatnonzero(train['route']==route);known=[i for i in ids if train['group_id'][i] in common]
        successful=[i for i in known if train['y'][i,0]==1 and str(records[(str(train['group_id'][i]),route)]['clearance_qualification']).startswith('pass')]
        path=float(np.mean([train['y'][i,3] for i in successful if train['mask'][i,3]])) if successful else float('inf')
        duration=float(np.mean([train['y'][i,4] for i in successful if train['mask'][i,4]])) if successful else float('inf')
        train_scores.append(dict(route=route,safety_qualified_success=len(successful),known=len(known),success_path_m=path,success_duration_s=duration))
    best=min(train_scores,key=lambda x:(-x['safety_qualified_success'],x['success_path_m'],x['success_duration_s'],ROUTES.index(x['route'])))['route']
    selector=dict(created_at=datetime.now(timezone.utc).isoformat(),train_best_fixed=best,train_scores=train_scores,
        train_best_common_sources=[str(x) for x in common],missing_sources_excluded_from_fixed_selection=sorted(set(map(str,train['group_id']))-set(map(str,common))),
        thresholds=[.05,.05,.05,.02,1.],tie='E<D<A',classification_heads='supported learnable only; constant collision excluded',
        primary_metric='native success AND existing clearance pass; separately from native success',comparison_scope='paired-outcome lookup on already observed development sources')
    write(out/'selector-freeze.json',selector)
    validation=np.load(r/'training/development-validation.npz',allow_pickle=False)
    X=np.concatenate([train['X'],validation['X']]);y=np.concatenate([train['y'],validation['y']]);mask=np.concatenate([train['mask'],validation['mask']])
    groups=np.concatenate([train['group_id'],validation['group_id']]);routes=np.concatenate([train['route'],validation['route']]);tasks=np.concatenate([train['task'],validation['task']])
    splits=np.array(['train']*len(train['X'])+['development-validation']*len(validation['X']))
    predictions={};raws={};latencies={}
    for name in ('MLP','Linear'):
        cp=torch.load(r/'training'/name/'step2000.pt',map_location='cpu')
        if cached is not None:
            assert np.array_equal(cached['group_id'],groups) and np.array_equal(cached['route'],routes)
            raw=cached[name+'_raw'];latencies[name]=prior_receipt['GPU_forward_seconds'][name]
        else:
            model=ActiveHead(name,cp['active']);model.load_state_dict(cp['model']);model.cuda().eval()
            tx=torch.from_numpy(scale(X,cp['scaler']['mean'],cp['scaler']['std'])).cuda()
            torch.cuda.synchronize();t=time.monotonic()
            with torch.no_grad():raw=model(tx).cpu().numpy()
            torch.cuda.synchronize();latencies[name]=time.monotonic()-t
        predictions[name]=transform(raw,cp['head_support']);raws[name]=raw
        if name=='MLP':heads=cp['head_support'];active=cp['active'];scaler=cp['scaler']
    train_scaled=scale(train['X'],scaler['mean'],scaler['std']);all_scaled=scale(X,scaler['mean'],scaler['std'])
    b3,raw3,fits=ridge(train_scaled,train['y'],train['mask'],train['group_id'],all_scaled,heads);predictions['B3']=b3;raws['B3']=raw3
    write(out/'B3-fit.json',dict(input='21 geometry from same train-only scaler as neural heads',fits=fits))
    np.savez_compressed(out/'final-predictions.npz',group_id=groups,route=routes,split=splits,**predictions,
        **{name+'_raw':raw for name,raw in raws.items()})
    feature_sources={x['group_id']:x for x in json.loads((r/'data/reference-sources.json').read_text())}
    selections=[];all_predictions=[]
    for group in sorted(set(groups)):
        idx=[int(np.flatnonzero((groups==group)&(routes==route))[0]) for route in ROUTES]
        metadata=feature_sources[str(group)];plan=json.loads(Path(metadata['candidate_features']).read_text())
        eligible=[bool(x['hard_valid']) for x in plan['records']]
        geom=metadata['geometry_preoutcome_choice']
        methods={'fixed_E':'E','fixed_D':'D','fixed_A':'A','train_best_fixed':best,'geometry':geom}
        for name,pred in predictions.items():
            chosen=select(pred[idx],eligible,active);methods[name]='X' if chosen is None else ROUTES[chosen]
        # Oracle only ranks routes with known success; future labels never enter learned features.
        oracle_candidates=[j for j,i in enumerate(idx) if mask[i,0]]
        def oracle_key(j):
            i=idx[j];record=records[(str(group),ROUTES[j])];safe=bool(y[i,0] and str(record['clearance_qualification']).startswith('pass'))
            return (-int(safe),-y[i,0],-y[i,2] if mask[i,2] else 0,y[i,3] if mask[i,3] else float('inf'),y[i,4] if mask[i,4] else float('inf'),j)
        methods['oracle']=ROUTES[min(oracle_candidates,key=oracle_key)] if oracle_candidates else 'X'
        for method,route in methods.items():
            i=idx[ROUTES.index(route)] if route in ROUTES else None;known=bool(i is not None and mask[i,0]);record=records.get((str(group),route),{})
            successful=bool(known and y[i,0]==1);safe=successful and str(record.get('clearance_qualification','')).startswith('pass')
            selections.append(dict(group_id=str(group),task=str(tasks[idx[0]]),split=str(splits[idx[0]]),method=method,route=route,
                native_success=int(successful) if known else '',safety_qualified_success=int(safe) if known else '',outcome_known=known,
                native_collision=float(y[i,1]) if i is not None and mask[i,1] else '',progress=float(y[i,2]) if i is not None and mask[i,2] else '',
                base_path_m=float(y[i,3]) if i is not None and mask[i,3] else '',terminal_duration_s=float(y[i,4]) if i is not None and mask[i,4] else '',
                success_duration_s=float(y[i,4]) if successful and mask[i,4] else '',failure_type=record.get('stop_reason','X'),
                scope='paired-outcome lookup; no new per-method simulation',label_missing_reason=record.get('exclusion_reason','')))
        for i in idx:
            for name,pred in predictions.items():all_predictions.append(dict(group_id=str(group),split=str(splits[i]),route=str(routes[i]),model=name,
                **{HEADS[j]:float(pred[i,j]) for j in range(5)}))
    csvwrite(out/'route-selections.csv',selections);csvwrite(out/'predictions.csv',all_predictions)
    table=[]
    for split in ('train','development-validation'):
        for method in methods:
            rows=[x for x in selections if x['split']==split and x['method']==method];known=[x for x in rows if x['outcome_known']];successful=[x for x in known if x['native_success']]
            table.append(dict(split=split,method=method,total_sources=len(rows),known_outcomes=len(known),missing=len(rows)-len(known),
                native_success=sum(x['native_success'] for x in known),safety_qualified_success=sum(x['safety_qualified_success'] for x in known),
                native_collisions=sum(x['native_collision'] for x in known if x['native_collision']!=''),
                success_duration_mean_s=float(np.mean([x['success_duration_s'] for x in successful if x['success_duration_s']!=''])) if successful else '',
                base_path_mean_m=float(np.mean([x['base_path_m'] for x in known if x['base_path_m']!=''])) if known else '',
                scope='offline paired-outcome lookup, development validation reused'))
    csvwrite(out/'baselines.csv',table);write(out/'baselines.json',table)
    write(out/'final-prediction-receipt.json',dict(created_at=datetime.now(timezone.utc).isoformat(),predictions_per_neural_model=0 if a.reuse_neural else 1,
        cumulative_neural_predictions_per_model=1,reused_neural_predictions=a.reuse_neural,B3_valid_scientific_fit_count=1,
        neural_fit_count=2,selector=selector,rows=len(X),source_count=len(set(groups)),GPU_forward_seconds=latencies,
        new_GPU_forward_seconds=0. if a.reuse_neural else sum(latencies.values()),forward_times_are_inherited=a.reuse_neural,
        input_dimensions=1048,validation_reads_after_fixed_final=True,collision_positive_support=heads[1]['positive'],formal_train_ready=False))
    write(r/'comparison/current.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),comparison=str(out),
        previous_invalid_B3=str(r/'comparison') if a.reuse_neural else None,
        reason='restore R3 train-only standardized geometry recipe; raw-geometry B3 invalidated' if a.reuse_neural else 'first final comparison'))
    print(json.dumps([x for x in table if x['split']=='development-validation']),flush=True)

if __name__=='__main__':main()
