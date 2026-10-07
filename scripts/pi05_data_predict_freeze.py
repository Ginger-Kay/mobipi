"""Freeze final-tier predictions and choices before evaluation outcomes exist."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mobiwam.pi05_data_learning import Head,scale,decode,choose,SCALES

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--tier',type=int,choices=[1,2],required=True);a=ap.parse_args();r=a.run;data=r/'training'/f'tier-{a.tier}'
    if (r/'policy/final-evaluation-freeze.json').exists():raise ValueError('evaluation already frozen')
    assert not list((r/'episodes').glob('primary-tier*-evaluation-*/*/engineering-attempt-*/completed.json'))
    decision=read(r/'design/tier-decision.json');assert decision['selected_tier']==a.tier
    parent_roster=read(r/'design/primary-roster-plan.json');configurations=[g for g in parent_roster['configurations'] if g['role']=='evaluation' and g['tier']<=a.tier]
    methods={};models={}
    for kind in ['MLP','Linear']:
        cp=data/kind/'step2000.pt';assert read(data/kind/'completed.json')['steps']==2000
        state=torch.load(cp,map_location='cpu',weights_only=False);model=Head(kind);model.load_state_dict(state['model']);model.eval();models[kind]=(model,state);methods[kind]={'checkpoint':str(cp),'sha256':sha(cp),'heads':state['heads']}
    ridge=read(data/'ridge/ridge.json');methods['ridge']={'checkpoint':str(data/'ridge/ridge.json'),'sha256':sha(data/'ridge/ridge.json'),'heads':ridge['recipe']['heads']}
    labels=[json.loads(x) for x in (data/'labels.jsonl').read_text().splitlines()];train=[x for x in labels if x['role']=='train'];parents=sorted({x['parent_group'] for x in train});bounds={}
    for route in 'EDA':
        lower=[];upper=[];costs=[]
        for parent in parents:
            rows=[x for x in train if x['parent_group']==parent and x['route']==route];lo=[];hi=[]
            for x in rows:
                known=x['task_success']==0 or (x['task_success']==1 and x['safety_pass'] is not None)
                val=bool(x['task_success']==1 and x['safety_pass'] is True)
                lo.append(float(val) if known else 0.);hi.append(float(val) if known else 1.)
                if val:costs.append((x['base_path_m'],x['terminal_duration_s']))
            lower.append(float(np.mean(lo)));upper.append(float(np.mean(hi)))
        bounds[route]=dict(lower=float(np.mean(lower)),upper=float(np.mean(upper)),mean_success_path=float(np.mean([x[0] for x in costs])) if costs else None,
            mean_success_duration=float(np.mean([x[1] for x in costs])) if costs else None,missing_results_not_dropped=True)
    decisive=[route for route in 'EDA' if all(bounds[route]['lower']>=bounds[other]['upper'] for other in 'EDA' if other!=route)]
    if decisive:
        trainbest=min(decisive,key=lambda route:(-bounds[route]['lower'],bounds[route]['mean_success_path'] or 0.,bounds[route]['mean_success_duration'] or 0.,'EDA'.index(route)))
    else:trainbest='E'
    predictions=[]
    for g in configurations:
        folder=r/'design/repaired-inputs-v1'/g['config_id'];feature=read(folder/'features.json') if (folder/'features.json').exists() else {'routes':[],'geometry_selection':None}
        valid={route:bool(next((x['hard_valid'] for x in feature['routes'] if x['route_family']==route),False)) for route in 'EDA'}
        x={route:np.load(folder/(route+'-X.npy')) for route in 'EDA' if valid[route]};values={};choices={}
        for kind,(model,state) in models.items():
            per={}
            for route,X in x.items():
                scaled=scale(X[None],state['scaler']['mean'],state['scaler']['std'])
                with torch.inference_mode():raw=model(torch.from_numpy(scaled)).numpy()
                per[route]=decode(raw,state['heads'])[0].tolist()
            values[kind]=per;choices[kind]=choose(per,valid,state['heads'])
        per={};recipe=ridge['recipe'];heads=recipe['heads']
        for route,X in x.items():
            xs=scale(X[None],np.asarray(recipe['mean']),np.asarray(recipe['std']))[0];pred=np.full(5,np.nan)
            for j,h in enumerate(heads):
                if h['status']=='constant':pred[j]=h['constant'];continue
                if h['status']=='unknown':continue
                fit=ridge['fits'][str(j)];v=float(xs[1024:1045]@np.asarray(fit['coef'])+fit['intercept'])
                pred[j]=(np.clip(v,0,1) if j<3 else max(v,0))*SCALES[j]
            per[route]=pred.tolist()
        values['ridge']=per;choices['ridge']=choose(per,valid,heads)
        geometry=feature.get('geometry_selection');geometry=geometry.get('route_family',geometry.get('route','X')) if isinstance(geometry,dict) else geometry
        geometry={x['candidate_id']:x['route_family'] for x in feature['routes']}.get(geometry,geometry)
        choices['geometry']=geometry if geometry in ('E','D','A') and valid.get(geometry,False) else 'X'
        choices['train-best-fixed']=trainbest if valid.get(trainbest,False) else 'X'
        predictions.append(dict(config=g,valid_routes=valid,at_least_two_hard_valid=sum(valid.values())>=2,predictions=values,choices=choices,
            fixed_routes={route:route if valid.get(route,False) else 'X' for route in 'EDA'},features_receipt=str(folder/'features.json')))
    frozen=dict(at=datetime.now(timezone.utc).isoformat(),status='frozen_before_any_evaluation_outcome',selected_tier=a.tier,methods=methods,
        train_best_fixed=trainbest,train_best_bounds=bounds,train_best_selection_supported=bool(decisive),unsupported_train_best_fallback='E by frozen tie order',
        configurations=predictions,planned_evaluation_configs=12 if a.tier==1 else 24,planned_primary_routes=36 if a.tier==1 else 72,
        planned_online_routes=36 if a.tier==1 else 72,online_methods=['MLP','geometry','train-best-fixed'],
        primary_routes=['E','D','A'],comparison_methods=['fixedE','fixedD','fixedA','geometry','train-best-fixed','ridge','Linear','MLP','observed-oracle'],
        statistical_cluster='parent; configurations are repeated starts, not independent Sources',main_component_freeze=str(r/'policy/main-component-freeze.json'),
        evaluation_outcomes_accessed=False,post_evaluation_fit_or_expansion_prohibited=True)
    (r/'policy/final-evaluation-freeze.json').write_text(json.dumps(frozen,indent=2)+'\n');print(json.dumps({k:frozen[k] for k in ['at','selected_tier','train_best_fixed','planned_evaluation_configs']}),flush=True)

if __name__=='__main__':main()
