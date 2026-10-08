"""Complete denominators, field masks, paired lookup and true online tables."""
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from mobiwam.pi05_data_learning import labels

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def csvwrite(p,rows,fieldnames=None):
 fields=fieldnames or list(dict.fromkeys(k for row in rows for k in row))
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def outcome(r,c,route,tag):
 folder=r/'episodes'/tag/f'slot-{c["global_config_index"]+1:02d}-{c["config_id"]}'/'engineering-attempt-0';p=folder/'completed.json';q=read(p) if p.exists() else {};audit={}
 if q:
  attempt=Path(q['attempt']);af=attempt/'sprint-safety-audit.json'
  if af.exists():audit=read(af)
  else:
   native=attempt/'formal-native-substeps-receipt.json'
   if native.exists():audit=dict(native_collision=read(native).get('forbidden_contact') is not None,all_safety_pass=None)
 inputs=r/'design/repaired-inputs-v1'/c['config_id']/'features.json';feat=read(inputs) if inputs.exists() else {'routes':[]};valid=bool(next((x['hard_valid'] for x in feat['routes'] if x['route_family']==route),False)) and c['static_config_legal']
 status=q.get('status','engineering_unknown' if (folder/'failure.json').exists() else 'X_static_or_route_invalid' if not valid else 'unrun')
 y,mask=labels(q,audit);safe=(bool(q['native_success'] and audit['all_safety_pass']) if q and audit.get('all_safety_pass') is not None else False if mask[0] and not q['native_success'] else None)
 row=dict(parent_group=c['parent_group'],config_id=c['config_id'],family_id=c['family_id'],task=c['task'],role=c['role'],start_slot=c['slot'],tier=c['tier'],route=route,tag=tag,planned=True,hard_valid=valid,
  executed=bool(q and not q['status'].startswith('X_') and (q['steps']>0 or q.get('policy_queries',0)>0)),physics_complete_steps=q.get('steps',0),status=status,task_success=float(y[0]) if mask[0] else None,collision=float(y[1]) if mask[1] else None,progress=float(y[2]) if mask[2] else None,
  base_path_m=float(y[3]) if mask[3] else None,terminal_duration_s=float(y[4]) if mask[4] else None,safety_pass=audit.get('all_safety_pass'),safe_success=safe,
  masks=mask.tolist(),censored=q.get('censored'),receipt=str(p) if q else None,unknown_prefix=str(folder/'failure.json') if (folder/'failure.json').exists() else None,
  route_semantics=q.get('route_semantics_pass'),initial_opening=q.get('initial_native_opening'),terminal_opening=q.get('terminal_native_opening'),policy_queries=q.get('policy_queries',0),steps=q.get('steps',0))
 return row

def aggregate(rows):
 groups={}
 for row in rows:groups.setdefault(row['parent_group'],[]).append(row)
 parents=[]
 for parent,records in sorted(groups.items()):
  denom=len(records);native_known=[x for x in records if x['task_success'] is not None];safe=sum(x['safe_success'] is True for x in records);uncertain=sum((x['safe_success'] is None and not x['status'].startswith('X_')) or (x.get('oracle_missing_valid_routes',False) and x['safe_success'] is not True) for x in records)
  parents.append(dict(parent_group=parent,planned_configurations=denom,executed=sum(x['executed'] for x in records),native_valid=len(native_known),native_success=sum(x['task_success']==1 for x in records),safe_success=safe,
   safe_opportunity_lower=safe/denom,safe_opportunity_upper=(safe+uncertain)/denom,X=sum(x['status'].startswith('X_') for x in records),unrun=sum(x['status']=='unrun' for x in records),unknown_or_pending=uncertain))
 return dict(parent_count=len(parents),planned=len(rows),executed=sum(x['executed'] for x in rows),native_valid=sum(x['task_success'] is not None for x in rows),native_success=sum(x['task_success']==1 for x in rows),
  safe_success=sum(x['safe_success'] is True for x in rows),parent_equal_safe_opportunity_lower=float(np.mean([x['safe_opportunity_lower'] for x in parents])) if parents else None,
  parent_equal_safe_opportunity_upper=float(np.mean([x['safe_opportunity_upper'] for x in parents])) if parents else None,X=sum(x['status'].startswith('X_') for x in rows),unrun=sum(x['status']=='unrun' for x in rows),
  parents=parents,bound_definition='Success credit per planned opportunity; X retained as no executable choice, no fabricated outcome label. Pending/unrun/unknown can raise upper bound. Parent equal weighting.',independent_sample_unit='parent; starts are repeats')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run;out=r/'paper-evidence';out.mkdir(exist_ok=True);final=read(r/'policy/final-evaluation-freeze.json');tier=final['selected_tier'];configs=[c for c in read(r/'design/primary-roster-plan.json')['configurations'] if c['tier']<=tier]
 primary=[];lookup={}
 for c in configs:
  for route in 'EDA':
   row=outcome(r,c,route,f'primary-tier{c["tier"]}-{c["role"]}-{route}');primary.append(row);lookup[(c['config_id'],route)]=row
 (out/'all-primary-outcomes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in primary));csvwrite(out/'all-primary-outcomes.csv',primary)
 paired=[];online=[];multivalid=set()
 for item in final['configurations']:
  c=item['config'];cid=c['config_id']
  if item['at_least_two_hard_valid']:multivalid.add(cid)
  choices={**{'fixed'+route:route if item['valid_routes'].get(route,False) else 'X' for route in 'EDA'},**item['choices']}
  observed=[lookup[(cid,route)] for route in 'EDA' if lookup[(cid,route)]['task_success'] is not None]
  if observed:
   best=min(observed,key=lambda x:(-(x['safe_success'] is True),-(x['task_success']==1),x['base_path_m'] if x['base_path_m'] is not None else float('inf'),x['terminal_duration_s'] if x['terminal_duration_s'] is not None else float('inf'),'EDA'.index(x['route'])))
   choices['observed-oracle']=best['route']
  else:choices['observed-oracle']='X'
  for method,route in choices.items():
   if route in 'EDA' and len(route)==1:row=lookup[(cid,route)].copy()
   else:row=outcome(r,c,'X','unexecuted-choice');row.update(status='X_no_executable_choice',hard_valid=False)
   oracle_missing=method=='observed-oracle' and any(item['valid_routes'].get(rr,False) and lookup[(cid,rr)]['task_success'] is None for rr in 'EDA')
   if method=='observed-oracle' and not observed and any(item['valid_routes'].values()):row.update(status='unrun',hard_valid=True)
   row.update(method=method,comparison='paired_lookup',at_least_two_hard_valid=item['at_least_two_hard_valid'],oracle_missing_valid_routes=oracle_missing,
    oracle_full_route_coverage=not oracle_missing if method=='observed-oracle' else None);paired.append(row)
  for method in final['online_methods']:
   route=item['choices'][method];row=outcome(r,c,route,f'online-tier{tier}-{method}');row.update(method=method,comparison='real_online',at_least_two_hard_valid=item['at_least_two_hard_valid']);online.append(row)
 csvwrite(out/'paired-lookup.csv',paired);csvwrite(out/'real-online.csv',online)
 summary={};parentrows=[];tables=[]
 for name,rows in [('paired_lookup',paired),('real_online',online)]:
  methods=sorted({x['method'] for x in rows});summary[name]={}
  for method in methods:
   records=[x for x in rows if x['method']==method];s=aggregate(records);summary[name][method]=s;tables.append(dict(comparison=name,method=method,**{k:v for k,v in s.items() if k not in ['parents','bound_definition','independent_sample_unit']}))
   for x in s['parents']:parentrows.append(dict(comparison=name,method=method,**x))
  summary[name+'_at_least_two_valid']={method:aggregate([x for x in rows if x['method']==method and x['config_id'] in multivalid]) for method in methods}
 csvwrite(out/'main-table.csv',tables);csvwrite(out/'parent-equal-comparison.csv',parentrows);dump(out/'comparison-summary.json',dict(at=datetime.now(timezone.utc).isoformat(),selected_tier=tier,summary=summary,evaluation_parents=4,families=2,statistical_equivalence_claim=False,template_heldout=False))
 costs=[]
 for comparison,rows in [('paired_lookup',paired),('real_online',online)]:
  methods={x['method'] for x in rows};base={x['config_id']:x for x in rows if x['method']=='MLP'}
  for method in sorted(methods-{'MLP','observed-oracle'}):
   other={x['config_id']:x for x in rows if x['method']==method};common=[cid for cid in base if base[cid]['safe_success'] is True and other[cid]['safe_success'] is True]
   for cid in common:
    x,z=base[cid],other[cid];costs.append(dict(comparison=comparison,comparator=method,parent_group=x['parent_group'],config_id=cid,MLP_path_m=x['base_path_m'],comparator_path_m=z['base_path_m'],path_difference_m=x['base_path_m']-z['base_path_m'],MLP_terminal_seconds=x['terminal_duration_s'],comparator_terminal_seconds=z['terminal_duration_s'],time_difference_seconds=x['terminal_duration_s']-z['terminal_duration_s']))
 csvwrite(out/'common-safe-success-costs.csv',costs,['comparison','comparator','parent_group','config_id','MLP_path_m','comparator_path_m','path_difference_m','MLP_terminal_seconds','comparator_terminal_seconds','time_difference_seconds'])
 failures=[x for x in primary+online if x['executed'] and x['task_success']!=1];csvwrite(out/'all-failure-outcomes.csv',failures)
 flow=[]
 for role in ['train','development','evaluation']:
  for route in 'EDA':
   rows=[x for x in primary if x['role']==role and x['route']==route];flow.append(dict(role=role,route=route,planned=len(rows),hard_valid=sum(x['hard_valid'] for x in rows),executed=sum(x['executed'] for x in rows),native_valid=sum(x['task_success'] is not None for x in rows),X=sum(x['status'].startswith('X_') for x in rows),unrun=sum(x['status']=='unrun' for x in rows),eligible_parents=len({x['parent_group'] for x in rows if x['task_success'] is not None}),families=len({x['family_id'] for x in rows}),success=sum(x['task_success']==1 for x in rows)))
 csvwrite(out/'dataset-flow.csv',flow)
 acoverage={role:len({x['parent_group'] for x in primary if x['role']==role and x['route']=='A' and x['task_success'] is not None}) for role in ['train','development','evaluation']}
 complete_train={c['parent_group'] for c in configs if c['role']=='train' and all(lookup[(c['config_id'],rr)]['task_success'] is not None for rr in 'EDA')}
 media=read(out/'watch-summary.json') if (out/'watch-summary.json').exists() else {};fitted=all((r/'training'/f'tier-{tier}'/kind/'completed.json').exists() for kind in ['MLP','Linear','ridge']);validplanned=sum(x['hard_valid'] for x in primary)
 checks=dict(canonical_demo_admission=read(r/'policy/integrated-corrective-fit2/freeze-verification.json')['frozen_leaves_unchanged'],selected_tier_all_planned_accounted=len(primary)==(180 if tier==1 else 360),all_legal_primary_native_fields=sum(x['task_success'] is not None for x in primary)==validplanned,
  A_train_parents_at_least6=acoverage['train']>=6,A_development_parents_at_least2=acoverage['development']>=2,A_evaluation_parents_at_least2=acoverage['evaluation']>=2,train_full_EDA_parents_at_least6=len(complete_train)>=6,models_final_fixed=fitted,
  all_legal_online_fields=all(x['task_success'] is not None or x['status'].startswith('X_') for x in online),native_media_all_complete=all((out/'watch-1x'/(x['tag']+'-slot-'+str(next(c['global_config_index']+1 for c in configs if c['config_id']==x['config_id'])))/'watch-binding.json').exists() for x in primary+online if x['physics_complete_steps']>0),all_executed_safety_closed=all(x['safety_pass'] is not None for x in primary+online if x['executed']))
 readiness=dict(at=datetime.now(timezone.utc).isoformat(),paper_evidence_ready=all(checks.values()),checks=checks,A_coverage=acoverage,full_EDA_train_parents=len(complete_train),A_coverage_status='complete_minimum' if all(acoverage[k]>=v for k,v in [('train',6),('development',2),('evaluation',2)]) else 'A_coverage_incomplete',human_review='pending',formal_train_ready=False,Gate='historical_unchanged',claim_review='Research pending',complete_success_demo_count=sum(x['safe_success'] is True for x in primary+online))
 dump(out/'paper-readiness.json',readiness);print(json.dumps(readiness),flush=True)
if __name__=='__main__':main()
