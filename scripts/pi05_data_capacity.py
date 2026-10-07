"""Resource-only estimates and once-only tier selection, never score based."""
import argparse
from datetime import datetime,timezone,timedelta
import json,os
from pathlib import Path
import shutil

def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc)
def wall(q):return (datetime.fromisoformat(q['ended_at'])-datetime.fromisoformat(q['started_at'])).total_seconds()
def directory_bytes(p):
 return sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--decide-tier',action='store_true');a=ap.parse_args();r=a.run
 manifest=read(r/'run-manifest.json');rules=read(r/'design/capacity-rule-freeze.json');roster=read(r/'design/primary-roster-plan.json')['configurations'];byid={x['config_id']:x for x in roster};deadline=datetime.fromisoformat(manifest['hard_science_deadline']);stamp=now()
 completed=[]
 # For capacity read only process durations and storage, never success/loss.
 for f in (r/'episodes').glob('primary-tier*-train-*/*/engineering-attempt-0/completed.json'):
  q=read(f)
  if q['steps']>0:completed.append((f,q))
 buckets={};maxbytes=0;starts=[];ends=[];sumwall=0.;audit_ratio=0.;audit_jobs=0;audits_done=0
 for f,q in completed:
  c=byid[q['config_id']];key=c['task']+'/'+q['route'];seconds=wall(q);buckets[key]=max(seconds,buckets.get(key,0.));sumwall+=seconds;starts.append(datetime.fromisoformat(q['started_at']));ends.append(datetime.fromisoformat(q['ended_at']));maxbytes=max(maxbytes,directory_bytes(f.parent))
 for f in (r/'episodes').glob('*/slot-*/engineering-attempt-0/audit-process.json'):
  z=read(f);audit_jobs+=1;q=read(f.parent/'completed.json');duration=float(q.get('terminal_duration_s',0))
  if duration<=0:continue
  elapsed=(datetime.fromisoformat(z.get('ended_at',stamp.isoformat()))-datetime.fromisoformat(z['at'])).total_seconds()
  if z.get('exit_code')==0:audits_done+=1
  audit_ratio=max(audit_ratio,elapsed/duration)
 unknown=max([2700.]+list(buckets.values()));effective=min(2.,max(1.,sumwall/max(1.,(max(ends)-min(starts)).total_seconds()))) if starts else 1.
 tasknames=sorted({c['task'] for c in roster});allbuckets={task+'/'+route:buckets.get(task+'/'+route,unknown) for task in tasknames for route in 'EDA'}
 freeze=read(r/'policy/main-component-freeze.json');component_ok=True
 import hashlib
 for name,digest in freeze['behavioral_components'].items():component_ok &= hashlib.sha256((r/'runtime/mobipi'/name).read_bytes()).hexdigest()==digest
 def estimate(configs,repetitions=1):
  sums=[0.,0.];total=0.;units=0
  for c in configs:
   if not c['static_config_legal']:continue
   feature=read(r/'design/repaired-inputs-v1'/c['config_id']/'features.json');valid={row['route_family']:row['hard_valid'] for row in feature['routes']}
   for route in 'EDA':
    if not valid.get(route,False):continue
    value=allbuckets[c['task']+'/'+route]*repetitions;sums[c['worker_assignment']]+=value;total+=value;units+=repetitions
  return dict(units=units,single_worker_seconds=total,isolated_assignment_seconds=max(sums),measured_concurrency_seconds=total/effective,estimated_seconds=max(max(sums),total/effective)*1.25)
 eval1=estimate([c for c in roster if c['role']=='evaluation' and c['tier']==1],2)
 eval2=estimate([c for c in roster if c['role']=='evaluation'],2)
 extra=estimate([c for c in roster if c['role']!='evaluation' and c['tier']==2])
 # Online choices are frozen later; use max bucket for each of3 online methods,
 # not a cheaper route assumption. Above two copies underestimate if counts of
 # hard-valid routes differ, so also bind a worst-three-route-per-config floor.
 for est,tier in [(eval1,1),(eval2,2)]:
  configs=[c for c in roster if c['role']=='evaluation' and c['tier']<=tier and c['static_config_legal']]
  online=sum(3*max(allbuckets[c['task']+'/'+route] for route in 'EDA') for c in configs)
  primary_estimate=estimate(configs);primary=primary_estimate['single_worker_seconds'];est['units']=primary_estimate['units']+3*len(configs);est['unit_definition']='hard-valid paired routes plus worst3 executable online choices per legal config';est['estimated_seconds']=max(est['estimated_seconds'],(online+primary)/effective*1.25);est['online_worst_route_seconds']=online
 # Conservative outstanding audit forecast uses at least10 CPU seconds per
 # simulated second,14 enabled auditors, and300 seconds for unrun legal units.
 auditors=14;audit_ratio=max(10.,audit_ratio);audit_remaining=0.
 for file in (r/'episodes').glob('*/slot-*/engineering-attempt-0/completed.json'):
  q=read(file)
  if not (Path(q['attempt'])/'sprint-safety-audit.json').exists():audit_remaining+=float(q.get('terminal_duration_s',0))*audit_ratio/auditors
 finalfit=1800.;reserve=rules['evidence_closeout_reserve_seconds'];audit_eval=eval1['units']*300*audit_ratio/auditors
 selected=read(r/'design/tier-decision.json')['selected_tier'] if (r/'design/tier-decision.json').exists() else 1
 finaleval=eval2 if selected==2 else eval1
 audit_eval=finaleval['units']*300*audit_ratio/auditors
 tail1=finaleval['estimated_seconds']+max(audit_remaining,audit_eval)*1.25+finalfit+reserve
 intake=deadline-timedelta(seconds=tail1+2700)
 # This cap may become earlier as resource costs grow, never silently extend.
 old=r/'design/capacity-plan.json'
 if old.exists() and read(old)['first6_task_route_buckets_available']:intake=min(intake,datetime.fromisoformat(read(old)['train_dev_stop_starting_at']))
 first=[c for c in roster if c['role']=='train' and c['tier']==1][:2];measured6=all(c['task']+'/'+route in buckets for c in first for route in 'EDA')
 existing=directory_bytes(r);space=shutil.disk_usage(r).free;perunit=max(maxbytes,2*1024**3);future_units=extra['units']+eval2['units'];peak_extra=future_units*perunit*1.25
 plan=dict(at=stamp.isoformat(),basis='per task-route maximum measured formal train wall; unknown>=2700;1.25factor; actual worker overlap bounded2',measured_formal_train_slots=len(completed),first6_task_route_buckets_available=measured6,
  task_route_max_wall_seconds=allbuckets,measured_effective_workers=effective,initial_concurrency_unmeasured=not bool(starts),single_worker_tier1=estimate([c for c in roster if c['role']!='evaluation' and c['tier']==1]),
  final_tier1_evaluation_online=eval1,final_tier2_evaluation_online=eval2,additional_train_dev=extra,audit_ratio_CPU_seconds_per_sim_second=audit_ratio,audit_workers=auditors,outstanding_audit_estimate_seconds=audit_remaining,
  fit_seconds_reserved=finalfit,evidence_closeout_reserve_seconds=reserve,train_dev_stop_starting_at=intake.isoformat(),hard_deadline=deadline.isoformat(),live_free_bytes=space,existing_artifact_bytes=existing,max_measured_episode_bytes=maxbytes,
  tier2_estimated_additional_peak_bytes=peak_extra,component_freeze_unchanged=bool(component_ok),outcome_scores_used=False)
 def write(p,x):
  p=Path(p);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()));tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
 write(r/'design/capacity-plan.json',plan);history=r/'design/capacity-history.jsonl'
 with history.open('a') as f:f.write(json.dumps(plan)+'\n')
 if a.decide_tier:
  dest=r/'design/tier-decision.json';assert not dest.exists();assert not list((r/'episodes').glob('primary-tier*-evaluation-*/*/engineering-attempt-0/completed.json'))
  assert all((r/'policy'/f'train1-worker{i}/completed.json').exists() for i in [0,1])
  parents={c['parent_group'] for c in roster};additional={c['parent_group'] for c in roster if c['tier']==2 and c['static_config_legal']};coverage=parents<=additional
  cost=extra['estimated_seconds']+eval2['estimated_seconds']+max(audit_remaining, (extra['units']+eval2['units'])*300*audit_ratio/auditors)*1.25+finalfit+reserve
  closedresults=[x for i in [0,1] for x in read(r/'policy'/f'train1-worker{i}/completed.json')['results']]
  first_complete=all(x.get('exit_code',0)==0 and not x.get('status','').startswith('unrun') for x in closedresults)
  checks=dict(firsttier_all_legal_units_closed=first_complete,static_additional_each_parent=coverage,measured_first6=measured6,behavior_unchanged=bool(component_ok),time_remaining_sufficient=(deadline-stamp).total_seconds()>=cost,disk_sufficient=space>=peak_extra,
   train1_binding_closed=(r/'training/tier-1/dataset-binding.json').exists(),two_worker_isolation=True)
  tier=2 if all(checks.values()) else 1;decision=dict(at=stamp.isoformat(),selected_tier=tier,checks=checks,estimated_tier2_remaining_seconds=cost,remaining_seconds=(deadline-stamp).total_seconds(),capacity_receipt=str(old),resource_only=True,evaluation_unopened=True,score_used=False,planned_primary=360 if tier==2 else 180,planned_online=72 if tier==2 else 36)
  write(dest,decision);print(json.dumps(decision),flush=True)
 else:print(json.dumps({k:plan[k] for k in ['at','measured_formal_train_slots','measured_effective_workers','train_dev_stop_starting_at','live_free_bytes']}),flush=True)
if __name__=='__main__':main()
