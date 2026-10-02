"""One pre-registered DR-v0.4 E/D/A group, pending strict post-route audit.

The preflight was planned under an older immutable code commit. The two
execution-only files are pinned to this collector's separate committed tree.
This script has no outcome-based search, restart, substitution or retries.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import mujoco
from teleop_reference import Reference, stamp, write_json
from reference_executor import compile_path, run_route
from mobiwam.reference_dispatch import rejection_reason
from mobiwam.reference_plan_reuse import verify_plan_seal
from mobiwam.reference_transfer import compile_transferred_path

PREPLAN_DEPENDENCIES=(
 'scripts/reference_geometry.py','scripts/reference_geometry_v16.py',
 'scripts/reference_planning.py','scripts/reference_prefix_preview.py',
 'scripts/teleop_reference.py','scripts/reference_control_diagnostics.py',
 'src/mobiwam/reference_collision.py','src/mobiwam/reference_ik.py',
 'src/mobiwam/reference_dispatch.py','src/mobiwam/scene004.py',
 'src/mobiwam/reference_plan_reuse.py','scripts/reference_stow.py',
 'src/mobiwam/reference_prefix_safety.py','src/mobiwam/reference_handoff.py',
 'src/mobiwam/reference_transfer.py','scripts/reference_transfer_plan.py',
)
NEW_DEPENDENCIES=('scripts/reference_executor.py','scripts/dr_v04_formal_collect.py',
                  'src/mobiwam/reference_formal_substep.py')

def sha(path):
 h=sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()

def checked_dependency(root, commit, file):
 historical=subprocess.check_output(['git','-C',str(root),'show',commit+':'+file])
 if historical!=(root/file).read_bytes():raise ValueError('code dependency changed: '+file)

def verify_dispatch(freeze, group_id, code_root, prior):
 if freeze['status']!='DR-v0.4_complete_preoutcome_freeze':raise ValueError('formal freeze not complete')
 if freeze['formal_route_outcomes']!=0:raise ValueError('preoutcome freeze has outcome contamination')
 execution_sha=subprocess.check_output(['git','-C',str(code_root),'rev-parse','HEAD'],text=True).strip()
 if execution_sha!=freeze['formal_execution_code_commit']:raise ValueError('formal execution commit differs')
 if subprocess.check_output(['git','-C',str(code_root),'status','--porcelain']).strip():
  raise ValueError('formal execution worktree is dirty')
 rows=[x for x in freeze['primary'] if x['group_id']==group_id]
 if len(rows)!=1 or rows[0]['split'] not in ('train','validation'):
  raise ValueError('sealed test / unknown group cannot execute')
 row=rows[0]
 if prior!=Path(row['plan_run']).resolve():raise ValueError('non-frozen preflight directory')
 old_commit=freeze['planning_code_commit']
 if old_commit!=json.loads((prior.parent/'manifest.json').read_text())['code_commit']:
  raise ValueError('planning provenance changed')
 for file in PREPLAN_DEPENDENCIES:checked_dependency(code_root,old_commit,file)
 for file in NEW_DEPENDENCIES:checked_dependency(code_root,execution_sha,file)
 for name,key in (('model.xml','model_sha256'),('integration.npy','integration_sha256')):
  if sha(Path(row['source'])/name)!=row[key]:raise ValueError('frozen Source mutated')
 if sha(prior/'planning/candidate-features.json')!=row['candidate_features_sha256']:
  raise ValueError('planned geometry features mutated')
 if sha(prior/'sealed-inputs.json')!=row['plan_seal_sha256']:
  raise ValueError('sealed planning inputs mutated')
 for name in ('waypoints.json','dock-plan.json','transfer-receipt.json'):
  if sha(prior/name)!=row['planning_aux_sha256'][name]:raise ValueError('preplanned path mutated: '+name)
 if mujoco.__version__!='3.2.6':raise ValueError('physics version differs')
 return row,old_commit,execution_sha

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--freeze',type=Path,required=True)
 p.add_argument('--group-id',required=True)
 p.add_argument('--output',type=Path,required=True)
 args=p.parse_args(); prior_freeze=args.freeze.resolve();out=args.output.resolve()
 root=Path(__file__).resolve().parent.parent
 freeze=json.loads(prior_freeze.read_text())
 rows=[x for x in freeze.get('primary',[]) if x['group_id']==args.group_id]
 if len(rows)!=1:raise ValueError('unknown frozen group')
 prior=Path(rows[0]['plan_run']).resolve()
 row,old_commit,new_commit=verify_dispatch(freeze,args.group_id,root,prior)
 preflight=json.loads((prior/'planning/candidate-features.json').read_text())
 spec=json.loads((prior/'executor-spec.json').read_text())
 source=Path(preflight['source']).resolve()
 if source.parent!=prior or sha(source/'model.xml')!=row['model_sha256']:
  raise ValueError('planned Source copy differs')
 seal=verify_plan_seal(prior,source.name)
 if seal['planning_code_commit']!=old_commit or seal.get('restored_source_integration')!=preflight.get('restored_source_integration'):
  raise ValueError('planned Source seal differs')
 if not preflight.get('source_integration_unchanged') or sha(source/'integration.npy')!=spec['source_sha256']:
  raise ValueError('Source integration changed')
 order=row['route_order']
 if len(order)!=3 or set(order)!={'E','D','A'}:raise ValueError('invalid frozen route order')
 rejected={route:rejection_reason(preflight,route,True) for route in order}
 if any(rejected.values()):raise ValueError('predeclared hard-invalid primary: '+repr(rejected))
 # No directory is created until all checks above have passed.
 out.mkdir(parents=True,exist_ok=False)
 dest=out/source.name;dest.mkdir()
 for name in ('model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json'):
  shutil.copy2(source/name,dest/name)
  if sha(dest/name)!=sha(source/name):raise ValueError('Source copy mismatch')
 shutil.copy2(prior/'env_config.json',out/'env_config.json')
 cfg=json.loads((out/'env_config.json').read_text())
 if cfg['env_name']!=row['task']:raise ValueError('frozen task mismatch')
 ref=Reference(argparse.Namespace(output=str(out),task=cfg['env_name'],layout=0,style=0,seed=7,
  self_test=True,source=str(dest),replay_attempt=None,resume_attempt=None,width=1920,height=1080))
 ref.label='DR-v0.4_prospective_formal_candidate_pending_full_route_audit'
 try:
  ref.restore()
  if spec.get('transfer_mode')=='moving_target_handle_frame':
   points,transfer=compile_transferred_path(ref,Path(spec['reference']))
   planned=json.loads((prior/'transfer-receipt.json').read_text())
   transfer.pop('new_source',None);planned.pop('new_source',None)
   if transfer!=planned:raise ValueError('transferred reference geometry differs')
  elif spec.get('transfer_mode') is None:
   points=compile_path(ref,Path(spec['reference']))
  else:raise ValueError('unknown transfer mode')
  if json.loads(json.dumps(points,default=lambda x:x.tolist()))!=json.loads((prior/'waypoints.json').read_text()):
   raise ValueError('preplanned points changed')
  ref.dock_plan=json.loads((prior/'dock-plan.json').read_text())
  if ref.dock_plan['selected']['id']!=preflight['selected_dock_id']:
   raise ValueError('preplanned D dock changed')
  restored=preflight.get('restored_source_integration')
  expected=np.load(prior/restored if restored else source/'integration.npy',allow_pickle=False)
  if expected.shape!=ref.integration().shape or not np.isfinite(expected).all():
   raise ValueError('invalid planned integration state')
  error=float(np.max(abs(ref.integration()-expected)))
  if error>1e-10:raise ValueError('Source restore differs from planner state')
  receipt=dict(created_at=stamp(),preoutcome_freeze=str(prior_freeze),preoutcome_freeze_sha256=sha(prior_freeze),
   group_id=args.group_id,split=row['split'],route_order=order,planning_code_commit=old_commit,
   formal_execution_code_commit=new_commit,reference_geometry_identical=True,source_restore_max_error=error,
   Source_model_sha256=sha(source/'model.xml'),Source_integration_sha256=sha(source/'integration.npy'),
   outcome_scope='prospective_formal_candidate_pending_machine_video_replay_audit',formal_train_ready=False)
  write_json(out/'formal-preexecution-receipt.json',receipt)
  write_json(out/'recording-provenance.json',dict(data_kind='DR-v0.4_prospective_formal_candidate_pending_audit',
   split=row['split'],group_id=args.group_id,model_code_commit=new_commit,planner_commit=old_commit,
   lineage_reference_sha256=row['derived_reference_trace_sha256'],preoutcome_freeze_sha256=sha(prior_freeze),
   formal_train_ready=False))
  results=[]
  for route in order:
   path=run_route(ref,route,points,2400,execution_scope='DR-v0.4_formal_candidate_pending_audit')
   results.append(dict(route=route,path=path))
   # Append immediately: stop or crash after any route leaves the complete attempt and no repeat.
   write_json(out/f'route-{route}-dispatched.json',dict(route=route,path=path,ended_at=stamp(),
    outcome_status='pending_independent_replay_and_full_video_substep_audit'))
  write_json(out/'completed.json',dict(ended_at=stamp(),attempts=results,
   route_outcomes=len(results),audit='pending',formal_train_ready=False))
 finally:
  if ref.observation_renderer is not None:ref.observation_renderer.close()
  if ref.renderer is not None:ref.renderer.close()
  ref.env.close()

if __name__=='__main__':main()
