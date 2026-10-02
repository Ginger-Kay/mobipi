"""Audit one prospective route without repairing, relabeling or replaying it.

A route is never promoted merely because the recorder reports checker success.
This checks the actual native substep trajectory with the predeclared 0.5 mm
swept rule and decodes every original video frame.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import h5py
import imageio.v2 as imageio
import mujoco
from mobiwam.reference_collision import SweptGeometry


def sha(path):
 h=sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def bind_modeled_failure(sweep, native_forbidden_contact):
    """Use the existing task adapter's irreversible_failure=collision scope.

    The two rigid-body closing tasks expose no separate irreversible-damage
    channel. A witnessed forbidden native contact is positive; absence is
    negative only when the entire actual path passes the strict swept gate.
    Unresolved or low-clearance paths remain missing and cannot enter Gate.
    """
    if native_forbidden_contact:
        return True, 'native_forbidden_contact_positive'
    if sweep.get('valid') is True:
        return False, 'full_native_and_swept_audit_negative'
    return None, 'missing_due_to_actual_swept_safety_failure'


def audit(freeze,group_id,route,attempt):
 from mobiwam.task_video_identity import validate_recording
 rows=[x for x in freeze['primary'] if x['group_id']==group_id]
 if freeze['status']!='DR-v0.4_complete_preoutcome_freeze' or len(rows)!=1:
  raise ValueError('unfrozen group')
 row=rows[0]
 if row['split'] not in ('train','validation') or route not in row['route_order']:
  raise ValueError('sealed test or nonregistered route')
 validate_recording(attempt/'task-video-manifest.json',dict(group_id=group_id,route=route,attempt_id=attempt.name))
 expected_parent=attempt.parent.parent
 if attempt.parent.name!=route or expected_parent.name!=Path(row['source']).name:
  raise ValueError('route/source identity differs')
 result=load(attempt/'result.json')
 if result['status']!='recorded' or result['route']!=route or result['steps']<1:
  raise ValueError('incomplete original route')
 if not (attempt/'formal-native-substeps.npz').is_file():
  raise ValueError('native control substeps missing, route cannot be audited')
 before=load(attempt/'restore-receipt.json')
 if before['max_integration_abs_error']>1e-10:
  raise ValueError('route did not restore sealed Source')
 trace=[json.loads(line) for line in (attempt/'trace.jsonl').read_text().splitlines()]
 feedback=[json.loads(line) for line in (attempt/'feedback.jsonl').read_text().splitlines()]
 with h5py.File(attempt/'demo.hdf5') as f:
  group=f['data/demo_0'];states=group['states'];actions=group['actions']
  n=result['steps']
  if states.shape[0]!=n+1 or actions.shape[0]!=n or n!=len(trace) or n!=len(feedback):
   raise ValueError('trace/action/state misalignment')
  if any(not np.isfinite(states[i:i+100]).all() or not np.isfinite(actions[i:i+100]).all()
         for i in range(0,n,100)):
   raise ValueError('nonfinite actual state or action')
  obs=group['obs']
  for key in ('robot0_agentview_left_image','robot0_agentview_right_image','robot0_eye_in_hand_image'):
   if obs[key].shape!=(n+1,256,256,3):raise ValueError('policy video frames not aligned')
  original_state=group['initial_integration'][:]
  if not np.isfinite(original_state).all():raise ValueError('invalid source integration')
 with np.load(attempt/'formal-native-substeps.npz',allow_pickle=False) as f:
  qpos=f['qpos'];times=f['sim_time'];phases=f['phases'].tolist();indices=f['step_index']
  if qpos.ndim!=2 or len(qpos)!=len(phases)+1 or len(times)!=len(qpos) or len(indices)!=len(phases):
   raise ValueError('native substep evidence misaligned')
  if not np.isfinite(qpos).all() or not np.isfinite(times).all() or (np.diff(times)<-1e-12).any():
   raise ValueError('invalid native physics timeline')
  if phases and min(indices)<0:raise ValueError('negative native substep index')
  if phases and max(indices)>=n:
   # A partial stopped action can have step index n, but no completed trace row.
   if max(indices)!=n or result['reason'] not in ('native_forbidden_contact_stop','joint_margin_stop'):
    raise ValueError('native substep index beyond original recording')
  native_states=np.asarray(qpos).copy()
  native_phases=list(phases)
 video_frames=0
 with imageio.get_reader(attempt/'original.mp4','ffmpeg') as reader:
  for frame in reader:
   if frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8:
    raise ValueError('original video failed full RGB decode')
   video_frames+=1
 if video_frames!=n:raise ValueError('video and recorded actions misaligned')
 end=trace[-1]['after']
 if bool(end['success'])!=bool(result['checker_success']):
  raise ValueError('native task checker and recorder disagree')
 if not np.isfinite(float(end['target']['door'])):
  raise ValueError('invalid native task progress')
 progress=float(np.clip(1.0-float(end['target']['door']),0.0,1.0))
 bases=np.array([trace[0]['before']['base_pos']]+[t['after']['base_pos'] for t in trace],dtype=float)
 if not np.isfinite(bases).all() or bases.shape[1]<2:raise ValueError('invalid base trajectory')
 base_path=float(np.linalg.norm(np.diff(bases[:,:2],axis=0),axis=1).sum())
 elapsed=float(end['sim_time']-trace[0]['before']['sim_time'])
 if elapsed<0 or not np.isfinite(elapsed):raise ValueError('invalid simulator elapsed time')
 model=mujoco.MjModel.from_xml_path(str(expected_parent/'model.xml'))
 if native_states.shape[1]!=model.nq:raise ValueError('native qpos/model dimension differs')
 geom=SweptGeometry(model,target_prefix=row['fixture_name'],margin=.0005)
 sweep=geom.path(native_states,native_phases)
 controller=load(attempt/'executor-result.json')
 substeps=load(attempt/'formal-native-substeps-receipt.json')
 margin=load(attempt/'joint-margin-monitor.json')
 contact_stop=(attempt/'formal-substep-stop.json').is_file() or (attempt/'contact-stop.json').is_file()
 actual_safety_pass=(sweep['valid'] and not contact_stop and
  margin['minimum'] is not None and margin['minimum']['margin_rad']>.015)
 replay_results=list(attempt.glob('replay-*/result.json'))
 if len(replay_results)!=1:raise ValueError('exactly one replay receipt required')
 replay=load(replay_results[0])
 reproducible=(replay['steps']==n and replay['checker_success']==result['checker_success']
  and replay['first_state_error_gt_1e-5'] is None and replay['max_state_abs_error'] is not None
  and replay['max_state_abs_error']<=1e-5)
 if substeps['substeps']!=len(native_phases):raise ValueError('substep count differs')
 if controller['execution_scope']!='DR-v0.4_formal_candidate_pending_audit':
  raise ValueError('development execution cannot become formal')
 failure,failure_binding=bind_modeled_failure(sweep,contact_stop)
 ready=bool(actual_safety_pass and reproducible and failure is not None and not substeps['forbidden_contact'])
 return dict(group_id=group_id,split=row['split'],route=route,attempt=str(attempt),
  status='machine_audit_pass_pending_research_review' if ready else 'route_ineligible_or_machine_audit_failed',
  checker_success=bool(result['checker_success']),task_progress_after=progress,
  actual_base_path_m=base_path,completion_time_s=elapsed,original_video_frames=video_frames,
  recorded_actions=n,native_substeps=len(native_phases),
  actual_swept_geometry=sweep,realized_contact_stop=contact_stop,
  arm_joint_margin=margin,replay_result_sha256=sha(replay_results[0]),replay_reproducible=reproducible,
  raw_executor_reason=result['reason'],
  irreversible_or_collision=failure,label_failure_status=failure_binding,
  failure_label_source='existing CloseDrawer/CloseSingleDoor adapter: irreversible_failure=trace.collision; actual native forbidden contact and conservative 0.5mm swept gate',
  machine_eligible_for_gate=ready,
  source_input_sha256=row['source_input_sha256'],
  original_video_sha256=sha(attempt/'original.mp4'),
  action_hdf5_sha256=sha(attempt/'demo.hdf5'),
  native_substeps_sha256=sha(attempt/'formal-native-substeps.npz'))


def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True)
 p.add_argument('--group-id',required=True);p.add_argument('--route',choices=('E','D','A'),required=True)
 p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 out=audit(load(a.freeze),a.group_id,a.route,a.attempt.resolve())
 with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':out['status'],'group_id':a.group_id,'route':a.route,
                   'machine_eligible_for_gate':out['machine_eligible_for_gate']}))

if __name__=='__main__':main()
