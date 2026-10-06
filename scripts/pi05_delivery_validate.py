"""Validate frozen provenance and every intended actual evaluation receipt."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
 freeze=json.loads((r/'evaluation/final-freeze.json').read_text());at=datetime.fromisoformat(freeze['at']);missing=[];rows=[];starts={}
 for path,value in freeze['provenance']['file_sha256'].items():assert sha(r/path)==value,path
 for name,model in freeze['models'].items():assert sha(model['checkpoint'])==model['sha256'],name
 binding=json.loads((r/'training/dataset-binding.json').read_text());assert sha(r/'training/train-only.npz')==binding['train_sha256']
 assert binding['train_parent_groups']>=4 and len(binding['supervised_train_routes'])>=2
 for g in freeze['predictions']:
  units=[('paired-v6-final-'+route,route,'paired') for route in 'EDA' if freeze['released_routes']['tasks'][g['task']].get(route,{}).get('released')]
  units+=[('online-v6-'+method,g['selected'][method],'online') for method in freeze['online_methods'] if g['selected'][method]!='X']
  for tag,route,purpose in units:
   paths=list((r/'episodes'/tag).glob(f'slot-{g["slot"]:02d}*/engineering-attempt-*/completed.json'));assert len(paths)<=1
   if not paths:missing.append(tag+'/'+str(g['slot'])+'/outcome');continue
   receipt=paths[0];q=json.loads(receipt.read_text());attempt=Path(q['attempt']);audit=attempt/'sprint-safety-audit.json'
   assert datetime.fromisoformat(q['started_at'])>at and q['purpose']==purpose and q['route']==route and q['parent_group']==g['parent_group']
   assert q['checkpoint_step']==2000 and q['adapter_version']=='v6' and not q['reference_actions_used'] and not q['human_intervention']
   policy=json.loads((receipt.parent/'policy-binding.json').read_text());assert policy['checkpoint']==freeze['provenance']['policy_components']['policy_checkpoint']
   z=np.load(attempt/'formal-native-substeps.npz',allow_pickle=False);start=z['qpos'][0].copy()
   if g['slot'] in starts:assert np.max(abs(start-starts[g['slot']]))<=1e-6
   else:starts[g['slot']]=start
   if not audit.exists():missing.append(tag+'/'+str(g['slot'])+'/safety')
   native=json.loads((attempt/'task-video-manifest.json').read_text());pan=json.loads((attempt/'panoramic-binding.json').read_text())
   assert len(native['decoded_frames_sha256'])==pan['frames']==len(pan['frame_bindings'])==q['steps']
   assert native['binding']['camera']!=pan['camera']
   rows.append(dict(tag=tag,slot=q['slot'],route=route,purpose=purpose,status=q['status'],native_success=q['native_success'],steps=q['steps'],real_queries=q['policy_queries'],receipt=str(receipt)))
 result=dict(at=datetime.now(timezone.utc).isoformat(),status='pending' if missing else 'passed',missing=missing,actual_evaluations=len(rows),
  frozen_bound_files_unchanged=True,frozen_models_and_training_data_unchanged=True,all_first_queries_after_component_freeze=True,
  same_parent_initial_native_states_within1e_6=True,all_native_video_lengths_and_camera_identity_pass=True,rows=rows,
  full_action_replay_claimed=False,human_review='pending',formal_train_ready=False)
 (r/'delivery/evaluation-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
