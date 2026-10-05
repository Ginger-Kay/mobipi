"""Task-independent human recording integrity audit, no simulator stepping."""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np

from human_reference_audit import load_attempt, load_partial_tail, phase_at
from mobiwam.human_postcapture import read_json, atomic_json, stamp, numeric_alignment, signature
from mobiwam.task_video_identity import validate_recording, sha, source_model
from mobiwam.teleop_feedback import touching_fingers


def inspect(attempt, output, expected_scene, expected_route):
    attempt=attempt.resolve();source=attempt.parent.parent
    result,meta,actions,states,integration,events=load_attempt(attempt)
    if meta['scene_id']!=expected_scene or result['route']!=expected_route or meta['route']!=expected_route:
        raise ValueError('Requested scene/route differs from original recording')
    if Path(result['source']).resolve()!=source or meta['source_id']!=source.name:
        raise ValueError('Source path differs')
    if not result.get('ended_at') or not meta.get('ended_at'):
        raise ValueError('Recording is still open')
    if meta['record_type'] in ('primary', 'reference_supplement'):
        cfg=read_json(meta['freeze_receipt'])['config']
        if any(meta.get(k)!=cfg.get(k) for k in ('scene_id','scene_family_id','config_version','environment_seed')):
            raise ValueError('Primary scene/family/seed differs from frozen configuration')
    binding=read_json(source/'target-binding.json')
    v=validate_recording(attempt/'task-video-manifest.json',dict(attempt=str(attempt),route=expected_route,group_id=expected_scene))
    rows=[read_json_line for read_json_line in map(__import__('json').loads,(attempt/'trace.jsonl').read_text().splitlines())]
    with np.load(attempt/'formal-native-substeps.npz') as z:
        native={k:z[k] for k in ('qpos','sim_time','step_index','phases')}
    dimensions=numeric_alignment(actions,states,rows,native,meta['control_dt'])
    m=source_model(str(source/'model.xml'),sha(source/'model.xml'))
    if (m.nq,m.nv)!=(dimensions['nq'],dimensions['nv']):
        raise ValueError('Actual compiled model dimensions differ')
    if len(native['phases'])!=len(native['step_index']) or any(p!=phase_at(result,int(i)) for p,i in zip(native['phases'],native['step_index'])):
        raise ValueError('Native phase differs from original dock boundary')
    tail=load_partial_tail(attempt,result)
    if dimensions['partial_native_steps'] and tail is None:
        raise ValueError('Native partial tail missing original action evidence')
    receipt=read_json(attempt/'formal-native-substeps-receipt.json')
    if receipt.get('forbidden_contact') and tail is None:
        raise ValueError('Forbidden contact lost its failure tail')
    pano=read_json(attempt/'panoramic-binding.json');n=dimensions['steps']
    if sha(attempt/'panoramic.mp4')!=pano['sha256'] or pano['frames']!=n or len(pano['frame_bindings'])!=n:
        raise ValueError('Panoramic bytes/frames differ')
    if pano['camera']==v['binding']['camera']:
        raise ValueError('Two videos use the same evidence camera')
    for row,frame in zip(rows,pano['frame_bindings']):
        for field in ['frame_index','native_model_geometry_sha256','actual_qpos_sha256','actual_qvel_sha256','actual_sim_time']:
            if frame[field]!=row['native_frame_binding'][field]:raise ValueError('Panoramic frame/state binding differs')
        if frame['camera']!=pano['camera']:raise ValueError('Panoramic camera changes within original video')
    with h5py.File(attempt/'demo.hdf5','r') as f:
        obs=f['data/demo_0/obs']
        if any(len(d)!=n+1 for d in obs.values()):raise ValueError('Observation length differs')
        if not np.array_equal(integration,np.load(source/'integration.npy')):
            raise ValueError('Actual recorded initial integration differs from Source')
    decoded={}
    for name in ['original','panoramic']:
        cap=cv2.VideoCapture(str(attempt/(name+'.mp4')));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if count in {0,n//2,n-1}:cv2.imwrite(str(output/f'{name}-{count:04d}.jpg'),frame)
            count+=1
        cap.release()
        if count!=n:raise ValueError('Decoded original video frame count differs')
        decoded[name]=count
    target=binding['fixture_name']
    handle=target+('_door_handle_handle' if binding['task']=='CloseDrawer' else '_door_handle')
    candidates=[m.geom(i).name for i in range(m.ngeom) if m.geom(i).name==handle]
    joint_ids=[m.joint(j['name']).id for j in binding['joints']]
    if candidates:
        body=int(m.geom(handle).bodyid);ancestors=[]
        while body:
            ancestors.append(body);body=int(m.body_parentid[body])
        if not any(int(m.jnt_bodyid[j]) in ancestors for j in joint_ids):
            raise ValueError('Handle is not attached beneath the bound target joint')
    pads=tuple((f'gripper0_right_finger{i}_collision',f'gripper0_right_finger{i}_pad_collision') for i in [1,2])
    if len(candidates)!=1:raise ValueError('Handle geometry ambiguous; do not infer contact counts')
    counts=[sum(touching_fingers(r['after']['contacts'],candidates[0],pads)) for r in rows]
    first=next((i for i,c in enumerate(counts) if c),None)
    success=[bool(r['after']['success']) for r in rows]
    if success[-1]!=result['checker_success']:raise ValueError('Recorded terminal checker differs')
    base=np.asarray([rows[0]['before']['base_pos']]+[r['after']['base_pos'] for r in rows])
    np.save(output/'initial-integration.npy',integration)
    data=dict(at=stamp(),attempt=str(attempt),source=str(source),scene_id=meta['scene_id'],scene_family_id=meta['scene_family_id'],
              config_version=meta['config_version'],record_type=meta['record_type'],route=meta['route'],
              initial_integration=str(output/'initial-integration.npy'),input_signature=signature(attempt),
              identity_verified=True,integrity_pass=True,route_input_semantics_pass=True,
              dimensions=dimensions,decoded_frames=decoded,checker_success=result['checker_success'],
              success_streak_10=len(success)>=10 and all(success[-10:]),stop_reason=result['reason'],
              contact_handle=candidates[0],first_handle_contact=first,
              contact_counts_after_first=dict(Counter(counts[first:])) if first is not None else {},
              base_world_path_m=float(np.linalg.norm(np.diff(base[:,:2],axis=0),axis=1).sum()),
              minimum_joint_margin=read_json(attempt/'joint-margin-monitor.json')['minimum'],
              videos={name:str(attempt/(name+'.mp4')) for name in decoded},
              continuous_clearance='pending',dynamic_replay='pending',human_review='pending',
              autonomous_executor_label=False,formal_train_ready=False)
    atomic_json(output/'result.json',data)
    return data


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',required=True);p.add_argument('--route',choices=['E','D','A'],required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        result=inspect(a.attempt,a.output,a.scene,a.route)
        print(result['scene_id'],result['route'],result['dimensions'],'integrity passed',flush=True)
    except Exception as e:
        atomic_json(a.output/'error.json',dict(at=stamp(),type=type(e).__name__,detail=str(e),automatic_retry=False))
        raise


if __name__=='__main__':main()
