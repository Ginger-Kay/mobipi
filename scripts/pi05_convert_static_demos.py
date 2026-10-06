"""Convert eligible E expert commands to base-anchored nominal EEF targets.

Only policy-fit parents are opened. Actions are commanded OSC goals derived
from current kinematics, never the next achieved displacement. Chunk targets
are transformed to the base at the query timestep by the OpenPI loader.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image

ROOT=Path('/share/personal/chensiyu/haokaijiang/MobiWAM')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--attempt',type=int,default=0);a=ap.parse_args()
    receipt=json.loads((a.run/'preflight/native-interface-retry-1/nominal-target-roundtrip.json').read_text())
    assert receipt['passed']
    candidates=json.loads((a.run/'data/policy-fit-candidates.json').read_text())
    splits=json.loads((a.run/'data/lineage-split.json').read_text())['parent_groups']
    fit={x['parent_group'] for x in splits if x['role']=='train'}
    dataset=ROOT/'data/obc-pi05-v1'/('static-command-v1' if a.attempt==0 else f'static-command-v1-retry-{a.attempt}');dataset.mkdir(exist_ok=False)
    index=[];rejected=[];all_states=[];all_targets=[]
    rule=dict(created_at=datetime.now(timezone.utc).isoformat(),rule='whole safe-success E records only; base command zero, mode negative, torso command zero; no partial segments; complete 10-step windows only',
        padding='state valid22 then10 zeros; action valid8 then24 zeros; padding normalized zeros; official model loss includes padding dimensions',
        camera_mapping={'base_0_rgb':'robot0_agentview_left','left_wrist_0_rgb':'robot0_eye_in_hand','right_wrist_0_rgb':'robot0_agentview_right'},
        preprocessing='recorded top-first native RGB uint8; PIL bilinear 256 to224; then model float [-1,1]',
        state_layout='current base EEF position3, rotation-vector3, gripper mean opening/.04, arm cos7,sin7,torso joint position1,zero padding10',
        target_layout='nominal EEF position3/rotation-vector3 in query-time mobilebase0_base; gripper original -1 open/+1 close; torso original zero; zero padding24',
        normalization='policy-fit-only per-channel quantile01/99; constant channels mapped to zero; no final/dev observations',
        future_base='target world command at t+k is expressed in base at query t; future base never enters input')
    (dataset/'conversion-rule.json').write_text(json.dumps(rule,indent=2)+'\n')
    for row in candidates:
        assert row['parent_source'] in fit and row['route']=='E' and row['success']
        path=Path(row['observations_path']);ep=dataset/row['record_id'];ep.mkdir()
        with h5py.File(path) as f:
            g=f['data/demo_0'];actions=g['actions'][:];T=len(actions)
            if np.max(abs(actions[:,7:11]),initial=0)>1e-12 or np.any(actions[:,11]>=0):
                rejected.append(dict(record=row['record_id'],reason='not static compatible base/torso/mode'));continue
            # Native RoboCasa performs this path repair during XML restoration.
            # Preserve the historical original XML and change only the transient
            # local asset URI, with geometry checked against recorded sensors.
            xml=g.attrs['model_file'].replace('/share/personal/haokaijiang/MobiWAM',str(ROOT))
            model=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(model)
            base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'mobilebase0_base')
            origin=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'robot0_right_center')
            eef=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'gripper0_right_grip_site')
            assert min(base,origin,eef)>=0
            states=g['states'][:T];obs=g['obs'];state=np.zeros((T,32),np.float32)
            base_p=np.zeros((T,3));base_R=np.zeros((T,3,3));world_p=np.zeros((T,3));world_R=np.zeros((T,3,3))
            local_targets=np.zeros((T,32),np.float32);error_obs=0.
            for t in range(T):
                d.time=states[t,0];d.qpos[:]=states[t,1:1+model.nq];d.qvel[:]=states[t,1+model.nq:1+model.nq+model.nv];mujoco.mj_forward(model,d)
                bp=d.xpos[base].copy();br=d.xmat[base].reshape(3,3).copy();op=d.site_xpos[origin];oR=d.site_xmat[origin].reshape(3,3);p=d.site_xpos[eef];R=d.site_xmat[eef].reshape(3,3)
                error_obs=max(error_obs,float(np.max(abs(p-obs['robot0_eef_pos'][t]))))
                # The native sensor cache precedes the final mj_step2 update.
                # Recomputed FK is a diagnostic, not the recorded controller
                # input. Use the actual simultaneous observed pose for labels.
                bp=obs['robot0_base_pos'][t];br=Rotation.from_quat(obs['robot0_base_quat'][t]).as_matrix()
                p=obs['robot0_eef_pos'][t];R=Rotation.from_quat(obs['robot0_eef_quat_site'][t]).as_matrix()
                observed_local=br.T@(p-bp)
                assert np.max(abs(observed_local-obs['robot0_base_to_eef_pos'][t]))<=1e-6
                # Static selected demos have fixed torso and parallel controller
                # origin axes, independently checked against the actual model.
                assert np.max(abs(oR-br))<=1e-5
                world_p[t]=p+br@(np.clip(actions[t,:3],-1,1)*.05)
                world_R[t]=br@Rotation.from_rotvec(np.clip(actions[t,3:6],-1,1)*.5).as_matrix()@br.T@R
                base_p[t]=bp;base_R[t]=br
                state[t,:3]=br.T@(p-bp);state[t,3:6]=Rotation.from_matrix(br.T@R).as_rotvec()
                state[t,6]=np.mean(abs(obs['robot0_gripper_qpos'][t]))/.04
                state[t,7:14]=obs['robot0_joint_pos_cos'][t];state[t,14:21]=obs['robot0_joint_pos_sin'][t];state[t,21]=d.qpos[3]
                local_targets[t,:3]=br.T@(world_p[t]-bp);local_targets[t,3:6]=Rotation.from_matrix(br.T@world_R[t]).as_rotvec()
                local_targets[t,6:8]=actions[t,[6,10]]
            np.savez(ep/'commands.npz',state=state,base_p=base_p,base_R=base_R,target_world_p=world_p,target_world_R=world_R,grip_torso=actions[:,[6,10]])
            for slot,key in rule['camera_mapping'].items():
                cache=np.lib.format.open_memmap(ep/(slot+'.npy'),mode='w+',dtype=np.uint8,shape=(T,224,224,3))
                for t in range(T):cache[t]=np.asarray(Image.fromarray(obs[key+'_image'][t]).resize((224,224),Image.Resampling.BILINEAR))
                cache.flush();del cache
            rec=dict(record_id=row['record_id'],task=row['task'],parent_group=row['parent_source'],config_id=row['config_id'],family_id=row['task']+'-layout1-style0',
                original_hdf5=str(path),derived=str(ep),frames=T,windows=max(0,T-9),sensor_fk_max_error_m=error_obs,
                source=str(Path(row['attempt']).parents[1]),source_checksum_receipt=row.get('existing_receipt'),
                sensor_fk_note='mj_step2 cached sites differ from FK recomputed after final integration; labels use recorded sensor poses; no tolerance relaxation or next-state targets',
                prompt='Close the drawer.' if row['task']=='CloseDrawer' else 'Close the microwave door.')
            index.append(rec);all_states.append(state);all_targets.append(local_targets)
            print(json.dumps(dict(record=rec['record_id'],task=rec['task'],frames=T,windows=rec['windows'],fk_error_m=error_obs)),flush=True)
    assert index
    norm={}
    for name,vals,width in [('state',all_states,22),('actions',all_targets,8)]:
        X=np.concatenate(vals);low=np.zeros(32);high=np.zeros(32);low[:width]=np.quantile(X[:,:width],.01,axis=0);high[:width]=np.quantile(X[:,:width],.99,axis=0)
        norm[name]=dict(q01=low.tolist(),q99=high.tolist(),valid_width=width,constant=((high-low)<1e-6).tolist())
    (dataset/'norm-stats.json').write_text(json.dumps(norm,indent=2)+'\n')
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),dataset=str(dataset),episodes=index,rejected=rejected,
        valid_windows=sum(x['windows'] for x in index),independent_episodes=len(index),parent_groups=len({x['parent_group'] for x in index}),family_count=len({x['family_id'] for x in index}),
        task_frame_counts={t:sum(x['frames'] for x in index if x['task']==t) for t in ('CloseDrawer','CloseSingleDoor')},
        split_manifest=str(a.run/'data/lineage-split.json'),horizon=10,rule=rule,not_obc_outcome_labels=True)
    (dataset/'dataset.json').write_text(json.dumps(manifest,indent=2)+'\n');(a.run/'data/dataset-binding.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('valid_windows','independent_episodes','parent_groups','task_frame_counts')}),flush=True)


if __name__=='__main__':main()
