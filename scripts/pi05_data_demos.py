"""Convert qualified human/reference actual commands into query-relative EEF.

Native OSC goals are reconstructed, including desired-mode goal accumulation.
The moving controller origin is retained in output targets; future origin
poses are target-only and never model inputs. No physics replay is performed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

CAMERAS={'base_0_rgb':'robot0_agentview_left','left_wrist_0_rgb':'robot0_eye_in_hand','right_wrist_0_rgb':'robot0_agentview_right'}
def write(p,x): Path(p).write_text(json.dumps(x,indent=2)+'\n')
def now(): return datetime.now(timezone.utc).isoformat()

def reconstruct(actions, bp, br, ep, er):
    """Actual native achieved/desired goals, evaluated at interval-end origin.

    Both instantaneous and end-origin targets are kept. EEF achieved motion is
    never used as an action label. Origin motion is the synchronized recorded
    response to the original base/torso command, not a second base command.
    """
    n=len(actions); gp=br[0].T@(ep[0]-bp[0]); gr=br[0].T@er[0]
    world_p=np.empty((n,3));world_r=np.empty((n,3,3));instant_p=np.empty((n,3));instant_r=np.empty((n,3,3))
    local_p=np.empty((n,3));local_r=np.empty((n,3,3))
    for t,act in enumerate(actions):
        if act[11]<=0:
            gp=br[t].T@(ep[t]-bp[t]);gr=br[t].T@er[t]
        gp=gp+np.clip(act[:3],-1,1)*.05
        gr=Rotation.from_rotvec(np.clip(act[3:6],-1,1)*.5).as_matrix()@gr
        local_p[t]=gp;local_r[t]=gr
        instant_p[t]=bp[t]+br[t]@gp;instant_r[t]=br[t]@gr
        world_p[t]=bp[t+1]+br[t+1]@gp;world_r[t]=br[t+1]@gr
    return world_p,world_r,instant_p,instant_r,local_p,local_r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    root=Path('/share/personal/chensiyu/haokaijiang/MobiWAM')
    dataset=root/'data/derived/obc-pi05-data-v1'/a.run.name/'unified-query-relative';dataset.mkdir(exist_ok=False)
    inventory=[json.loads(x) for x in (a.run/'inventory/inventory.jsonl').read_text().splitlines()]
    candidates=sorted((x for x in inventory if 'policy-demo-candidate' in x.get('new_use','')),key=lambda x:x['record_id'])
    split=json.loads((a.run/'inventory/source-split.json').read_text())
    reserved=set(split['reserved_development_evaluation_ancestors'])
    cutoff={'at':now(),'finalized_candidate_ids':[x['record_id'] for x in candidates],'late_arrivals':'next-fit only','policy_fit_count_limit':None,'reserved_ancestors':sorted(reserved)}
    write(a.run/'inventory/fit-data-cutoff.json',cutoff)
    records=[];admission=[];norm_states=[];norm_actions=[];seen_hash={}
    for row in candidates:
        ident=row['record_id'];path=Path(row['observations_path']);parent=row['parent_group']
        if parent in reserved:raise AssertionError('held-out ancestry in policy pool')
        try:
            assert path.is_file()
            with h5py.File(path,'r') as f:
                g=f['data/demo_0'];obs=g['obs'];actions=g['actions'][:];n=len(actions)
                assert actions.shape==(n,12) and n>=10 and np.isfinite(actions).all()
                assert len(g['states'])>=n and all(len(obs[k+'_image'])>=n for k in CAMERAS.values())
                # Final obs[T] is required for commanded goal in moving origin.
                assert all(len(obs[k])>=n+1 for k in ['robot0_base_pos','robot0_base_quat','robot0_eef_pos','robot0_eef_quat_site'])
                stamps=obs['sim_time'][:n+1].reshape(-1);assert np.max(abs(np.diff(stamps)-.05))<1e-6
                bp=obs['robot0_base_pos'][:n+1];br=Rotation.from_quat(obs['robot0_base_quat'][:n+1]).as_matrix()
                ep=obs['robot0_eef_pos'][:n+1];er=Rotation.from_quat(obs['robot0_eef_quat_site'][:n+1]).as_matrix()
                assert np.max(abs(np.einsum('tji,tj->ti',br[:n],ep[:n]-bp[:n])-obs['robot0_base_to_eef_pos'][:n]))<1e-6
                wp,wr,ip,ir,lp,lr=reconstruct(actions,bp,br,ep,er)
                state=np.zeros((n,32),np.float32);state[:,:3]=np.einsum('tji,tj->ti',br[:n],ep[:n]-bp[:n])
                state[:,3:6]=Rotation.from_matrix(np.einsum('tji,tjk->tik',br[:n],er[:n])).as_rotvec()
                state[:,6]=np.mean(abs(obs['robot0_gripper_qpos'][:n]),axis=1)/.04
                state[:,7:14]=obs['robot0_joint_pos_cos'][:n];state[:,14:21]=obs['robot0_joint_pos_sin'][:n]
                # Recorded native qpos layout was independently bound in HARNESS.
                state[:,21]=g['states'][:n,4]
                digest=hashlib.sha256(actions.tobytes()+g['states'][:].tobytes()).hexdigest()
                if digest in seen_hash:
                    admission.append(dict(record_id=ident,status='duplicate',duplicate_of=seen_hash[digest]));continue
                seen_hash[digest]=ident
                target=np.zeros((n,32),np.float32)
                target[:,:3]=np.einsum('tji,tj->ti',br[:n],wp-ep[:n])
                target[:,3:6]=Rotation.from_matrix(np.einsum('tji,tjk,tkl->til',br[:n],wr,np.swapaxes(er[:n],1,2))@br[:n]).as_rotvec()
                target[:,6:8]=actions[:,[6,10]]
                # Mathematical inverse: query-relative command -> original goal.
                restored_p=ep[:n]+np.einsum('tij,tj->ti',br[:n],target[:,:3])
                restored_r=np.einsum('tij,tjk,tkl,tlm->tim',br[:n],Rotation.from_rotvec(target[:,3:6]).as_matrix(),np.swapaxes(br[:n],1,2),er[:n])
                poserr=float(np.max(abs(restored_p-wp)));roterr=float(np.max(abs(restored_r-wr)))
                assert poserr<1e-6 and roterr<1e-6
                epdir=dataset/ident;epdir.mkdir()
                np.savez(epdir/'commands.npz',state=state,base_p=bp[:n],base_R=br[:n],target_world_p=wp,target_world_R=wr,
                    grip_torso=actions[:,[6,10]],original_native_actions=actions,instantaneous_target_world_p=ip,instantaneous_target_world_R=ir,
                    nominal_controller_goal_pos=lp,nominal_controller_goal_rot=lr,synchronized_origin_p=bp,synchronized_origin_R=br)
                for slot,key in CAMERAS.items():
                    mm=np.lib.format.open_memmap(epdir/(slot+'.npy'),mode='w+',dtype=np.uint8,shape=(n,224,224,3))
                    for t in range(n):mm[t]=np.asarray(Image.fromarray(obs[key+'_image'][t]).resize((224,224),Image.Resampling.BILINEAR))
                    mm.flush();del mm
                source=Path(row.get('source') or row['attempt']).resolve()
                if source.name.startswith('attempt-'):source=source.parents[1]
                sourceinfo=json.loads((source/'source.json').read_text()) if (source/'source.json').exists() else {}
                rec=dict(record_id=ident,original_record_id=ident,parent_group=parent,config_id=row.get('config_id'),family_id=row.get('family_id',row['task']+'-layout1-style0'),
                    task=row['task'],route=row['route'],demonstrator_type='human' if row['view']=='B_human' else 'reference_controller',controller_version=row.get('controller_version',row.get('controller')),
                    template_ancestry=sourceinfo.get('lineage',sourceinfo.get('source_lineage','shared historical template; exact receipt in raw Source')),
                    raw_hdf5=str(path),raw_quality_receipt=row.get('qualification_receipt',row.get('existing_receipt')),derived=str(epdir),frames=n,windows=n-9,
                    prompt='Close the drawer.' if row['task']=='CloseDrawer' else 'Close the microwave door.',command_state_fingerprint=digest,
                    base_command_nonzero_frames=int(np.sum(np.max(abs(actions[:,7:10]),axis=1)>1e-12)),desired_mode_frames=int(np.sum(actions[:,11]>0)),
                    torso_command_nonzero_frames=int(np.sum(abs(actions[:,10])>1e-12)),idle_command_frames=int(np.sum(np.max(abs(actions[:,:6]),axis=1)<1e-12)),
                    source_use='policy-fit' if parent in {x['parent_group'] for x in split['parent_groups'] if x['role']=='train'} else 'policy-fit-only',
                    native_goal_roundtrip_max_position_m=poserr,native_goal_roundtrip_max_rotation=roterr,
                    moving_target_definition='native local OSC commanded goal composed with synchronized interval-end controller origin; next achieved EEF never a command',
                    original_result_executor_unchanged=True,obc_outcome_label=False)
                # Exactly the same complete10-step targets used by the loader.
                chunks=[]
                for t in range(n-9):
                    chunk=np.zeros((10,32),np.float32)
                    chunk[:,:3]=(wp[t:t+10]-ep[t])@br[t]
                    chunk[:,3:6]=Rotation.from_matrix(np.einsum('ij,tjk,kl,lm->tim',br[t].T,wr[t:t+10],er[t].T,br[t])).as_rotvec()
                    chunk[:,6:8]=actions[t:t+10,[6,10]]
                    chunks.append(chunk)
                records.append(rec);norm_states.append(state[:n-9]);norm_actions.append(np.concatenate(chunks))
                admission.append(dict(record_id=ident,status='admitted',raw_quality_receipt=rec['raw_quality_receipt'],frames=n,windows=n-9,parent_group=parent,route=row['route']))
                print(json.dumps(dict(at=now(),record_id=ident,frames=n,route=row['route'],moving_frames=rec['base_command_nonzero_frames'],admitted=len(records))),flush=True)
        except (AssertionError,ValueError,KeyError,OSError) as exc:
            admission.append(dict(record_id=ident,status='isolated_conversion_failed',reason=type(exc).__name__+': '+str(exc),raw_unchanged=True))
            print(json.dumps(admission[-1]),flush=True)
    assert records
    # Equal parent, equal original episode, equal window exposure; same hierarchy
    # also used to fit train-only normalization rather than raw frame counts.
    parents=sorted({x['parent_group'] for x in records});byparent={p:[i for i,x in enumerate(records) if x['parent_group']==p] for p in parents}
    stats={}
    for name,arrays,width in [('state',norm_states,22),('actions',norm_actions,8)]:
        vals=[];weights=[]
        for p,indices in byparent.items():
            for i in indices:
                vals.append(arrays[i]);weights.append(np.full(len(arrays[i]),1/(len(parents)*len(indices)*len(arrays[i]))))
        vals=np.concatenate(vals);weights=np.concatenate(weights);low=np.zeros(32);high=np.zeros(32)
        for j in range(width):
            order=np.argsort(vals[:,j],kind='stable');cdf=np.cumsum(weights[order]);low[j],high[j]=np.interp([.01,.99],cdf,vals[order,j])
        stats[name]=dict(q01=low.tolist(),q99=high.tolist(),valid_width=width,constant=((high-low)<1e-6).tolist())
    write(dataset/'norm-stats.json',stats)
    manifest=dict(created_at=now(),dataset=str(dataset),episodes=records,valid_windows=sum(x['windows'] for x in records),independent_episodes=len(records),parent_groups=len(parents),
        action_representation='query_relative_eef',horizon=10,execute_prefix=5,sampling='uniform Source then uniform original episode then uniform valid window',
        family_count=2,split_manifest=str(a.run/'inventory/source-split.json'),fit_data_cutoff=str(a.run/'inventory/fit-data-cutoff.json'),not_obc_outcome_labels=True,
        task_route_counts={f'{t}/{r}':sum(x['task']==t and x['route']==r for x in records) for t in ('CloseDrawer','CloseSingleDoor') for r in 'EDA'},
        target_future_observations='synchronized moving origin for supervised target only; inputs current state/RGB only; never next EEF displacement')
    write(dataset/'dataset.json',manifest);write(a.run/'inventory/dataset-binding.json',manifest)
    write(a.run/'inventory/policy-demo-admission.json',{'at':now(),'records':admission,'summary':dict(Counter(x['status'] for x in admission)),'normalization':'train-only parent/episode/frame equal weighted quantiles','no_physics_replay':True})
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['episodes']}),flush=True)

if __name__=='__main__':main()
