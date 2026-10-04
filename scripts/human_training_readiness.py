"""Prepare a non-training human-data index and exercise an HDF5 batch reader.

No split assignment, automatic-executor target fabrication, model or optimizer.
"""
import argparse
import csv
import json
from datetime import datetime,timezone
from pathlib import Path

import h5py
import numpy as np


def load(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def prepare(batch,output):
    output.mkdir(exist_ok=False)
    pointer=load(batch/'manifests/current-human-training-candidates.json')
    candidates=load(pointer['manifest'])
    entries=candidates['entries']
    allowed={Path(x['attempt']).name:x for x in entries}
    assert len(allowed)==len(entries)==3
    audit_pointer=load(batch/'manifests/current-qualification.json')
    trio=load(audit_pointer['qualification'])
    audited={Path(x['attempt']).name:x for x in trio['routes'].values()}
    raw=list(csv.DictReader((batch/'attempt-index.csv').open()))
    ids=[r['attempt_id'] for r in raw];assert len(ids)==len(set(ids))
    indexed=[]
    for r in raw:
        item={**r,'training_selected':False,'split':'unassigned','autonomous_executor_label':False,
              'audit_overlay':None,'future_training_allowed_by_user':r['attempt_id'] in allowed}
        if r['attempt_id'] in allowed:
            c=allowed[r['attempt_id']];qual=load(c['qualification'])
            assert qual['machine_qualification_pass'] and Path(c['trajectory'])==Path(r['trajectory_path'])
            item.update(scene_family=c['scene_family'],audit_overlay=c['qualification'],
                        machine_qualified_success=True,selected_reference=True)
        elif r['attempt_id'] in audited:
            q=audited[r['attempt_id']]
            item.update(scene_family='DR-PILOT-family',audit_overlay=str(Path(audit_pointer['qualification']).parent/r['route']/'qualification.json'),
                        machine_qualified_success=q['machine_qualified_success'],selected_reference=False)
        else:
            source=Path(r['trajectory_path']).parent.parent.parent
            cfg=source.parent/'pilot.json'
            item.update(scene_family=load(cfg)['scene_family_id'] if cfg.exists() else 'unresolved',
                        machine_qualified_success=False,selected_reference=False)
        indexed.append(item)
    write(output/'attempt-audit-overlay.json',dict(raw_index=str(batch/'attempt-index.csv'),entries=indexed,
          raw_index_modified=False,scope='Inherited qualification overlay only; unreviewed records are not newly audited'))
    dry=[];batch_actions=[];batch_proprio=[];batch_next_proprio=[];sample_ids=[];keys=[]
    for c in entries:
        with h5py.File(c['trajectory'],'r') as f:
            g=f['data/demo_0'];n=g['actions'].shape[0];assert n==c['steps']
            assert g['actions'].shape==(n,12) and g['states'].shape[0]==n+1
            obs=g['obs'];assert all(d.shape[0]==n+1 for d in obs.values())
            keys.append(set(obs.keys()))
            assert np.isfinite(g['actions'][:]).all()
            assert np.isfinite(obs['robot0_proprio-state'][:]).all()
            times=obs['sim_time'][:];assert np.allclose(np.diff(times),.05,atol=1e-9,rtol=0)
            indices=sorted(set([0,n//2,n-1]))
            for i in indices:
                # x[t], action[t], x[t+1]; do not shift the label onto future observations.
                batch_actions.append(g['actions'][i]);batch_proprio.append(obs['robot0_proprio-state'][i])
                batch_next_proprio.append(obs['robot0_proprio-state'][i+1]);sample_ids.append(dict(attempt=Path(c['attempt']).name,step=i))
                for name in ['robot0_agentview_left_image','robot0_agentview_right_image','robot0_eye_in_hand_image']:
                    rgb=obs[name][i];assert rgb.shape==(256,256,3) and rgb.dtype==np.uint8
            dry.append(dict(attempt=c['attempt'],trajectory=c['trajectory'],scene_family=c['scene_family'],
                       steps=n,action_shape=list(g['actions'].shape),state_shape=list(g['states'].shape),
                       proprio_shape=list(obs['robot0_proprio-state'].shape),checked_indices=indices,
                       source_count_increment=0,split='unassigned',record_type=c['record_type']))
    np.savez_compressed(output/'reader-dry-batch.npz',actions=np.stack(batch_actions),proprio=np.stack(batch_proprio),next_proprio=np.stack(batch_next_proprio))
    assert len({r['scene_family'] for r in dry})==2
    write(output/'reader-receipt.json',dict(at=datetime.now(timezone.utc).isoformat(),records=dry,sampled_transitions=sample_ids,
          total_candidate_transitions=sum(x['steps'] for x in dry),common_observation_keys=sorted(set.intersection(*keys)),
          observation_alignment='obs[t], action[t], obs[t+1]; T actions / T+1 observations',
          full_physics_state_widths=sorted(set(x['state_shape'][1] for x in dry)),
          state_stacking_warning='Task physics state widths differ. Do not pad/concatenate without a versioned training schema.',
          trained=False,model_created=False,formal_train_ready=False))
    write(output/'readiness.json',dict(at=datetime.now(timezone.utc).isoformat(),raw_attempt_rows=len(raw),
          human_practice=sum(r['record_type']=='practice' for r in raw),human_primary=sum(r['record_type']=='primary' for r in raw),
          machine_qualified_successful_human_records=sum(r['machine_qualified_success'] for r in indexed),
          explicitly_authorized_future_training_candidates=len(entries),independent_environment_families=2,
          training_started=False,formal_train_ready=False,split_assigned=False,
          remaining=['Finish bounded human scene collection and audits','Specify intended learning task and versioned human-data schema',
                     'If learning automatic E/D/A selection, obtain actual automatic-executor paired outcomes under a separately approved run',
                     'Keep environment families grouped; these operator-viewed records are not blind test data']))
    print('HUMAN_DATA_READER_READY',sum(x['steps'] for x in dry),'candidate transitions; no training',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();prepare(args.batch,args.output)
