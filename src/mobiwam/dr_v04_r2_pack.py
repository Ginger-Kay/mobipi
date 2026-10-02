"""Pack already-frozen Source inputs and audited labels; no encoder or model."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from mobiwam.scene004 import CANDIDATE_FEATURE_FIELDS,candidate_feature_vector,build_minimal_input

LABELS=('success','progress','failure','base_path_m','completion_time_s')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def labels(rows):
    y=np.full((len(rows),5),np.nan,dtype=np.float32)
    for i,row in enumerate(rows):
        for j,key in enumerate(LABELS):
            if row.get(key) is not None:y[i,j]=float(row[key])
    mask=np.isfinite(y)
    route_eligible=np.asarray([r.get('machine_eligible_for_gate') is True and bool(mask[i].all()) for i,r in enumerate(rows)])
    group_ok={g:all(route_eligible[i] for i,r in enumerate(rows) if r['group_id']==g) and sum(r['group_id']==g for r in rows)==3 for g in {r['group_id'] for r in rows}}
    return y,mask,route_eligible,np.asarray([group_ok[r['group_id']] for r in rows])

def cached_inputs(run,freeze,roster,root):
    path=run/'gate/preoutcome-inputs.npz';receipt=run/'gate/preoutcome-inputs-receipt.json'
    keys=[(r['group_id'],route) for r in roster for route in ('E','D','A')]
    if path.exists() or receipt.exists():
        if not (path.exists() and receipt.exists()):raise ValueError('partial input cache; inspect before repair')
        meta=json.loads(receipt.read_text())
        if meta['freeze_sha256']!=sha(freeze) or sha(path)!=meta['npz_sha256']:raise ValueError('input cache binding changed')
        data=np.load(path,allow_pickle=False)
        if list(zip(data['group_id'].tolist(),data['route'].tolist()))!=keys:raise ValueError('input row order changed')
        return data['X'],meta
    spec=json.loads(freeze.read_text())
    original=subprocess.check_output(['git','-C',str(root),'show',spec['planning_code_commit']+':src/mobiwam/scene004.py'])
    if original!=(root/'src/mobiwam/scene004.py').read_bytes():raise ValueError('frozen input assembler differs')
    xs=[];lineage=[]
    for row in roster:
        if row['split'] not in ('train','validation'):raise ValueError('sealed test cannot load features')
        context=Path(row['context_path']);features=Path(row['candidate_features'])
        if sha(context)!=row['context_sha256'] or sha(features)!=row['candidate_features_sha256']:raise ValueError('immutable Source inputs changed')
        tokens=np.load(context,allow_pickle=False);candidate=json.loads(features.read_text())
        records={r['route']:r for r in candidate['records']}
        if len(candidate['records'])!=3 or set(records)!={'E','D','A'}:raise ValueError('candidate denominator differs')
        for route in ('E','D','A'):
            f=records[route]['features']
            if set(f)!=set(CANDIDATE_FEATURE_FIELDS):raise ValueError('planner feature allowlist differs')
            if f['hard_valid']!=1. or f['stage_precontact']!=1.:raise ValueError('unfrozen invalid/other-stage input')
            if any(f['route_'+r]!=float(r==route) for r in ('E','D','A')):raise ValueError('route input identity differs')
            if any(f['task_'+task]!=float(row['task']==task) for task in ('CloseDrawer','CloseSingleDoor')):raise ValueError('task input identity differs')
            xs.append(build_minimal_input(tokens,candidate_feature_vector(f)))
        lineage.append(dict(group_id=row['group_id'],split=row['split'],context=str(context),context_sha256=row['context_sha256'],candidate_features=str(features),candidate_features_sha256=row['candidate_features_sha256']))
    X=np.stack(xs)
    if X.shape!=(108,1045):raise ValueError('108x1045 immutable inputs required')
    np.savez_compressed(path,X=X,group_id=np.asarray([k[0] for k in keys]),route=np.asarray([k[1] for k in keys]))
    meta=dict(freeze=str(freeze),freeze_sha256=sha(freeze),npz=str(path),npz_sha256=sha(path),shape=list(X.shape),lineage=lineage,
              feature_order=list(CANDIDATE_FEATURE_FIELDS),context='existing 4x1024 frozen tokens, mean pooling; no encoder invocation',sealed_test_read=False,model_inference=0,training=0)
    receipt.write_text(json.dumps(meta,indent=2)+'\n')
    return X,meta

def pack(run,freeze,roster,rows,output,root):
    X,meta=cached_inputs(run,freeze,roster,root);y,mask,eligible,group_eligible=labels(rows)
    path=output/'paired-supervision-candidate.npz'
    np.savez_compressed(path,X=X,y=y,label_observed=mask,route_machine_eligible=eligible,group_machine_eligible=group_eligible,
        group_id=np.asarray([r['group_id'] for r in rows]),route=np.asarray([r['route'] for r in rows]),split=np.asarray([r['split'] for r in rows]),
        known_outcome_exception=np.asarray([r['outcome_previously_observed'] for r in rows]))
    receipt=dict(path=str(path),sha256=sha(path),X_shape=list(X.shape),y_shape=list(y.shape),labels_order=list(LABELS),
        observed_label_values=int(mask.sum()),machine_eligible_routes=int(eligible.sum()),machine_eligible_groups=int(group_eligible.sum()//3),
        missing_value='NaN with label_observed=false, never zero-imputed',input_receipt=str(run/'gate/preoutcome-inputs-receipt.json'),
        dataset_kind='paired_supervision_candidate_pending_full_qualification_and_research_review',formal_train_ready=False,training_authorized=False,
        encoder_invocations=0,model_fits=0,model_inference=0,sealed_test_read=False)
    (output/'paired-supervision-candidate.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
