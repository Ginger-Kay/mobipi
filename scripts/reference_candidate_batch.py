"""Join frozen Source contexts with planned scalar features; no outcome read."""
import argparse
import datetime
import json
from pathlib import Path
import numpy as np
from mobiwam.extract_features import sha256_file
from mobiwam.reference_feature_interface import assemble_input


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--contexts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--planning-stage',default='safety-refresh',help='Planning subdirectory within run; default retains v17 compatibility')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((args.run/'plan.json').read_text())
    context_manifest=json.loads((args.contexts/'manifest.json').read_text())
    contexts={x['id']:x for x in context_manifest['sources']};rows=[];arrays=[]
    for label,cfg in plan['sources'].items():
        context_id=cfg['context_id'];entry=contexts[context_id]
        path=args.contexts/context_id/'context.npy'
        if sha256_file(path)!=entry['context_sha256']:raise ValueError('context bytes changed')
        context_source=Path(entry['observation_container']).parent.parent.parent
        for filename in ('model.xml','integration.npy'):
            if sha256_file(context_source/filename)!=sha256_file(Path(cfg['source'])/filename):
                raise ValueError('Source identity differs from frozen context: '+label)
        candidate_path=args.run/args.planning_stage/label/'run/planning/candidate-features.json'
        candidates=json.loads(candidate_path.read_text());context=np.load(path,allow_pickle=False)
        for row in candidates['records']:
            vector=assemble_input(context,row['features']);arrays.append(vector)
            rows.append(dict(source_id=label,route=row['route'],lineage_group=entry['lineage_group'],
                candidate_hard_valid=row['hard_valid'],candidate_path=str(candidate_path.resolve()),
                candidate_sha256=sha256_file(candidate_path),context_path=str(path.resolve()),
                context_sha256=entry['context_sha256'],features=row['features']))
    x=np.stack(arrays);assert x.shape==(len(rows),1045) and np.isfinite(x).all()
    np.savez_compressed(args.output/'inputs.npz',inputs=x,
        source_ids=np.array([r['source_id'] for r in rows]),routes=np.array([r['route'] for r in rows]),
        candidate_hard_valid=np.array([r['candidate_hard_valid'] for r in rows]))
    manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='development_input_interface_complete',review_status='pending',formal_train_ready=False,
        input_shape=list(x.shape),candidate_feature_count=21,missing_features=0,
        hard_valid_candidates=sum(r['candidate_hard_valid'] for r in rows),
        independent_lineage_groups=sorted(set(r['lineage_group'] for r in rows)),
        no_outcome_table_read=True,no_training_labels_exported=True,
        scope='planned features plus frozen time-zero context; historical development outcomes already exist',
        note='Complete numeric inputs do not establish planner hard validity or formal trainability.',rows=rows)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print('exported',x.shape,'hard_valid',manifest['hard_valid_candidates'],flush=True)


if __name__=='__main__':main()
