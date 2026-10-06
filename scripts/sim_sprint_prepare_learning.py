"""Freeze a compatible reference view only after policy readiness is closed."""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from mobiwam.sim_sprint_learning import require_group_split,ROUTES

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    readiness=json.loads((r/'policy/readiness.json').read_text())
    assert readiness['main_controller']=='reference_fallback'
    prior=Path('/share/personal/chensiyu/haokaijiang/MobiWAM/artifacts/MMWAM-OBC-002-DR/DR-v0.4/r3-contact-rule-removal/20261003T055050Z-r3-receipt-reuse')
    z=np.load(prior/'dataset/supervision.npz',allow_pickle=False)
    onehot=np.stack([z['route']==route for route in ROUTES],axis=1).astype(np.float32)
    X=np.concatenate([z['X'],onehot],axis=1);assert X.shape==(108,1048)
    require_group_split(z['group_id'][z['split']=='train'],z['group_id'][z['split']=='validation'])
    for split,name in [('train','train-only'),('validation','development-validation')]:
        idx=z['split']==split
        np.savez_compressed(r/'training'/f'{name}.npz',X=X[idx],y=z['y'][idx],mask=z['mask'][idx],
            group_id=z['group_id'][idx],route=z['route'][idx],task=z['task'][idx])
    binding=dict(created_at=datetime.now(timezone.utc).isoformat(),controller='reference_fallback',parent_dataset=str(prior/'dataset/supervision.npz'),
        parent_dataset_sha256='97dc64899af246bfc1f5969c56d1af15520f1879700dc657f1a235f026e4f3fd',
        compatibility='same frozen two-task reference-conditioned feedback executor and R3 monitor/mask/context; old reference identities retained',
        schema='frozen1024 visual/proprio context +21 geometry +3 EDA onehot, total1048',
        training_sources=24,development_validation_sources=12,old_validation_reused=True,
        human_failure_labels_used=False,policy_labels_mixed=False,sealed_test_sources_in_dataset=0,
        dataset_hashes={name:hashlib.sha256((r/'training'/f'{name}.npz').read_bytes()).hexdigest() for name in ('train-only','development-validation')},
        mask=z['mask'].sum(axis=0).tolist(),formal_train_ready=False)
    (r/'training/dataset-binding.json').write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps(binding),flush=True)

if __name__=='__main__':main()
