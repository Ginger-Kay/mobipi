"""Correct candidate slot encoding in a new immutable pre-outcome view."""
import argparse,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from mobiwam.scene004 import build_minimal_input,candidate_feature_vector,geometry_rule_select

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
    old=r/'inputs/geometry-prefix-v1';out=r/'inputs/geometry-prefix-v2';out.mkdir(exist_ok=False)
    assert (old/'completed.json').exists()
    roster=json.loads((old/'roster.json').read_text());(out/'roster.json').write_text(json.dumps(roster,indent=2)+'\n');results=[]
    for g in roster['groups']:
        d=old/g['config_id'];dest=out/g['config_id'];dest.mkdir();src=d/'features.json'
        if not src.exists():
            results.append(dict(config_id=g['config_id'],status='unavailable_original_inputs'));continue
        q=json.loads(src.read_text());ctxdir=Path(q['context_manifest']).parent
        context=np.load(ctxdir/'context.npy',allow_pickle=False)
        for row in q['routes']:
            route=row['route_family'];plan=json.loads((d/(route+'-plan.json')).read_text()) if route!='E' else None
            row['derived']['slot_index_normalized']=int(plan['primary']['candidate_id'].rsplit('-',1)[-1])/4. if plan else 0.
            row['field_notes']['slot_index_normalized']='geometric proposal index/4; no Source or split identity'
            X=np.r_[build_minimal_input(context,candidate_feature_vector(row['derived'])),[route==x for x in 'EDA']].astype(np.float32)
            assert X.shape==(1048,) and np.isfinite(X).all();np.save(dest/(route+'-X.npy'),X)
        q.update(parent_features=str(src),parent_features_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),correction='FEATURE-SLOT-001: encode candidate proposal index, remove Source roster order')
        (dest/'features.json').write_text(json.dumps(q,indent=2)+'\n')
        results.append(dict(config_id=g['config_id'],status='corrected_preoutcome',routes=len(q['routes'])))
    (out/'completed.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),groups=results,parent_version=str(old),
        original_version_retained=True,new_env_steps=0,new_task_outcomes=0,obc_fits_before_correction=0),indent=2)+'\n')
    print(json.dumps(results))
if __name__=='__main__':main()
