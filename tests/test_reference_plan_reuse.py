import hashlib
import json
import pytest
from mobiwam.reference_plan_reuse import verify_plan_seal


def test_reuse_rejects_changed_geometry_and_incomplete_seal(tmp_path):
    source_files=['model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json']
    names=['source/'+name for name in source_files]+['env_config.json','executor-spec.json',
        'waypoints.json','dock-plan.json','planning/candidate-features.json']
    hashes={}
    for name in names:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    seal=tmp_path/'sealed-inputs.json';seal.write_text(json.dumps({'sha256':hashes}))
    verify_plan_seal(tmp_path,'source')
    (tmp_path/'source/model.xml').write_text('changed obstacle geometry')
    with pytest.raises(ValueError,match='preflight input changed'):verify_plan_seal(tmp_path,'source')
    del hashes['source/model.xml'];seal.write_text(json.dumps({'sha256':hashes}))
    with pytest.raises(ValueError,match='incomplete'):verify_plan_seal(tmp_path,'source')
