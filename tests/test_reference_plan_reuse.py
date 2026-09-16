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


def test_restored_solver_state_must_be_sealed_and_cannot_be_changed(tmp_path):
    names=['source/'+name for name in ('model.xml','integration.npy','ep_meta.json',
        'rng.json','source.json','target-binding.json')]
    names+=['env_config.json','executor-spec.json','waypoints.json','dock-plan.json',
            'planning/candidate-features.json','planning/restored-source-integration.npy']
    hashes={}
    for name in names:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    restored=names[-1]
    seal=tmp_path/'sealed-inputs.json'
    seal.write_text(json.dumps(dict(sha256=hashes,restored_source_integration=restored)))
    verify_plan_seal(tmp_path,'source')
    (tmp_path/restored).write_bytes(b'changed warm-start')
    with pytest.raises(ValueError,match='preflight input changed'):verify_plan_seal(tmp_path,'source')
    del hashes[restored]
    seal.write_text(json.dumps(dict(sha256=hashes,restored_source_integration=restored)))
    with pytest.raises(ValueError,match='incomplete'):verify_plan_seal(tmp_path,'source')
