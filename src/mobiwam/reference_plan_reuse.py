"""Immutable artifact checks for reusing a completed development preflight."""
import hashlib
import json


def verify_plan_seal(prior, source_name):
    seal = json.loads((prior/'sealed-inputs.json').read_text())
    required = {source_name+'/'+name for name in (
        'model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json')}
    required |= {'env_config.json','executor-spec.json','waypoints.json','dock-plan.json',
                 'planning/candidate-features.json'}
    preflight=json.loads((prior/'planning/candidate-features.json').read_text())
    if preflight.get('spec',{}).get('version')=='reference-geometric-candidate-v4-actual-prefix':
        slot=preflight['selected_dock_id']
        if type(slot) is not int or not 0<=slot<=8:raise ValueError('invalid selected dock id')
        if preflight.get('D_prefix_executable'):
            required |= {f'planning/D-prefix-{slot}/'+name for name in
                         ('result.json','trace.json','prefix-prediction.npz')}
            required.add('planning/D-path.npz')
    restored = seal.get('restored_source_integration')
    if restored is not None:
        if restored != 'planning/restored-source-integration.npy':
            raise ValueError('invalid restored Source state path')
        required.add(restored)
    hashes = seal['sha256']
    if not required.issubset(hashes): raise ValueError('incomplete preflight input seal')
    for name in hashes:
        path = (prior/name).resolve()
        if not path.is_relative_to(prior.resolve()): raise ValueError('sealed input escapes plan directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError('preflight input changed: '+name)
    return seal
