"""Execute an already checked development plan after exact input/code checks.

No planner search or duplicate preview is performed at execution time. This is
not a formal collector; raw source and reference inputs must still be present.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import mujoco
from teleop_reference import Reference, stamp, write_json
from reference_executor import compile_path, run_route
from mobiwam.reference_dispatch import rejection_reason
from mobiwam.reference_plan_reuse import verify_plan_seal
from mobiwam.reference_transfer import compile_transferred_path


DEPENDENCIES = (
    'scripts/reference_execute_plan.py',
    'scripts/reference_executor.py', 'scripts/reference_geometry.py',
    'scripts/reference_geometry_v16.py', 'scripts/reference_planning.py',
    'scripts/reference_prefix_preview.py', 'scripts/teleop_reference.py',
    'scripts/reference_control_diagnostics.py', 'src/mobiwam/reference_collision.py',
    'src/mobiwam/reference_ik.py', 'src/mobiwam/reference_dispatch.py',
    'src/mobiwam/scene004.py',
    'src/mobiwam/reference_plan_reuse.py',
    'scripts/reference_stow.py', 'src/mobiwam/reference_prefix_safety.py',
    'src/mobiwam/reference_handoff.py',
    'src/mobiwam/reference_transfer.py',
    'scripts/reference_transfer_plan.py',
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--routes', required=True)
    p.add_argument('--horizon', type=int, default=2400)
    args = p.parse_args(); prior = args.plan_run.resolve(); out = args.output.resolve()
    manifest = json.loads((prior.parent/'manifest.json').read_text())
    code_root = Path(__file__).resolve().parent.parent
    for name in DEPENDENCIES:
        frozen = subprocess.check_output(['git', '-C', str(code_root), 'show', manifest['code_commit']+':'+name])
        if frozen != (code_root/name).read_bytes():
            raise ValueError('planning/execution dependency changed: '+name)
    if mujoco.__version__ != '3.2.6': raise ValueError('MuJoCo version changed')
    spec = json.loads((prior/'executor-spec.json').read_text())
    preflight = json.loads((prior/'planning/candidate-features.json').read_text())
    source = Path(preflight['source']).resolve()
    if source.parent != prior: raise ValueError('plan Source is not local to the supplied plan')
    seal = verify_plan_seal(prior, source.name)
    restored = preflight.get('restored_source_integration')
    if restored != seal.get('restored_source_integration'):
        raise ValueError('restored Source state is not sealed consistently')
    if seal['planning_code_commit'] != manifest['code_commit']:
        raise ValueError('input seal and planning provenance disagree')
    routes = args.routes.split(',')
    rejected = {route: rejection_reason(preflight, route, True) for route in routes}
    if any(rejected.values()): raise ValueError('full-plan dispatch rejected: '+str(rejected))
    if not preflight.get('source_integration_unchanged'): raise ValueError('planning changed Source')
    if hashlib.sha256((source/'integration.npy').read_bytes()).hexdigest() != spec['source_sha256']:
        raise ValueError('planned Source state changed')
    out.mkdir(parents=True, exist_ok=False); dest = out/source.name; dest.mkdir()
    for name in ('model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json'):
        shutil.copy2(source/name, dest/name)
    shutil.copy2(prior/'env_config.json', out/'env_config.json')
    cfg = json.loads((out/'env_config.json').read_text())
    ref = Reference(argparse.Namespace(output=str(out),task=cfg['env_name'],layout=0,style=0,seed=7,
        self_test=True,source=str(dest),replay_attempt=None,resume_attempt=None,width=1920,height=1080))
    ref.label = ('autonomous_development_reference_feedback_transfer_pilot_v1'
                 if spec.get('transfer_mode') else 'autonomous_development_reference_feedback_v18')
    try:
        ref.restore()
        if spec.get('transfer_mode') == 'moving_target_handle_frame':
            points, transfer = compile_transferred_path(ref, Path(spec['reference']))
            planned_transfer = json.loads((prior/'transfer-receipt.json').read_text())
            # Both stages make isolated Source copies under their own output;
            # compare the semantic transfer, while Source bytes are sealed.
            transfer.pop('new_source', None)
            planned_transfer.pop('new_source', None)
            if transfer != planned_transfer:
                raise ValueError('reference transfer differs from preflight')
        elif spec.get('transfer_mode') is None:
            points = compile_path(ref, Path(spec['reference']))
        else:
            raise ValueError('unknown reference transfer mode')
        fresh = json.loads(json.dumps(points,default=lambda x:x.tolist()))
        if fresh != json.loads((prior/'waypoints.json').read_text()):
            raise ValueError('reference geometry changed since preflight')
        ref.dock_plan = json.loads((prior/'dock-plan.json').read_text())
        if ref.dock_plan['selected']['id'] != preflight['selected_dock_id']:
            raise ValueError('selected dock changed since preflight')
        expected = np.load(prior/restored if restored else source/'integration.npy',allow_pickle=False)
        if expected.shape!=ref.integration().shape or not np.isfinite(expected).all():
            raise ValueError('invalid planned Source integration state')
        error = float(np.max(abs(ref.integration()-expected)))
        if error > 1e-10: raise ValueError('Source restore differs from planned state')
        files = [source/name for name in ('model.xml','integration.npy','ep_meta.json','rng.json')]
        files += [prior/name for name in ('env_config.json','waypoints.json','dock-plan.json','planning/candidate-features.json')]
        write_json(out/'plan-reuse-receipt.json',dict(created_at=stamp(),plan_run=str(prior),
            planning_code=manifest['code_commit'],production_dependencies_identical=True,
            reference_geometry_identical=True,source_restore_max_error=error,require_full_plan=True,
            source_restore_comparison='sealed post-restore full integration' if restored else 'legacy saved Source integration',
            raw_source_restore_max_error=float(np.max(abs(ref.integration()-np.load(source/'integration.npy',allow_pickle=False)))),
            formal_train_ready=False,inputs=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]))
        write_json(out/'recording-provenance.json',dict(data_kind='autonomous_development',
            planner=preflight['spec']['version'],transfer_mode=spec.get('transfer_mode'),
            formal_train_ready=False))
        results = [run_route(ref,route,points,args.horizon) for route in routes]
        write_json(out/'completed.json',dict(ended_at=stamp(),attempts=results))
    finally:
        if ref.observation_renderer is not None: ref.observation_renderer.close()
        if ref.renderer is not None: ref.renderer.close()
        ref.env.close()


if __name__ == '__main__': main()
