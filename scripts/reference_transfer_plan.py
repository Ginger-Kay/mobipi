"""Strict Source-only transfer and full E/D/A preflight for a new scene."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

import numpy as np

from teleop_reference import Reference, stamp, write_json
from reference_geometry_v16 import plan_dock
from reference_planning import compile_candidates
from mobiwam.reference_plan_reuse import verify_plan_seal
from mobiwam.reference_transfer import compile_transferred_path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    reference = args.reference.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    copied = out / source.name
    copied.mkdir()
    for name in ('model.xml', 'integration.npy', 'ep_meta.json', 'rng.json',
                 'source.json', 'target-binding.json'):
        shutil.copy2(source / name, copied / name)
        if sha(source / name) != sha(copied / name):
            raise ValueError('Source copy changed')
    shutil.copy2(source.parent / 'env_config.json', out / 'env_config.json')
    cfg = json.loads((out / 'env_config.json').read_text())
    ref = None
    try:
        ref = Reference(argparse.Namespace(output=str(out), task=cfg['env_name'], layout=0,
                                           style=0, seed=7, self_test=True, source=str(copied),
                                           replay_attempt=None, resume_attempt=None,
                                           width=1920, height=1080))
        ref.label = 'development_reference_conditioned_transfer_pilot'
        ref.restore()
        points, transfer = compile_transferred_path(ref, reference)
        write_json(out / 'transfer-receipt.json', transfer)
        write_json(out / 'waypoints.json', points)
        write_json(out / 'executor-spec.json', {
            'created_at': stamp(), 'version': 'reference-feedback-transfer-pilot-v1',
            'transfer_mode': 'moving_target_handle_frame',
            'reference': str(reference), 'source': str(source),
            'source_sha256': sha(source / 'integration.npy'),
            'routes': 'E,D,A', 'horizon': 2400, 'require_full_plan': True,
            'base_speed_caps': {'CloseDrawer': .015, 'CloseSingleDoor': .09},
            'collision_margin_m': .0005, 'formal_train_ready': False,
            'scope': 'approved development pilot; no route outcomes yet',
        })
        ref.dock_plan = plan_dock(ref, points)
        candidates = compile_candidates(ref, points, ref.dock_plan, out / 'planning')
        write_json(out / 'dock-plan.json', ref.dock_plan)
        files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()
                 and p.name != 'sealed-inputs.json'}
        write_json(out / 'sealed-inputs.json', {
            'sha256': files,
            'restored_source_integration': 'planning/restored-source-integration.npy',
            'transfer_mode': 'moving_target_handle_frame',
            'reference_trace_sha256': sha(reference / 'trace.jsonl'),
            'reference_demo_sha256': sha(reference / 'demo.hdf5'),
        })
        verify_plan_seal(out, copied.name)
        write_json(out / 'completed.json', {
            'ended_at': stamp(), 'status': 'completed_pre_outcome_preflight',
            'routes': [{'route': row['route'], 'hard_valid': row['hard_valid']}
                       for row in candidates['records']],
            'source_state_unchanged': candidates['source_integration_unchanged'],
            'route_outcomes': 0, 'formal_train_ready': False,
        })
        print('PREFLIGHT', cfg['env_name'], source, [(row['route'], row['hard_valid'])
                                                    for row in candidates['records']], flush=True)
    except Exception:
        write_json(out / 'failure.json', {
            'ended_at': stamp(), 'status': 'pre_outcome_rejected',
            'traceback': traceback.format_exc(), 'route_outcomes': 0,
            'formal_train_ready': False,
        })
        raise
    finally:
        if ref is not None:
            if ref.observation_renderer is not None:
                ref.observation_renderer.close()
            if ref.renderer is not None:
                ref.renderer.close()
            ref.env.close()


if __name__ == '__main__':
    main()
