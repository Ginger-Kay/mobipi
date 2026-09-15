"""Export actual frozen CLIP Source contexts and explicit planner gaps.

Development-only: the historical route outcomes already exist. This is not
the pre-outcome encoder freeze required for future formal collection.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from mobiwam.extract_features import FrozenCLIPVisionEncoder, VISUAL_KEYS, sha256_file
from mobiwam.reference_feature_interface import (
    PROPRIO_KEYS, candidate_readiness, source_context, source_observations,
)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    manifest = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='preparing', review_status='pending', scope='development feature interface',
        formal_train_ready=False, historical_outcomes_already_exist=True,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        model_path=str(args.model.resolve()), revision=args.model.name,
        model_files={p.name: {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
                     for p in args.model.iterdir() if p.name in ('config.json', 'pytorch_model.bin')},
        model='openai/clip-vit-large-patch14', encoder_output='CLIPVisionModel.pooler_output',
        preprocessing='uint8 NHWC -> float NCHW /255; bicubic224 antialias; CLIP mean/std; per-view L2 normalization',
        observations='time index0 only, before actions[0]; three policy cameras and explicit proprio allowlist',
        visual_keys=list(VISUAL_KEYS), proprio_keys=list(PROPRIO_KEYS),
        context_shape=[4,1024], input_shape_if_complete=[1045],
        pooling='mean of three normalized vision tokens and one existing tanh proprio token',
        device=args.device, precision='bfloat16 autocast on CUDA; float32 output',
        candidate_schema='scene004.CANDIDATE_FEATURE_FIELDS,21D; missing values block assembly',
        plan=str(args.plan.resolve()), plan_sha256=sha256_file(args.plan), sources=[])
    dump(args.output/'manifest.json', manifest)
    encoder = FrozenCLIPVisionEncoder(args.model, device=args.device, batch_size=3)
    for entry in plan['sources']:
        run = Path(entry['run'])
        # Fixed route A is an observation container, never selected by outcome.
        paths = sorted(run.glob('source-*/A/attempt-*/demo.hdf5'))
        if len(paths) != 1:
            raise ValueError(f'Expected one explicit route-A observation container: {run}')
        observations = source_observations(paths[0])
        digest = hashlib.sha256()
        for key in sorted(observations):
            a = np.ascontiguousarray(observations[key])
            digest.update(key.encode()); digest.update(str(a.shape).encode())
            digest.update(str(a.dtype).encode()); digest.update(a.tobytes())
        context = source_context(observations, encoder)
        target = args.output/entry['id']; target.mkdir()
        np.save(target/'context.npy', context, allow_pickle=False)
        task = json.loads((run/'env_config.json').read_text())['env_name']
        rows = []
        for route in ('E','D','A'):
            known = {'route_E': float(route=='E'), 'route_D': float(route=='D'),
                'route_A': float(route=='A'), 'task_CloseDrawer': float(task=='CloseDrawer'),
                'task_CloseSingleDoor': float(task=='CloseSingleDoor'),
                'stage_precontact': 1., 'simulator_oracle_pre_outcome': 1.}
            rows.append({'route': route, **candidate_readiness(known)})
        dump(target/'candidate-readiness.json', rows)
        row = {**entry, 'observation_container':str(paths[0].resolve()),
            'source_observation_sha256':digest.hexdigest(), 'context_sha256':sha256_file(target/'context.npy'),
            'context_finite': bool(np.isfinite(context).all()),
            'candidate_complete_count':sum(r['complete'] for r in rows),
            'provenance_note':'physical Source input frame from completed development recording; no future frames, actions, outcomes or trace read'}
        manifest['sources'].append(row)
        dump(args.output/'manifest.json', manifest)
        print(entry['id'], context.shape, 'candidate gaps',len(rows[0]['missing']), flush=True)
    manifest.update(status='contexts_exported_planner_adapter_incomplete',
                    ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dump(args.output/'manifest.json',manifest)


if __name__ == '__main__':
    main()
