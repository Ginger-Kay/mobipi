"""Load a checked planning-preview prefix, never a task rollout outcome."""
import hashlib
import json
from pathlib import Path
import numpy as np


def load_preview_prefix(folder, initial_qpos, candidate_id):
    folder = Path(folder)
    result = json.loads((folder / 'result.json').read_text())
    if not result['valid'] or result['candidate_id'] != candidate_id:
        raise ValueError('invalid prefix preview or candidate')
    monitor = result.get('joint_margin', {})
    minimum = monitor.get('minimum') or {}
    margin=minimum.get('margin_rad',0.)
    if (monitor.get('required_strictly_greater_than_rad') != .015
            or not np.isfinite(margin) or margin <= .015 or result.get('failure') is not None):
        raise ValueError('prefix dynamic joint margin not verified')
    archive = folder / 'prefix-prediction.npz'
    expected = result.get('prefix_qpos_actions_sha256')
    if expected is None or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError('unsealed or changed prefix states')
    with np.load(archive, allow_pickle=False) as data:
        states = data['qpos'].copy()
        actions = data['actions']
        if (states.ndim != 2 or states.shape[1:] != np.asarray(initial_qpos).shape
                or actions.ndim != 2 or len(states) < 2 or len(states) != len(actions) + 1
                or not np.isfinite(states).all() or not np.isfinite(actions).all()
                or not np.array_equal(states[0], initial_qpos)
                or result['control_steps'] != len(actions)):
            raise ValueError('prefix state alignment or source mismatch')
    trace_path = folder / 'trace.json'
    if hashlib.sha256(trace_path.read_bytes()).hexdigest() != result.get('prefix_trace_sha256'):
        raise ValueError('unsealed or changed prefix trace')
    trace = json.loads(trace_path.read_text())
    if len(trace) != len(states) - 1:
        raise ValueError('prefix trace length mismatch')
    phases = [r['phase'] for r in trace]
    if not set(phases).issubset({'stow', 'navigate'}):
        raise ValueError('non-prefix phase')
    # These states are already generated, not IK solutions. Zero placeholders
    # exclude the prefix from the manipulation IK residual; not tracking errors.
    return states, phases, np.zeros((len(phases), 2))
