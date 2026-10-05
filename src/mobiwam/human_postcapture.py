"""Post-capture engineering primitives. Never relabel human data as autonomous."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, attempts=3, delay=.05):
    """Bounded read of a live non-atomic status file; persistent corruption fails."""
    for attempt in range(attempts):
        try:
            return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt+1 == attempts:
                raise
            time.sleep(delay)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.tmp-{os.getpid()}')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp, path)


def signature(attempt):
    """Bind small manifests. Full payload hashes are checked by identity audit."""
    attempt = Path(attempt).resolve()
    names = ['result.json', 'collection-metadata.json', 'task-video-manifest.json',
             'panoramic-binding.json', 'formal-native-substeps-receipt.json']
    values = {}
    for name in names:
        path = attempt/name
        if not path.is_file():
            raise ValueError('Finalized recording missing '+name)
        values[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['safety-stop.json', 'partial-control-step.npz']:
        path = attempt/name
        values[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    payload_stats={}
    for name in ['demo.hdf5','formal-native-substeps.npz','trace.jsonl','original.mp4','panoramic.mp4']:
        stat=(attempt/name).stat()
        payload_stats[name]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    return dict(attempt=str(attempt), files=values, payload_stats=payload_stats)


def claim(ledger, attempt, stage, input_signature):
    """Atomic reservation shared across runs; interrupted stages never auto-retry."""
    key = hashlib.sha256(str(Path(attempt).resolve()).encode()).hexdigest()
    path = Path(ledger)/key/(stage+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(at=stamp(), pid=os.getpid(), status='reserved', stage=stage,
                 attempt=str(Path(attempt).resolve()), input_signature=input_signature)
    try:
        with path.open('x') as stream:
            json.dump(value, stream);stream.flush();os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError('Stage already reserved; inspect receipt, never automatically replay again: '+str(path)) from exc
    return path


def numeric_alignment(actions, states, rows, native, control_dt=.05):
    n = len(rows)
    if n < 1:
        raise ValueError('Zero complete-frame recording needs separate diagnosis; preserve outcome, do not retry')
    nq = len(rows[0]['before']['qpos']);nv = len(rows[0]['before']['qvel'])
    if actions.shape != (n, 12) or states.shape != (n+1, 1+nq+nv):
        raise ValueError('Action/state dimensions differ from actual task')
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError('Nonfinite states/actions')
    for i,row in enumerate(rows):
        for boundary,index in [('before',i), ('after',i+1)]:
            observed = np.r_[row[boundary]['qpos'], row[boundary]['qvel']]
            if not np.array_equal(observed, states[index,1:]):
                raise ValueError('Trace/HDF5 state mismatch')
    q=np.asarray(native['qpos']);times=np.asarray(native['sim_time']);indices=np.asarray(native['step_index'])
    extra=len(indices)-n*25
    if q.shape != (len(indices)+1,nq) or len(times)!=len(q) or not 0<=extra<25:
        raise ValueError('Native tail alignment differs')
    if not np.isfinite(q).all() or not np.isfinite(times).all():
        raise ValueError('Nonfinite native data')
    if not np.array_equal(indices[:n*25],np.repeat(np.arange(n),25)):
        raise ValueError('Native control boundaries differ')
    if extra and not np.all(indices[n*25:]==n):
        raise ValueError('Partial native boundary differs')
    if not np.array_equal(q[::25][:n+1],states[:,1:1+nq]):
        raise ValueError('Native/control state mismatch')
    if not np.allclose(np.diff(times),control_dt/25,atol=1e-10,rtol=0):
        raise ValueError('Native dt mismatch')
    if not np.allclose(np.diff(states[:,0]),control_dt,atol=1e-10,rtol=0):
        raise ValueError('Control dt mismatch')
    return dict(steps=n,nq=nq,nv=nv,partial_native_steps=extra,
                native_intervals=len(indices),simulation_seconds=(n*25+extra)*control_dt/25)


def pair_records(records):
    """Failures remain in the paired denominator; duplicate or mixed starts reject."""
    groups={}
    for row in records:
        if row['record_type']!='primary':continue
        key=(row['source'],row['config_version'])
        group=groups.setdefault(key,[]);group.append(row)
    out=[]
    for (source,version),rows in groups.items():
        routes=[r['route'] for r in rows]
        if len(routes)!=len(set(routes)):
            raise ValueError('Duplicate primary route; do not pick the successful one')
        initial=[np.load(r['initial_integration']) for r in rows]
        if any(not np.array_equal(initial[0],x) for x in initial[1:]):
            raise ValueError('Mixed initial states in a primary group')
        out.append(dict(source=source,config_version=version,routes=routes,
                        paired_attempt_set_complete=set(routes)=={'E','D','A'},
                        successful_routes=[r['route'] for r in rows if r['checker_success']],
                        failure_routes=[r['route'] for r in rows if not r['checker_success']],
                        autonomous_executor_labels=False,formal_train_ready=False))
    return out


def summarize_sweeps(parts, total):
    parts=sorted(parts,key=lambda p:p['begin_interval'])
    cursor=0
    for part in parts:
        if part['begin_interval']!=cursor or part['end_interval']<=cursor:
            raise ValueError('Sweep gap/overlap')
        if part['total_native_intervals']!=total or part['required_clearance_m']!=.0005:
            raise ValueError('Sweep scope/threshold differs')
        if part['contact_rule_version']!='DR-v0.4-R3-exact-finger-pad':
            raise ValueError('Contact rule differs')
        if part['valid'] and part['checked_native_intervals']!=part['end_interval']:
            raise ValueError('Truncated successful sweep')
        cursor=part['end_interval']
    if cursor!=total:raise ValueError('Incomplete sweep coverage')
    valid=all(p['valid'] for p in parts)
    return dict(valid=valid,planned_intervals=total,
                lower_bound_m=min(p['lower_bound_m'] for p in parts) if valid else None,
                failures=[p for p in parts if not p['valid']],
                scope='Conservative native-state interpolation; failure partitions stop at first witness')


def merge_audit_caches(paths):
    """Combine disjoint queues; equal inherited entries deduplicate, conflicts hold."""
    merged={}
    for path in paths:
        for entry in read_json(path)['entries']:
            attempt=entry['attempt']
            if attempt in merged and merged[attempt] != entry:
                raise ValueError('Conflicting audit cache entry; retain both and inspect: '+attempt)
            if entry['input_signature'] != signature(attempt):
                raise ValueError('Audit cache input changed: '+attempt)
            checked=read_json(entry['integrity']);qualified=read_json(entry['qualification'])
            for receipt in [checked,qualified]:
                if receipt.get('attempt') != attempt or receipt.get('input_signature') != entry['input_signature']:
                    raise ValueError('Audit cache receipt binding differs: '+attempt)
            if not checked.get('integrity_pass'):
                raise ValueError('Audit cache integrity did not pass: '+attempt)
            merged[attempt]=entry
    return dict(at=stamp(),entries=list(merged.values()))
