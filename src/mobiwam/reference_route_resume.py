"""Fail-closed consumption of a completed route prefix; never repeat an outcome."""
import json
import hashlib
from pathlib import Path


def completed_prefix(out, order, run_id, group_id):
    out=Path(out);results=[];missing=False
    for route in order:
        start=out/f'route-{route}-start.json';dispatch=out/f'route-{route}-dispatched.json';replay=out/f'route-{route}-replay.json'
        exists=[p.exists() for p in (start,dispatch,replay)]
        if not any(exists):
            if list(out.glob(f'source-*/{route}/attempt-*')):raise ValueError('unledgered attempt cannot resume')
            missing=True;continue
        if missing or not all(exists):raise ValueError('partial route or non-prefix ledger cannot resume')
        begun=json.loads(start.read_text())
        if any(begun.get(k)!=v for k,v in dict(route=route,run_id=run_id,group_id=group_id).items()):raise ValueError('resume route identity differs')
        item=json.loads(dispatch.read_text());attempt=Path(item['path']).resolve()
        if attempt.parent.name!=route or attempt.parent.parent.parent!=out.resolve():raise ValueError('resume attempt escaped group')
        outcome=json.loads((attempt/'result.json').read_text())
        saved=json.loads(replay.read_text());rp=Path(saved['result']).resolve()
        if rp.parent.parent!=attempt or hashlib.sha256(rp.read_bytes()).hexdigest()!=saved['sha256']:raise ValueError('resume replay binding differs')
        verification=json.loads(rp.read_text())
        if outcome['route']!=route or saved['reproducible'] is not True or verification['reproducible'] is not True:raise ValueError('resume requires verified completed prefix')
        if verification['steps']!=outcome['steps'] or verification['max_state_abs_error']>1e-5:raise ValueError('resume replay threshold differs')
        results.append(dict(route=route,path=str(attempt)))
    return results
