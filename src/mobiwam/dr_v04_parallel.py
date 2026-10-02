"""Outcome-blind FIFO allocation and immutable existing-route consumption."""
import json
from pathlib import Path


def group_output(run,index,group):
    batch=0 if index<=4 else 1+(index-5)//6
    return Path(run)/'batches'/f'{batch:02d}'/f'{index:02d}-{group}'


def completed_groups(run,order,rows,run_id):
    from mobiwam.reference_route_resume import completed_prefix
    done=[];missing=False
    for index,group in enumerate(order,1):
        out=group_output(run,index,group);ledger=Path(run)/'batches'/f'group-{index:02d}-dispatch.json'
        if not out.exists() and not ledger.exists():missing=True;continue
        if missing:raise ValueError('existing group schedule has gap')
        if not (out/'completed.json').exists():raise ValueError('partial group must finish before parallel transition')
        saved=json.loads((out/'completed.json').read_text())
        prefix=completed_prefix(out,rows[group]['route_order'],run_id,group)
        if len(prefix)!=3 or saved['route_outcomes']!=3 or saved['attempts']!=prefix:raise ValueError('completed group receipt differs')
        done.append(group)
    return done


def choose_gpu(free,task,counts):
    if not free:raise ValueError('no free GPU')
    # Read no task outcomes. Avoid assigning one task exclusively to one GPU.
    return min(free,key=lambda gpu:(counts.get((gpu,task),0),sum(v for (g,t),v in counts.items() if g==gpu),gpu))


def batches(order):
    return [order[:4]]+[order[i:i+6] for i in range(4,len(order),6)]


def continuation_batches(order,done):
    if list(done)!=list(order[:len(done)]):raise ValueError('completed prefix order differs')
    remaining=order[len(done):]
    return [remaining[i:i+6] for i in range(0,len(remaining),6)]
