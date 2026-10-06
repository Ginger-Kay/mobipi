"""Termination metrics include the recorded partial native control-step tail."""
import json
from pathlib import Path
import numpy as np

def native_metrics(attempt,task,z):
    attempt=Path(attempt);binding=json.loads((attempt.parents[1]/'target-binding.json').read_text())
    assert binding['task']==task and len(binding['joints'])==1
    joint=binding['joints'][0];assert joint['name'].endswith('_slidejoint' if task=='CloseDrawer' else '_microjoint')
    assert task in ('CloseDrawer','CloseSingleDoor')
    address=joint['qpos_address'];initial=float(z['qpos'][0,address]);terminal=float(z['qpos'][-1,address])
    # Both native fixtures normalize -q/(positive span), with closed q=0.
    # The span cancels in relative progress; no checker threshold is changed.
    assert initial < -1e-6
    return dict(progress=float(np.clip(1-terminal/initial,0,1)),
        path_m=float(np.linalg.norm(np.diff(z['qpos'][:,:2],axis=0),axis=1).sum()),
        duration_s=float(z['sim_time'][-1]-z['sim_time'][0]))
