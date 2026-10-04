"""Versioned human D navigation constraints; no controller/physics changes."""
import numpy as np

VERSION='human-eda-v2-stowed'

def route_inputs(parts,route,docked):
    out={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in parts.items()}
    if route=='D' and not docked:
        out['right']=np.zeros_like(out['right'])
        out['right_gripper']=np.array([-1.])
    if route=='E' or (route=='D' and docked):
        out['base']=np.zeros_like(out['base']);out['base_mode']=-1.
    return out

def validate_dock(route,docked,base_qvel,arm_qpos,stow_qpos,closed,target_contact):
    if route!='D' or docked:raise ValueError('Dock once, on D only')
    v=np.asarray(base_qvel);q=np.asarray(arm_qpos);s=np.asarray(stow_qpos)
    if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('Invalid base velocity')
    if q.shape!=(7,) or s.shape!=(7,) or not np.isfinite(q).all() or not np.isfinite(s).all():
        raise ValueError('Invalid stow state')
    if np.any(abs(v)>[.01,.01,.02]):raise ValueError('Release base keys and wait until stopped before F5')
    if np.max(abs(q-s))>.05:raise ValueError('D must retain its saved stowed arm before docking')
    if closed or target_contact:raise ValueError('Dock before grasping or contacting the target')
    return True
