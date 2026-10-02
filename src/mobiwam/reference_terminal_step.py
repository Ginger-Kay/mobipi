"""Validate and replay a saved interrupted action without changing native guards."""
import json
from pathlib import Path
import numpy as np


def terminal_record(attempt, result, feedback):
    attempt=Path(attempt);n=result['steps'];partial=attempt/'partial-control-step.npz'
    if len(feedback)==n and not partial.exists():return None
    if result['reason'] not in ('native_forbidden_contact_stop','joint_margin_stop') or len(feedback)!=n+1:
        raise ValueError('unexpected feedback/action count or partial action')
    if [x['step'] for x in feedback]!=list(range(n+1)):
        raise ValueError('partial feedback indices differ')
    with np.load(partial,allow_pickle=False) as f:record={k:f[k].copy() for k in f.files}
    for key in ('initial_integration','terminal_integration','attempted_action'):
        if key not in record or not np.isfinite(record[key]).all():raise ValueError('invalid partial action payload')
    if record['initial_integration'].shape!=record['terminal_integration'].shape:raise ValueError('partial integration shapes differ')
    with np.load(attempt/'formal-native-substeps.npz',allow_pickle=False) as f:
        indices=f['step_index'];mask=indices==n
        if len(indices) and (indices.max()>n or np.any(np.diff(indices)<0)):raise ValueError('partial substep ordering differs')
        record['prefix_substeps']=int(np.sum(indices<n))
        record['tail_qpos']=f['qpos'][record['prefix_substeps']:].copy()
        record['tail_times']=f['sim_time'][record['prefix_substeps']:].copy()
        if int(record['completed_physics_substeps'])!=len(indices):raise ValueError('partial substep total differs')
    name='formal-substep-stop.json' if result['reason']=='native_forbidden_contact_stop' else 'joint-margin-stop.json'
    stop=json.loads((attempt/name).read_text());record['failure']=stop.get('failure',stop)
    if record['failure']['step']!=n or record['failure']['phase']!=feedback[-1]['stage']:raise ValueError('partial stop boundary differs')
    if result['reason']=='native_forbidden_contact_stop' and record['failure']['substep']!=int(record['completed_physics_substeps'])-1:raise ValueError('contact stop substep differs')
    record['phase']=feedback[-1]['stage'];record['step']=n
    return record


def replay_terminal(ref, record):
    from mobiwam.reference_formal_substep import FormalSubstepMonitor,FormalSafetyStop
    from mobiwam.reference_prefix_safety import JointMarginMonitor,JointMarginStop,GuardedIntegration
    import mujoco
    initial_error=float(np.max(np.abs(ref.integration()-record['initial_integration'])))
    if initial_error>1e-5:raise ValueError('replayed prefix integration differs at interrupted action')
    m,d=ref.model_data();arm=ref.robot.part_controllers['right']
    joints=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in arm.qpos_index]
    margin=JointMarginMonitor(arm.qpos_index,m.jnt_range[joints],[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in joints])
    fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
    native=FormalSubstepMonitor(ref,fixture.name);native.set_boundary(record['step'],record['phase'])
    guard=GuardedIntegration(ref.env.sim,d,margin,lite_physics=ref.env.lite_physics,step=record['step'],phase=record['phase'])
    failure=None
    try:
        with native:
            with guard:ref.env.step(record['attempted_action'])
    except (FormalSafetyStop,JointMarginStop) as exc:
        failure=dict(exc.failure)
        if isinstance(exc,FormalSafetyStop):failure['substep']+=record['prefix_substeps']
    if failure is None:raise ValueError('saved terminal safety stop did not recur')
    expected=record['failure']
    if failure!=expected:raise ValueError('native terminal stop receipt differs')
    actual=np.asarray(native.states)
    if actual.shape!=record['tail_qpos'].shape:raise ValueError('terminal native substep count differs')
    error=float(np.max(np.abs(ref.integration()-record['terminal_integration'])))
    qerror=float(np.max(np.abs(actual-record['tail_qpos'])))
    terror=float(np.max(np.abs(np.asarray(native.times)-record['tail_times'])))
    return dict(reproducible=max(initial_error,error,qerror,terror)<=1e-5,
        initial_integration_error=initial_error,terminal_integration_error=error,
        native_qpos_error=qerror,native_time_error=terror,partial_physics_substeps=len(native.phases),
        failure=failure,terminal_trace=ref.trace(),saved_partial_action_replayed=True,
        scope='completed action prefix followed by saved partial action under original unmodified native guards; no state injection')
