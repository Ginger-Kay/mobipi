"""Retain replay states and locate numerical drift without changing dynamics."""
import numpy as np
import mujoco


def state_fields(model):
    fields = [dict(component='time', name='simulation_time', coordinate=0)]
    for component, addresses, size in (
            ('qpos', model.jnt_qposadr, model.nq),
            ('qvel', model.jnt_dofadr, model.nv)):
        for joint in range(model.njnt):
            start = int(addresses[joint])
            end = int(addresses[joint+1]) if joint+1 < model.njnt else size
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or f'joint_{joint}'
            fields.extend(dict(component=component, name=name, coordinate=k-start)
                          for k in range(start, end))
    fields.extend(dict(component='act', name='actuator_activation', coordinate=k)
                  for k in range(model.na))
    return fields


def summarize_drift(expected, actual, fields, threshold=1e-5):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if (expected.shape != actual.shape or expected.ndim != 2 or
            expected.shape[1] != len(fields) or not len(expected) or
            not np.isfinite(expected).all() or not np.isfinite(actual).all()):
        raise ValueError('invalid replay state alignment')
    errors = np.abs(actual-expected)
    result = []
    for column, field in enumerate(fields):
        values = errors[:, column]
        hits = np.flatnonzero(values > threshold)
        result.append(dict(field, state_column=column, max_abs_error=float(values.max()),
                           max_error_after_action=int(values.argmax())-1,
                           first_gt_threshold_after_action=int(hits[0])-1 if len(hits) else None))
    return dict(diagnostic_threshold=threshold, threshold_is_not_a_safety_gate=True,
                indexing='-1 is initial state; k is state after zero-based action k',
                units='native per field; do not interpret mixed-state maximum as metres',
                max_state_abs_error=float(errors.max()), fields=result)
