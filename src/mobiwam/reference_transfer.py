"""Compile a human reference on a new Source without copying its qpos array.

The old trajectory is a frozen control template. New route outcomes are never
read. Every transferred waypoint still needs full geometry and controller
preflight before a route may execute.
"""
import json
import hashlib
from pathlib import Path

import h5py
import mujoco
import numpy as np


def _body(model, name):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        raise ValueError(f'missing reference transfer body: {name}')
    return body


def _joint(model, binding, suffix):
    matches = [j for j in binding['joints'] if j['name'].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError('ambiguous target joint')
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, matches[0]['name'])
    if joint < 0 or int(model.jnt_qposadr[joint]) != matches[0]['qpos_address']:
        raise ValueError('target joint binding changed')
    return joint, int(model.jnt_qposadr[joint])


def _pose(data, body):
    return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()


def _valid_rotation(matrix):
    return (np.isfinite(matrix).all() and
            np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8, rtol=0) and
            np.isclose(np.linalg.det(matrix), 1., atol=1e-8))


def _yaw(rotation):
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _wrap(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _map_drawer_joint_by_closed_progress(old_q, old_initial, new_initial,
                                         old_range, new_range):
    """Map remaining distance to each drawer model's closed hard stop.

    The upper slide-joint limit is the physical closed endpoint. The initial
    joint pose and closed endpoint both map exactly; out-of-range values fail.
    """
    old_closed, new_closed = float(old_range[1]), float(new_range[1])
    old_span = old_closed - old_initial
    new_span = new_closed - new_initial
    if (not np.isfinite([old_q, old_initial, new_initial, old_closed,
                         new_closed]).all() or old_span <= 1e-9 or
            new_span <= 1e-9 or abs(old_closed) > 1e-9 or
            abs(new_closed) > 1e-9 or
            not old_range[0] <= old_q <= old_closed or
            not new_range[0] <= new_initial <= new_closed):
        raise ValueError('invalid drawer joint progress or source opening')
    progress = (old_q - old_initial) / old_span
    mapped = new_initial + progress * new_span
    if not new_range[0] <= mapped <= new_closed:
        raise ValueError('mapped drawer joint leaves new limits')
    return float(mapped)


def _drawer_opening_from_joint(qpos, slide_extent):
    """Use the RoboCasa Drawer.get_door_state normalization for a target point."""
    if not np.isfinite([qpos, slide_extent]).all() or slide_extent <= 0:
        raise ValueError('invalid drawer joint or opening extent')
    opening = -qpos / slide_extent
    if not 0 <= opening <= 1:
        raise ValueError('mapped drawer opening leaves checker range')
    return float(opening)


def compile_transferred_path(ref, attempt):
    """Return points and a pre-outcome transfer receipt for one fixed template."""
    from reference_geometry import adjust_reference
    from reference_executor import orientation_error

    attempt = Path(attempt).resolve()
    old_source = attempt.parent.parent
    old_cfg = json.loads((old_source.parent / 'env_config.json').read_text())
    new_cfg = json.loads((ref.source.parent / 'env_config.json').read_text())
    for key in ('env_name', 'layout_ids', 'style_ids', 'robots', 'control_freq'):
        if old_cfg[key] != new_cfg[key]:
            raise ValueError(f'reference/new Source {key} differs')
    if old_source.resolve() == ref.source.resolve():
        raise ValueError('transfer requires a new Source')
    old_binding = json.loads((old_source / 'target-binding.json').read_text())
    new_binding = json.loads((ref.root / 'target-binding.json').read_text())
    if (old_binding['task'] != new_binding['task'] or
            old_binding['fixture_class'] != new_binding['fixture_class']):
        raise ValueError('reference/new target class differs')
    suffix = '_slidejoint' if ref.args.task == 'CloseDrawer' else '_microjoint'
    from mobiwam.task_video_identity import source_model
    old_model = source_model(str(old_source / 'model.xml'), hashlib.sha256((old_source / 'model.xml').read_bytes()).hexdigest())
    old = mujoco.MjData(old_model)
    new_model, live = ref.model_data()
    new = mujoco.MjData(new_model)
    old_state = np.load(old_source / 'integration.npy', allow_pickle=False)
    mujoco.mj_setState(old_model, old, old_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(old_model, old)
    new.qpos[:] = live.qpos
    new.mocap_pos[:] = live.mocap_pos
    new.mocap_quat[:] = live.mocap_quat
    mujoco.mj_forward(new_model, new)
    old_joint, old_address = _joint(old_model, old_binding, suffix)
    new_joint, new_address = _joint(new_model, new_binding, suffix)
    old_range = old_model.jnt_range[old_joint]
    new_range = new_model.jnt_range[new_joint]
    if (not old_model.jnt_limited[old_joint] or not new_model.jnt_limited[new_joint] or
            not np.isfinite(old_range).all() or not np.isfinite(new_range).all() or
            old_range[1] <= old_range[0] or new_range[1] <= new_range[0]):
        raise ValueError('invalid target joint ranges')
    old_initial = float(old.qpos[old_address])
    new_initial = float(new.qpos[new_address])
    old_handle_name = old_binding['fixture_name'] + ('_door_handle_main' if suffix == '_slidejoint' else '_door')
    new_handle_name = new_binding['fixture_name'] + ('_door_handle_main' if suffix == '_slidejoint' else '_door')
    old_handle = _body(old_model, old_handle_name)
    new_handle = _body(new_model, new_handle_name)
    old_main = _body(old_model, old_binding['fixture_name'] + '_main')
    new_main = _body(new_model, new_binding['fixture_name'] + '_main')
    old_main_pos, old_main_rot = _pose(old, old_main)
    new_main_pos, new_main_rot = _pose(new, new_main)
    main_rotation = new_main_rot @ old_main_rot.T
    if not _valid_rotation(main_rotation) or np.max(abs(main_rotation[2, :2])) > 1e-6:
        raise ValueError('non-planar target frame transfer')
    old_site_name = mujoco.mj_id2name(new_model, mujoco.mjtObj.mjOBJ_SITE,
                                      ref.robot.eef_site_id['right'])
    old_site = mujoco.mj_name2id(old_model, mujoco.mjtObj.mjOBJ_SITE, old_site_name)
    if old_site < 0:
        raise ValueError('reference gripper site missing')
    old_base_ids = []
    new_base_ids = []
    for address in ref.robot.part_controllers['base'].qpos_index:
        joint_id = int(np.flatnonzero(new_model.jnt_qposadr == address)[0])
        name = mujoco.mj_id2name(new_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        old_joint_id = mujoco.mj_name2id(old_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if old_joint_id < 0:
            raise ValueError('reference base joint missing')
        old_base_ids.append(int(old_model.jnt_qposadr[old_joint_id]))
        new_base_ids.append(int(address))
    old_arm_ids = []
    new_arm_ids = []
    for address in ref.robot.part_controllers['right'].qpos_index:
        joint_id = int(np.flatnonzero(new_model.jnt_qposadr == address)[0])
        name = mujoco.mj_id2name(new_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        old_joint_id = mujoco.mj_name2id(old_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if old_joint_id < 0:
            raise ValueError('reference arm joint missing')
        old_arm_ids.append(int(old_model.jnt_qposadr[old_joint_id]))
        new_arm_ids.append(int(address))
    records = [json.loads(line) for line in (attempt / 'trace.jsonl').read_text().splitlines()]
    first_success = next((i for i, record in enumerate(records) if record['after']['success']), None)
    if first_success is None:
        raise ValueError('selected human reference has no success endpoint')
    records = records[:first_success + 1]
    with h5py.File(attempt / 'demo.hdf5') as handle:
        grasp_states = np.asarray(handle['data/demo_0/actions'][:, 6])
    if len(grasp_states) < len(records):
        raise ValueError('reference action/trace mismatch')
    old_opening = float(old_binding['opening']['door'])
    new_opening = float(new_binding['opening']['door'])
    drawer_slide_extent = None
    if ref.args.task == 'CloseDrawer':
        drawer_slide_extent = float(ref.env.drawer.size[1] * .55)
        if abs(_drawer_opening_from_joint(new_initial, drawer_slide_extent) - new_opening) > 1e-6:
            raise ValueError('new drawer opening disagrees with native checker')
    old_base_initial = old.qpos[old_base_ids].copy()
    new_base_initial = new.qpos[new_base_ids].copy()
    old_base_body = _body(old_model, ref.base_body)
    new_base_body = _body(new_model, ref.base_body)
    new_base_world_initial, new_base_world_rotation = _pose(new, new_base_body)
    points = []
    max_base_offset = 0.
    max_target_displacement = 0.
    max_base_basis_condition = 0.
    joint_mapping = ('drawer_closed_progress_v0.2' if ref.args.task == 'CloseDrawer'
                     else 'range_scaled_displacement_v0.1')
    for i, record in enumerate(records):
        if i % 6 and i != len(records) - 1 and (i == 0 or grasp_states[i] == grasp_states[i-1]):
            continue
        state = record['after']
        old.qpos[:] = state['qpos']
        mujoco.mj_forward(old_model, old)
        old_q = float(old.qpos[old_address])
        if ref.args.task == 'CloseDrawer':
            new_q = _map_drawer_joint_by_closed_progress(
                old_q, old_initial, new_initial, old_range, new_range)
            target_q = new_q
        else:
            scale = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
            new_q = new_initial + (old_q - old_initial) * scale
            if new_q < new_range[0] - 1e-6 or new_q > new_range[1] + 1e-6:
                raise ValueError('mapped target joint leaves new limits')
            target_q = np.clip(new_q, *new_range)
        new.qpos[:] = live.qpos
        new.qpos[new_address] = target_q
        old_base_world_pos, old_base_world_rot = _pose(old, old_base_body)
        desired_base_pos = new_main_pos + main_rotation @ (old_base_world_pos - old_main_pos)
        desired_base_yaw = _yaw(old_base_world_rot) + _yaw(main_rotation)
        base = new_base_initial.copy()
        base[2] += _wrap(desired_base_yaw - _yaw(new_base_world_rotation))
        # The body origin can move when yaw changes. Evaluate the XY basis at
        # this waypoint's yaw, then invert only the two translational joints.
        new.qpos[:] = live.qpos
        new.qpos[new_address] = target_q
        new.qpos[new_base_ids[2]] = base[2]
        mujoco.mj_forward(new_model, new)
        base_at_yaw = new.xpos[new_base_body].copy()
        basis = np.empty((2, 2))
        for axis in range(2):
            new.qpos[new_base_ids[axis]] = new_base_initial[axis] + .01
            mujoco.mj_forward(new_model, new)
            basis[:, axis] = (new.xpos[new_base_body, :2] - base_at_yaw[:2]) / .01
            new.qpos[new_base_ids[axis]] = new_base_initial[axis]
        condition = float(np.linalg.cond(basis))
        if not np.isfinite(condition) or condition > 1e6:
            raise ValueError('new base world-coordinate basis is singular')
        max_base_basis_condition = max(max_base_basis_condition, condition)
        base[:2] += np.linalg.solve(basis, desired_base_pos[:2] - base_at_yaw[:2])
        new.qpos[:] = live.qpos
        new.qpos[new_address] = target_q
        new.qpos[new_base_ids] = base
        new.qpos[new_arm_ids] = old.qpos[old_arm_ids]
        mujoco.mj_forward(new_model, new)
        base_error = float(np.linalg.norm(new.xpos[new_base_body, :2] - desired_base_pos[:2]))
        if base_error > 1e-5:
            raise ValueError(f'new base forward geometry differs from derived mapping: {base_error}')
        old_handle_pos, old_handle_rot = _pose(old, old_handle)
        new_handle_pos, new_handle_rot = _pose(new, new_handle)
        rotation = new_handle_rot @ old_handle_rot.T
        if not _valid_rotation(rotation):
            raise ValueError('invalid moving handle frame transfer')
        old_site_pos = old.site_xpos[old_site]
        old_site_rot = old.site_xmat[old_site].reshape(3, 3)
        target_opening = (_drawer_opening_from_joint(new_q, drawer_slide_extent)
                          if ref.args.task == 'CloseDrawer' else
                          float(np.clip(new_opening + float(state['target']['door']) - old_opening, 0, 1)))
        point = dict(reference_step=i,
                     pos=new_handle_pos + rotation @ (old_site_pos - old_handle_pos),
                     rot=rotation @ old_site_rot,
                     base=base,
                     opening=target_opening,
                     grasp=float(grasp_states[i]), arm_qpos=old.qpos[old_arm_ids].copy(),
                     fixture_qpos_address=new_address, fixture_qpos=float(new_q))
        point = adjust_reference(ref, new, point)
        if (points and point['grasp'] == points[-1]['grasp'] and
                np.linalg.norm(point['pos']-points[-1]['pos']) < .006 and
                np.linalg.norm(point['base']-points[-1]['base']) < .006 and
                np.linalg.norm(orientation_error(point['rot'], points[-1]['rot'])) < .03 and
                (ref.args.task != 'CloseDrawer' or
                 (i != len(records) - 1 and
                  abs(point['opening'] - points[-1]['opening']) < .005))):
            continue
        max_base_offset = max(max_base_offset, float(np.linalg.norm(base[:2] - new_base_initial[:2])))
        max_target_displacement = max(max_target_displacement, float(np.linalg.norm(point['pos'] - old_site_pos)))
        points.append(point)
    if not points:
        raise ValueError('empty transferred reference path')
    receipt = {'old_source': str(old_source), 'new_source': str(ref.source),
               'old_fixture': old_binding['fixture_name'], 'new_fixture': new_binding['fixture_name'],
               'old_nq': old_model.nq, 'new_nq': new_model.nq,
               'old_joint_range': old_range.tolist(), 'new_joint_range': new_range.tolist(),
               'target_joint_mapping': joint_mapping,
               'target_opening_mapping': ('native_drawer_slide_normalization_v0.2'
                                          if ref.args.task == 'CloseDrawer' else
                                          'relative_opening_displacement_v0.1'),
               'old_initial_target_qpos': old_initial, 'new_initial_target_qpos': new_initial,
               'old_initial_base': old_base_initial.tolist(), 'new_initial_base': new_base_initial.tolist(),
               'max_base_basis_condition': max_base_basis_condition,
               'first_transferred_base': points[0]['base'].tolist(),
               'first_base_gap_m': float(np.linalg.norm(points[0]['base'][:2] - new_base_initial[:2])),
               'max_base_offset_m': max_base_offset,
               'max_target_displacement_m': max_target_displacement,
               'waypoints': len(points), 'human_first_success_step': first_success,
               'new_route_outcomes_read': 0, 'formal_train_ready': False}
    return points, receipt
