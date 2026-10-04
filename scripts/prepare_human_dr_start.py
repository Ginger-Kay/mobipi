"""Bounded static precontact start design; no physics or task rollouts.

Keep the original fixture and nonrobot state. A derived start remains in the
same source family and must be published separately as unfrozen practice.
"""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from mobiwam.reference_collision import SweptGeometry
from mobiwam.contact_rules import allowed_contact


def search(source, output):
    output.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(source / 'model.xml'))
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, np.load(source / 'integration.npy'), mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m, d)
    original = d.qpos.copy()
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'gripper0_right_grip_site')
    handle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'stack_4_main_group_3_door_handle_handle')
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'mobilebase0_base')
    assert min(site, handle, base) >= 0
    joint_names = ['robot0_joint' + str(i) for i in range(1, 8)]
    joints = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names])
    qids = m.jnt_qposadr[joints]
    limits = m.jnt_range[joints]
    h = d.geom_xpos[handle].copy()
    rot = d.geom_xmat[handle].reshape(3, 3).copy()
    outward = -rot[:, 1]
    tangent = rot[:, 2]
    tangent[2] = 0.
    tangent /= np.linalg.norm(tangent)
    target = h + .12 * outward
    base_origin = d.xpos[base].copy()
    yaw_origin = np.arctan2(d.xmat[base].reshape(3, 3)[1, 0], d.xmat[base].reshape(3, 3)[0, 0])
    translation = []
    for i in [0, 1]:
        d.qpos[:] = original
        d.qpos[i] += .01
        mujoco.mj_forward(m, d)
        translation.append((d.xpos[base][:2] - base_origin[:2]) / .01)
    translation = np.array(translation).T
    geom = SweptGeometry(m, target_prefix='stack_4_main_group_3')
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '' for i in range(m.ngeom)]
    rows = []
    valid = []
    # Thirty-six declared geometric candidates; no adaptive outcome retries.
    for index, (distance, lateral, flip, lift) in enumerate(itertools.product([.15, .25, .35], [.40, .50, .60], [1., -1.], [0., .10])):
        d.qpos[:] = original
        xy = (h + distance * outward + lateral * tangent)[:2]
        d.qpos[2] = np.arctan2(target[1]-xy[1], target[0]-xy[0]) - yaw_origin
        # The native yaw hinge has an offset pivot. Measure translation after
        # setting yaw instead of treating it as a rotation about base origin.
        mujoco.mj_forward(m, d)
        d.qpos[:2] += np.linalg.solve(translation, xy - d.xpos[base][:2])
        d.qpos[3] = lift  # versioned robot lift initial coordinate, within native range
        # Finger bodies rotate their slide axes by 90 degrees: closure is
        # grip-site X, not Y. Set closure vertical, across horizontal handle.
        approach = -outward
        closure = np.array([0., 0., flip])
        target_rot = np.column_stack([closure, np.cross(approach, closure), approach])
        assert np.linalg.det(target_rot) > .999
        def residual(q):
            d.qpos[qids] = q
            mujoco.mj_forward(m, d)
            pos = d.site_xpos[site] - target
            ori = Rotation.from_matrix(target_rot @ d.site_xmat[site].reshape(3, 3).T).as_rotvec()
            return np.r_[pos, .2*ori, .001*(q-original[qids])]
        fit = least_squares(residual, original[qids], bounds=(limits[:, 0]+.12, limits[:, 1]-.12), max_nfev=160)
        residual(fit.x)
        assert np.linalg.norm(d.xpos[base][:2]-xy)<1e-7, 'Native base coordinate mapping differs'
        pos_error = float(np.linalg.norm(d.site_xpos[site]-target))
        rot_error = float(Rotation.from_matrix(target_rot @ d.site_xmat[site].reshape(3,3).T).magnitude())
        margin = float(np.min(np.minimum(fit.x-limits[:,0],limits[:,1]-fit.x)))
        bad=[]
        for contact in d.contact[:d.ncon]:
            a,b=names[contact.geom1],names[contact.geom2]
            if not any(n.startswith(('robot0_','gripper0_','mobilebase0_')) for n in [a,b]):continue
            if not allowed_contact(a,b,'precontact','stack_4_main_group_3'):
                bad.append(dict(a=a,b=b,distance=float(contact.dist)))
        pairs, distances = geom.distances(d.qpos, 'precontact')
        min_index = int(np.argmin(distances))
        clearance = float(distances[min_index])
        row = dict(index=index,base_distance=distance,lateral=lateral,flip=flip,lift=lift,
                   position_error_m=pos_error,rotation_error_rad=rot_error,joint_margin_rad=margin,
                   minimum_clearance_m=clearance,closest_pair=[names[i] for i in pairs[min_index]],
                   forbidden_contacts=bad,base_position=d.xpos[base].tolist(),eef_position=d.site_xpos[site].tolist())
        row['accepted'] = pos_error<.002 and rot_error<.02 and margin>.12 and clearance>=.0005 and not bad
        rows.append(row)
        np.save(output/f'candidate-{index:02d}-qpos.npy',d.qpos.copy())
        if row['accepted']:valid.append(row)
        print(json.dumps(row),flush=True)
    # Prefer joint slack, then clearance, without inspecting an outcome.
    selected=max(valid,key=lambda x:(x['joint_margin_rad'],x['minimum_clearance_m'])) if valid else None
    result=dict(parent_source=str(source),kind='static_design_not_task_outcome',candidates=rows,selected=selected,
                handle_position=h.tolist(),drawer_outward=outward.tolist(),precontact_distance_m=.12,
                robot_lift_initial_m=selected['lift'] if selected else None,new_physics_steps=0,independent_source_increment=0,
                unchanged_nonrobot_qpos=True,required_clearance_m=.0005)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if selected is None:raise RuntimeError('No declared precontact candidate passed; preserve rejection evidence')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    search(args.source,args.output)
