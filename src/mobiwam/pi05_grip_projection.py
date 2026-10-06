"""Gripper goal projection for rigid finger gaps the arm QP cannot change."""
import mujoco
import numpy as np
from mobiwam.reference_collision import SweptGeometry


def protect_grip(ref,policy_command,margin=.001):
    m,d=ref.model_data();gripper=ref.robot.gripper['right']
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
    first=[i for i,n in enumerate(names) if 'gripper0_right_finger1' in n and (m.geom_contype[i] or m.geom_conaffinity[i])]
    second=[i for i,n in enumerate(names) if 'gripper0_right_finger2' in n and (m.geom_contype[i] or m.geom_conaffinity[i])]
    fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
    if getattr(ref,'pi05_qp_geometry_model',None)!=id(m):
        ref.pi05_qp_geometry=SweptGeometry(m,target_prefix=fixture.name,margin=.0005);ref.pi05_qp_geometry_model=id(m)
    # Reuse the established pair set and configured native GJK precision. The
    # unconfigured raw mj_geomDistance can disagree for these convex meshes.
    pairs,values=ref.pi05_qp_geometry.distances(d.qpos.copy(),'manipulate')
    first=set(first);second=set(second);distances=[]
    for (i,j),dist in zip(pairs,values):
        if (int(i) in first and int(j) in second) or (int(j) in first and int(i) in second):distances.append((float(dist),int(i),int(j)))
    minimum=min(distances,default=(.02,-1,-1));active=policy_command>0 and minimum[0]<margin+.0005
    if active:
        # Native Panda controller consumes normalized position targets from
        # current_action. Change its goal, never the physical finger qpos.
        joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in gripper.joints]
        goals=[];normalized=[]
        for joint in joints:
            q=float(d.qpos[m.jnt_qposadr[joint]]);lo,hi=m.jnt_range[joint]
            direction=np.sign(hi if abs(hi)>abs(lo) else lo)
            target=float(np.clip(q+direction*.0005,lo,hi));goals.append(target)
            normalized.append(float(2*(target-lo)/(hi-lo)-1))
        gripper.current_action=np.asarray(normalized)
        ref.pi05_grip_cap=np.asarray(normalized)
        return 0.,True,dict(minimum_rigid_finger_gap_m=minimum[0],pair=[names[minimum[1]],names[minimum[2]]],target_qpos=goals,
            normalized_position_goal=normalized,planning_margin_m=margin,actual_safety_margin_unchanged_m=.0005,hold_arm_until_physical_gap_recovers=True)
    if policy_command>0 and getattr(ref,'pi05_grip_cap',None) is not None:
        gripper.current_action=ref.pi05_grip_cap.copy()
        return 0.,False,dict(minimum_rigid_finger_gap_m=minimum[0],normalized_position_goal=gripper.current_action.tolist(),closure_limited=True)
    if policy_command<0:ref.pi05_grip_cap=None
    return policy_command,False,dict(minimum_rigid_finger_gap_m=minimum[0],closure_limited=False)


def native_coupled_closure(ref,policy_command):
    """Retain native closure goals and force error, with a rigid-self floor."""
    m,d=ref.model_data();gripper=ref.robot.gripper['right'];physical=d.qpos.copy()
    joints=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in gripper.joints])
    ids=m.jnt_qposadr[joints];ranges=m.jnt_range[joints]
    directions=np.sign(np.where(abs(ranges[:,1])>abs(ranges[:,0]),ranges[:,1],ranges[:,0]))
    if getattr(ref,'pi05_native_grip_floor_model',None)!=id(m):
        fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
        check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005)
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
        pairs,_=check.distances(d.qpos.copy(),'manipulate')
        indices=[k for k,(i,j) in enumerate(pairs) if
            (names[i].startswith('gripper0_right_finger1') and names[j].startswith('gripper0_right_finger2')) or
            (names[j].startswith('gripper0_right_finger1') and names[i].startswith('gripper0_right_finger2'))]
        if not indices:raise ValueError('native grip floor has no bound rigid finger pairs')
        def gap(aperture):
            q=physical.copy();q[ids]=directions*aperture
            _,distances=check.distances(q,'manipulate')
            return float(np.min(distances[indices]))
        lo=0.;hi=float(min(np.maximum(directions*ranges[:,0],directions*ranges[:,1])))
        if gap(hi)<.0011:raise ValueError('native rigid finger aperture has no safe closure floor')
        for _ in range(32):
            mid=(lo+hi)/2
            if gap(mid)>=.0011:hi=mid
            else:lo=mid
        ref.pi05_native_grip_floor=hi;ref.pi05_native_grip_floor_model=id(m)
        ref.pi05_native_grip_floor_proof=dict(aperture_floor_m=hi,minimum_rigid_self_gap_m=gap(hi),pair_count=len(indices),
            inherited_solver_buffer_m=.001,extra_tracking_buffer_m=.0001,actual_safety_margin_m=.0005)
    # Apply the original scalar Panda formatter exactly once. env.step receives
    # zero, preserving this goal, rather than losing the squeeze error by
    # replacing the goal with physical qpos plus a small receding increment.
    previous=gripper.current_action.copy()
    native=np.asarray(gripper.format_action(np.array([policy_command]))).copy()
    desired=ranges[:,0]+(native+1)/2*(ranges[:,1]-ranges[:,0])
    aperture=float(np.mean(directions*desired))
    aperture=float(np.clip(aperture,ref.pi05_native_grip_floor,.04))
    targets=directions*aperture
    normalized=2*(targets-ranges[:,0])/(ranges[:,1]-ranges[:,0])-1
    gripper.current_action=normalized
    assert np.array_equal(physical,d.qpos) and abs(targets.sum())<1e-12
    return dict(ref.pi05_native_grip_floor_proof,policy_command=float(policy_command),previous_native_goal=previous.tolist(),
        original_formatter_goal=native.tolist(),target_qpos=targets.tolist(),normalized_position_goal=normalized.tolist(),
        native_formatter_applied_once=True,native_closure_force_error_retained=True,physical_qpos_exactly_unchanged=True)
