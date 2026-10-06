"""Gripper goal projection for rigid finger gaps the arm QP cannot change."""
import mujoco
import numpy as np


def protect_grip(ref,policy_command,margin=.001):
    m,d=ref.model_data();gripper=ref.robot.gripper['right']
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
    first=[i for i,n in enumerate(names) if 'gripper0_right_finger1' in n and (m.geom_contype[i] or m.geom_conaffinity[i])]
    second=[i for i,n in enumerate(names) if 'gripper0_right_finger2' in n and (m.geom_contype[i] or m.geom_conaffinity[i])]
    distances=[]
    for i in first:
        for j in second:
            if 'pad_collision' in names[i] and 'pad_collision' in names[j]:continue
            dist=float(mujoco.mj_geomDistance(m,d,i,j,.02,np.zeros(6)))
            distances.append((dist,i,j))
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
