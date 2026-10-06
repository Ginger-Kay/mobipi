"""Outcome-blind geometric docks and native nominal-intent whole-body QP.

No human/reference manipulation points enter this module. It reuses existing
bounded stow IK, lattice A*, signed distances, swept checks and velocity QP.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from mobiwam.planner_min import occupancy_lattice_astar,densify_path,rank_primary,velocity_level_qp
from mobiwam.reference_collision import SweptGeometry
from mobiwam.reference_ik import constrained_pose_ik,distance_rows
from reference_stow import build_stow
from reference_executor import mapped_action

VERSION='pi05-geometric-motion-v1'
DOCK_OFFSETS=np.array([[.075,0.],[-.075,0.],[0.,.075],[0.,-.075],[.075,.075]])


def arm_indices(ref):
    m,_=ref.model_data();arm=ref.robot.part_controllers['right']
    qids=np.asarray(arm.qpos_index);dofs=np.asarray(arm.qvel_index)
    joints=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in qids]
    return qids,dofs,m.jnt_range[joints].copy()


def docks(ref):
    m,live=ref.model_data();q0=live.qpos.copy();base=ref.robot.part_controllers['base'];bids=np.asarray(base.qpos_index)
    qids,dofs,limits=arm_indices(ref);site=ref.robot.eef_site_id['right']
    target=dict(pos=live.site_xpos[site].copy(),rot=live.site_xmat[site].reshape(3,3).copy())
    stow=build_stow(ref);stowed=q0.copy();stowed[qids]=stow['arm_qpos']
    fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
    check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005)
    stow_sweep=check.path([q0,stowed],['manipulate'])
    candidates=[]
    for i,offset in enumerate(DOCK_OFFSETS):
        goal=q0[bids].copy();goal[:2]+=offset
        candidate=dict(candidate_id=f'dock-{i}',base_goal=goal.tolist(),hard_valid=False,reason='not_completed')
        try:
            if not stow_sweep['valid']:raise ValueError('stow swept path hard-invalid')
            def is_free(xy):
                q=stowed.copy();q[bids[:2]]=xy
                pairs,dist=check.distances(q,'precontact')
                return bool(np.min(dist,initial=.10)>=.0005)
            start=q0[bids[:2]];bounds=[start[0]-.20,start[0]+.20,start[1]-.20,start[1]+.20]
            path=occupancy_lattice_astar(start,goal[:2],is_free,bounds_xy=bounds,resolution_m=.025,maximum_expansions=2000)
            path=densify_path(path,.01);goal[:2]=path[-1]
            states=np.repeat(stowed[None],len(path),axis=0);states[:,bids[:2]]=path
            navigation=check.path(states,['precontact']*(len(states)-1))
            scratch=mujoco.MjData(m);scratch.qpos[:]=states[-1];mujoco.mj_forward(m,scratch)
            q,pe,re,ik=constrained_pose_ik(m,scratch,site,qids,dofs,target,stowed[qids],limits,check,phase='manipulate')
            reach_sweep=check.path([states[-1],scratch.qpos.copy()],['manipulate'])
            valid=bool(navigation['valid'] and reach_sweep['valid'] and pe<.01 and re<.10)
            length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
            margin=float(np.min(np.minimum(q-limits[:,0],limits[:,1]-q)))
            candidate.update(hard_valid=valid,reason='passed_geometric_prefix_and_endpoint_reach' if valid else 'continuous_or_reach_failure',
                base_goal=goal.tolist(),base_path=np.c_[path,np.full(len(path),goal[2])].tolist(),
                minimum_continuous_clearance_m=min(float(stow_sweep.get('lower_bound_m',0)),float(navigation.get('lower_bound_m',0)),float(reach_sweep.get('lower_bound_m',0))),
                minimum_manipulability_or_joint_margin=margin,minimum_policy_view_compatibility=1.,
                policy_view_note='same three existing cameras; frustum/occlusion compatibility not certified here',
                total_planned_base_path_m=length,total_planned_time_s=length/(.015 if ref.args.task=='CloseDrawer' else .09),
                endpoint_ik=ik,stow_sweep=stow_sweep,navigation_sweep=navigation,reach_sweep=reach_sweep)
        except Exception as exc:candidate['reason']=str(exc)
        candidates.append(candidate)
    primary=None
    if any(x['hard_valid'] for x in candidates):primary=rank_primary(candidates)
    assert np.array_equal(q0,live.qpos)
    return dict(version=VERSION,environment_step_calls=0,live_qpos_unchanged=True,proposal_count=5,candidates=candidates,primary=primary,stow=stow,
        provenance='existing native primitives and fixture geometry; no reference manipulation trajectory; geometry only, actual runtime still needs qualification')


def prefix_action(ref,plan,phase,index=0):
    m,d=ref.model_data();body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ref.base_body)
    R=d.xmat[body].reshape(3,3);point=dict(pos=d.xpos[body]+R@np.asarray(plan['stow']['local_pos']),rot=R@np.asarray(plan['stow']['local_rot']),grasp=-1.)
    base=ref.robot.part_controllers['base'];goal=d.qpos[base.qpos_index].copy() if phase=='stow' else np.asarray(plan['primary']['base_path'][index])
    action,pe,re,be=mapped_action(ref,point,goal,arm_enabled=True)
    return action,dict(phase=phase,geometric_stow_position_error_m=pe,geometric_stow_rotation_error_rad=re,base_error_generalized=be,base_goal=goal.tolist())


def whole_body_action(ref,intent,base_goal,previous_velocity=None):
    m,d=ref.model_data();arm=ref.robot.part_controllers['right'];base=ref.robot.part_controllers['base'];qids,ad,limits=arm_indices(ref)
    bd=np.asarray(base.qvel_index);dofs=np.r_[bd,ad];site=ref.robot.eef_site_id['right'];dt=.05
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(m,d,jp,jr,site);J=np.vstack([jp[:,dofs],jr[:,dofs]])
    ep=np.asarray(intent['pos'])-d.site_xpos[site]
    er=Rotation.from_matrix(np.asarray(intent['rot'])@d.site_xmat[site].reshape(3,3).T).as_rotvec()
    task=ref.args.task;speed=.015 if task=='CloseDrawer' else .09
    desired_base=np.clip((np.asarray(base_goal)-d.qpos[base.qpos_index])*1.5,-speed,speed)
    twist=np.r_[np.clip(ep*4,-.1,.1),np.clip(er*4,-.1,.1)]
    # The fixed base goal is planner output; only the frozen policy supplies EEF
    # intent. This is one convex tracking QP through the existing primitive.
    augmented=np.vstack([J,np.c_[np.eye(3)*.1,np.zeros((3,len(ad)))]])
    target=np.r_[twist,desired_base*.1]
    lower=np.r_[np.full(3,-speed),np.maximum(-1.,(limits[:,0]+.01501-d.qpos[qids])/dt)]
    upper=np.r_[np.full(3,speed),np.minimum(1.,(limits[:,1]-.01501-d.qpos[qids])/dt)]
    if previous_velocity is not None:
        acceleration=np.r_[np.full(3,.2),np.full(len(ad),2.)]
        lower=np.maximum(lower,previous_velocity-acceleration*dt);upper=np.minimum(upper,previous_velocity+acceleration*dt)
    fixture=ref.env.drawer if task=='CloseDrawer' else ref.env.door_fxtr
    check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005)
    rows,distances=distance_rows(check,d.qpos.copy(),'manipulate',dofs)
    inequalities=(rows,(.001-distances)/dt) if len(rows) else None
    velocity,receipt=velocity_level_qp(augmented,target,lower,upper,base_weight=1.,damping=.001,inequalities=inequalities)
    if not receipt['feasible']:raise ValueError('whole-body QP constraints infeasible')
    predicted=d.qpos.copy();predicted[np.r_[base.qpos_index,qids]]+=velocity*dt
    swept=check.path([d.qpos.copy(),predicted],['manipulate'])
    if not swept['valid']:raise ValueError('whole-body predicted swept clearance failure')
    # Express the arm part of QP motion through the unchanged native OSC.
    arm_twist=J[:,3:]@velocity[3:]
    point=dict(pos=d.site_xpos[site]+arm_twist[:3]*dt,rot=Rotation.from_rotvec(arm_twist[3:]*dt).as_matrix()@d.site_xmat[site].reshape(3,3),grasp=intent['grasp'])
    action,pe,re,be=mapped_action(ref,point,d.qpos[base.qpos_index]+velocity[:3]/1.5,arm_enabled=True)
    receipt=dict(receipt,velocity=velocity.tolist(),predicted_sweep=swept,nominal_eef_pos=np.asarray(intent['pos']).tolist(),qp_eef_pos=point['pos'].tolist(),base_goal=np.asarray(base_goal).tolist())
    return action,receipt,velocity
