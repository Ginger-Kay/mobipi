"""Measured 21-field features of the outcome-blind geometric prefixes."""
import mujoco
import numpy as np
from mobiwam.pi05_motion import arm_indices,frustum_compatibility
from mobiwam.reference_collision import SweptGeometry
from mobiwam.reference_ik import constrained_pose_ik
from mobiwam.scene004 import CANDIDATE_FEATURE_FIELDS,candidate_feature_vector

def feature_record(ref,route,plan,slot):
    m,live=ref.model_data();q0=live.qpos.copy();base=ref.robot.part_controllers['base'];bids=np.asarray(base.qpos_index)
    qids,ad,limits=arm_indices(ref);site=ref.robot.eef_site_id['right'];target=dict(pos=live.site_xpos[site].copy(),rot=live.site_xmat[site].reshape(3,3).copy())
    scratch=mujoco.MjData(m);states=[q0.copy()];fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
    check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005);tracking=[];solver=[]
    primary=plan['primary'] if plan else None
    if route!='E' and primary is None:return None
    if route=='D':
        stowed=q0.copy();stowed[qids]=plan['stow']['arm_qpos'];states.append(stowed)
        for p in primary['base_path'][1:]:
            q=stowed.copy();q[bids]=p;states.append(q)
        endpoint=states[-1].copy();endpoint[qids]=primary['endpoint_arm_qpos'];states.append(endpoint)
        tracking.append(primary['endpoint_ik']['position_error_m'])
        solver.append(primary['endpoint_ik']['position_error_m'])
    elif route=='A':
        for p in primary['base_path'][1:]:
            scratch.qpos[:]=states[-1];scratch.qpos[bids]=p;mujoco.mj_forward(m,scratch)
            q,pe,re,ik=constrained_pose_ik(m,scratch,site,qids,ad,target,scratch.qpos[qids].copy(),limits,check,phase='manipulate')
            states.append(scratch.qpos.copy());tracking.append(pe);solver.append(pe)
    jacp=np.zeros((3,m.nv));jacr=jacp.copy();manip=[];joint=[];view=[];eef=[]
    for q in states:
        scratch.qpos[:]=q;mujoco.mj_forward(m,scratch);mujoco.mj_jacSite(m,scratch,jacp,jacr,site)
        dofs=np.r_[base.qvel_index,ad] if route=='A' else ad
        manip.append(float(np.linalg.svd(np.vstack([jacp[:,dofs],.2*jacr[:,dofs]]),compute_uv=False)[-1]))
        joint.append(float(np.min(np.minimum(q[qids]-limits[:,0],limits[:,1]-q[qids]))))
        view.append(frustum_compatibility(m,scratch,target['pos'],ref.policy_cameras));eef.append(scratch.site_xpos[site].copy())
    speed=.015 if ref.args.task=='CloseDrawer' else .09
    velocity_caps=np.r_[np.full(3,speed),np.ones(7)];acceleration_caps=np.r_[np.full(3,.2),np.full(7,2.)]
    duration=0.;v_margin=float(velocity_caps.min());a_margin=float(acceleration_caps.min())
    for before,after in zip(states[:-1],states[1:]):
        delta=abs((after-before)[np.r_[bids,qids]])
        # A cubic zero-velocity endpoint schedule is a cost proxy for this
        # geometric prefix; runtime uses feedback and is verified separately.
        t=max(.05,float(np.max(1.5*delta/velocity_caps)),float(np.max(np.sqrt(6*delta/acceleration_caps))))
        duration+=t;v_margin=min(v_margin,float(np.min(velocity_caps-1.5*delta/t)))
        a_margin=min(a_margin,float(np.min(acceleration_caps-6*delta/t**2)))
    _,initial_distances=check.distances(q0,'manipulate')
    clearance=float(np.min(initial_distances,initial=.10)) if route=='E' else primary['minimum_continuous_clearance_m']
    base_path=float(np.linalg.norm(np.diff(np.asarray(states)[:,bids[:2]],axis=0),axis=1).sum())
    derived=dict(route_E=float(route=='E'),route_D=float(route=='D'),route_A=float(route=='A'),
        task_CloseDrawer=float(ref.args.task=='CloseDrawer'),task_CloseSingleDoor=float(ref.args.task=='CloseSingleDoor'),stage_precontact=1.,
        hard_valid=float(clearance>=.0005 and min(joint)>.015),minimum_continuous_clearance_m=clearance,
        minimum_manipulability=min(manip),minimum_joint_margin_rad=min(joint),minimum_policy_view_compatibility=min(view),
        total_planned_base_path_m=base_path,planned_time_normalized=duration/120,
        planned_base_net_m=float(np.linalg.norm((states[-1]-states[0])[bids[:2]])),
        planned_eef_path_m=float(np.linalg.norm(np.diff(eef,axis=0),axis=1).sum()),
        maximum_eef_tracking_error_m=max(tracking,default=0.),minimum_velocity_margin=v_margin,minimum_acceleration_margin=a_margin,
        solver_residual=max(solver,default=0.),slot_index_normalized=int(primary['candidate_id'].rsplit('-',1)[-1])/4. if primary else 0.,simulator_oracle_pre_outcome=1.)
    assert set(derived)==set(CANDIDATE_FEATURE_FIELDS);candidate_feature_vector(derived)
    assert np.array_equal(q0,live.qpos)
    return dict(route_family=route,candidate_id=route+'-primary',stage_eligible=True,hard_valid=bool(derived['hard_valid']),
        minimum_continuous_clearance_m=clearance,minimum_manipulability_or_joint_margin=min(joint),
        minimum_policy_view_compatibility=min(view),total_planned_base_path_m=base_path,total_planned_time_s=duration,derived=derived,
        definition='features cover the initial geometry and proposed mobile/stow/reach prefix; unknown future pi05 manipulation is not certified or filled in',
        timing='cubic geometric cost proxy using unchanged caps, not an executed-velocity certificate',
        field_notes=dict(minimum_manipulability='smallest singular value of native EEF Jacobian; rotational rows scaled0.2',
            minimum_policy_view_compatibility='fraction of existing camera frusta containing initial EEF, without occlusion prediction',
            slot_index_normalized='frozen geometric proposal index0..4 divided by4; no Source id or split encoding'))
