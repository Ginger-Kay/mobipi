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


class QPProtectionStop(RuntimeError):
    """Scientific protective stop from an infeasible policy intent."""
    pass


def frustum_compatibility(model,data,point,cameras):
    visible=[]
    for camera in cameras:
        cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,camera)
        local=data.cam_xmat[cid].reshape(3,3).T@(point-data.cam_xpos[cid])
        tangent=np.tan(np.deg2rad(float(model.cam_fovy[cid]))/2)
        visible.append(bool(local[2]<0 and abs(local[0])<=-local[2]*tangent and abs(local[1])<=-local[2]*tangent))
    return float(np.mean(visible))


def observed_articulation_coupling(ref,dofs):
    """Infer joint co-motion only from actual bilateral pad/handle contact."""
    m,d=ref.model_data();fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
    pads={};positions=[]
    for c in d.contact:
        for handle,pad in ((int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))):
            hn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,handle) or ''
            pn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,pad) or ''
            if hn.startswith(fixture.name) and 'handle' in hn and 'pad_collision' in pn:
                for finger in (1,2):
                    if f'finger{finger}' in pn:pads[finger]=pad;positions.append(c.pos.copy())
    if set(pads)!={1,2}:return None
    joints=[j for j in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or '').startswith(fixture.name) and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE))]
    if len(joints)!=1:return None
    joint=joints[0];fd=int(m.jnt_dofadr[joint]);fq=int(m.jnt_qposadr[joint]);point=np.mean(positions,axis=0)
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    mujoco.mj_jac(m,d,jp,jr,point,int(m.geom_bodyid[pads[1]]));robot=jp[:,dofs].copy()
    mujoco.mj_jac(m,d,jp,jr,point,int(m.jnt_bodyid[joint]));fixture_jac=jp[:,[fd]]
    coupling=np.linalg.pinv(fixture_jac)@robot
    return dict(joint=joint,dof=fd,qpos=fq,matrix=coupling,pads=pads,point=point,
        note='kinematic prediction from current bilateral contact and policy-induced velocity; live fixture state never assigned')


def coupled_distance_rows(check,q,dofs,coupling,activation=.02):
    pairs,distances=check.distances(q,'manipulate');active=np.flatnonzero(distances<activation)
    ja=np.zeros((3,check.m.nv));jb=ja.copy();jr=ja.copy();rows=[]
    fd=coupling['dof'];C=coupling['matrix']
    for i in active:
        a,b=map(int,pairs[i]);segment=np.zeros(6);distance=check.geom_distance(a,b,.10,segment)
        normal=segment[3:]-segment[:3];length=np.linalg.norm(normal)
        if length<1e-12:rows.append(np.zeros(len(dofs)));continue
        normal/=length
        if distance<0:normal=-normal
        mujoco.mj_jac(check.m,check.d,ja,jr,segment[:3],int(check.m.geom_bodyid[a]))
        mujoco.mj_jac(check.m,check.d,jb,jr,segment[3:],int(check.m.geom_bodyid[b]))
        rows.append(normal@((jb[:,dofs]+jb[:,[fd]]@C)-(ja[:,dofs]+ja[:,[fd]]@C)))
    return np.asarray(rows).reshape(-1,len(dofs)),distances[active]


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
            view=min(frustum_compatibility(m,scratch,target['pos'],ref.policy_cameras),frustum_compatibility(m,live,target['pos'],ref.policy_cameras))
            candidate.update(hard_valid=valid,reason='passed_geometric_prefix_and_endpoint_reach' if valid else 'continuous_or_reach_failure',
                base_goal=goal.tolist(),base_path=np.c_[path,np.full(len(path),goal[2])].tolist(),
                minimum_continuous_clearance_m=min(float(stow_sweep.get('lower_bound_m',0)),float(navigation.get('lower_bound_m',0)),float(reach_sweep.get('lower_bound_m',0))),
                minimum_manipulability_or_joint_margin=margin,minimum_policy_view_compatibility=view,
                policy_view_note='computed Source/endpoint fraction of three camera frusta containing initial EEF near handle; no occlusion or learned-policy compatibility claim',
                total_planned_base_path_m=length,total_planned_time_s=length/(.015 if ref.args.task=='CloseDrawer' else .09),
                endpoint_ik=ik,stow_sweep=stow_sweep,navigation_sweep=navigation,reach_sweep=reach_sweep)
            candidate['endpoint_arm_qpos']=q.tolist()
        except Exception as exc:candidate['reason']=str(exc)
        candidates.append(candidate)
    primary=None
    if any(x['hard_valid'] for x in candidates):primary=rank_primary(candidates)
    assert np.array_equal(q0,live.qpos)
    return dict(version=VERSION,environment_step_calls=0,live_qpos_unchanged=True,proposal_count=5,candidates=candidates,primary=primary,stow=stow,
        provenance='existing native primitives and fixture geometry; no reference manipulation trajectory; geometry only, actual runtime still needs qualification')


def collaborative_paths(ref):
    """Five outcome-blind base paths while retaining the initially observed EEF.

    Endpoint geometry defines only a mobile candidate, never online manipulation
    points. Runtime EEF intentions still come entirely from new policy queries.
    """
    m,live=ref.model_data();q0=live.qpos.copy();base=ref.robot.part_controllers['base'];bids=np.asarray(base.qpos_index)
    qids,dofs,limits=arm_indices(ref);site=ref.robot.eef_site_id['right']
    target=dict(pos=live.site_xpos[site].copy(),rot=live.site_xmat[site].reshape(3,3).copy())
    fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
    check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005);candidates=[]
    for i,offset in enumerate(DOCK_OFFSETS):
        goal=q0[bids].copy();goal[:2]+=offset
        rec=dict(candidate_id=f'collaborative-{i}',base_goal=goal.tolist(),hard_valid=False)
        try:
            base_path=densify_path(np.array([q0[bids[:2]],goal[:2]]),.01);states=[q0.copy()]
            scratch=mujoco.MjData(m);margin=.1;view=1.;errors=[]
            for xy in base_path[1:]:
                scratch.qpos[:]=states[-1];scratch.qpos[bids[:2]]=xy;mujoco.mj_forward(m,scratch)
                q,pe,re,ik=constrained_pose_ik(m,scratch,site,qids,dofs,target,scratch.qpos[qids].copy(),limits,check,phase='manipulate')
                if pe>=.01 or re>=.10:raise ValueError('initial-EEF geometric reach failure')
                states.append(scratch.qpos.copy());errors.append([pe,re])
                margin=min(margin,float(np.min(np.minimum(q-limits[:,0],limits[:,1]-q))))
                view=min(view,frustum_compatibility(m,scratch,target['pos'],ref.policy_cameras))
            swept=check.path(states,['manipulate']*(len(states)-1));length=float(np.linalg.norm(np.diff(base_path,axis=0),axis=1).sum())
            rec.update(hard_valid=bool(swept['valid']),reason='geometry_pass' if swept['valid'] else 'swept_failure',
                minimum_continuous_clearance_m=float(swept.get('lower_bound_m',0)),minimum_manipulability_or_joint_margin=margin,
                minimum_policy_view_compatibility=view,total_planned_base_path_m=length,
                total_planned_time_s=length/(.015 if ref.args.task=='CloseDrawer' else .09),predicted_sweep=swept,
                base_path=np.c_[base_path,np.full(len(base_path),goal[2])].tolist(),endpoint_errors=errors)
        except Exception as exc:rec['reason']=str(exc)
        candidates.append(rec)
    assert np.array_equal(q0,live.qpos)
    return dict(version=VERSION+'-collaborative',environment_step_calls=0,live_qpos_unchanged=True,proposal_count=5,candidates=candidates,
        primary=rank_primary(candidates) if any(x['hard_valid'] for x in candidates) else None,
        provenance='simulator-oracle geometry, observed initial EEF only; no policy or task rollout')


def prefix_action(ref,plan,phase,index=0):
    m,d=ref.model_data();body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ref.base_body)
    R=d.xmat[body].reshape(3,3);point=dict(pos=d.xpos[body]+R@np.asarray(plan['stow']['local_pos']),rot=R@np.asarray(plan['stow']['local_rot']),grasp=-1.)
    base=ref.robot.part_controllers['base'];goal=d.qpos[base.qpos_index].copy() if phase=='stow' else np.asarray(plan['primary']['base_path'][index])
    action,pe,re,be=mapped_action(ref,point,goal,arm_enabled=True)
    return action,dict(phase=phase,geometric_stow_position_error_m=pe,geometric_stow_rotation_error_rad=re,base_error_generalized=be,base_goal=goal.tolist())


def whole_body_action(ref,intent,base_goal,previous_velocity=None,locked_base=False,co_motion=False,actuated_grip=False):
    m,d=ref.model_data();arm=ref.robot.part_controllers['right'];base=ref.robot.part_controllers['base'];qids,ad,limits=arm_indices(ref)
    bd=np.asarray(base.qvel_index);dofs=np.r_[bd,ad];site=ref.robot.eef_site_id['right'];dt=.05
    grip=None
    if actuated_grip:
        gripper=ref.robot.gripper['right']
        gj=np.asarray([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in gripper.joints])
        gq=m.jnt_qposadr[gj];gd=m.jnt_dofadr[gj];gl=m.jnt_range[gj]
        directions=np.sign(np.where(abs(gl[:,1])>abs(gl[:,0]),gl[:,1],gl[:,0]))
        grip=dict(qids=gq,dofs=gd,limits=gl,directions=directions)
        dofs=np.r_[dofs,gd]
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(m,d,jp,jr,site);J=np.vstack([jp[:,dofs],jr[:,dofs]])
    ep=np.asarray(intent['pos'])-d.site_xpos[site]
    er=Rotation.from_matrix(np.asarray(intent['rot'])@d.site_xmat[site].reshape(3,3).T).as_rotvec()
    task=ref.args.task;speed=.015 if task=='CloseDrawer' else .09
    desired_base=np.clip((np.asarray(base_goal)-d.qpos[base.qpos_index])*1.5,-speed,speed)
    twist=np.r_[np.clip(ep*4,-.1,.1),np.clip(er*4,-.1,.1)]
    # The fixed base goal is planner output; only the frozen policy supplies EEF
    # intent. This is one convex tracking QP through the existing primitive.
    augmented=np.vstack([J,np.c_[np.eye(3)*.1,np.zeros((3,len(dofs)-3))]])
    target=np.r_[twist,desired_base*.1]
    lower=np.r_[np.full(3,-speed),np.maximum(-1.,(limits[:,0]+.01501-d.qpos[qids])/dt)]
    upper=np.r_[np.full(3,speed),np.minimum(1.,(limits[:,1]-.01501-d.qpos[qids])/dt)]
    if locked_base:lower[:3]=0.;upper[:3]=0.;desired_base[:]=0.;target[-3:]=0.
    if grip is not None:
        # Include the two physically actuated finger slides in the same
        # distance constraints. Arm/base cannot change a rigid finger gap.
        # Preserve the inherited 1mm solver buffer and 0.5mm actual sweep.
        gv=-grip['directions']*np.sign(intent['grasp'])*.01
        augmented=np.vstack([augmented,np.c_[np.zeros((2,10)),np.eye(2)]])
        target=np.r_[target,gv]
        lower=np.r_[lower,np.maximum(-.01,(gl[:,0]-d.qpos[gq])/dt)]
        upper=np.r_[upper,np.minimum(.01,(gl[:,1]-d.qpos[gq])/dt)]
    if previous_velocity is not None:
        acceleration=np.r_[np.full(3,.2),np.full(len(ad),2.),np.full(2,.2) if grip is not None else np.empty(0)]
        lower=np.maximum(lower,previous_velocity-acceleration*dt);upper=np.minimum(upper,previous_velocity+acceleration*dt)
    fixture=ref.env.drawer if task=='CloseDrawer' else ref.env.door_fxtr
    if getattr(ref,'pi05_qp_geometry_model',None)!=id(m):
        ref.pi05_qp_geometry=SweptGeometry(m,target_prefix=fixture.name,margin=.0005);ref.pi05_qp_geometry_model=id(m)
    check=ref.pi05_qp_geometry
    coupling=observed_articulation_coupling(ref,dofs) if co_motion and intent['grasp']>0 else None
    rows,distances=coupled_distance_rows(check,d.qpos.copy(),dofs,coupling) if coupling is not None else distance_rows(check,d.qpos.copy(),'manipulate',dofs)
    inequalities=(rows,(.001-distances)/dt) if len(rows) else None
    velocity,receipt=velocity_level_qp(augmented,target,lower,upper,base_weight=1.,damping=.001,inequalities=inequalities)
    if not receipt['feasible']:raise QPProtectionStop('whole-body QP constraints infeasible')
    predicted=d.qpos.copy();controlled=np.r_[base.qpos_index,qids,grip['qids'] if grip is not None else np.empty(0,dtype=int)]
    predicted[controlled]+=velocity*dt
    coupling_record=None
    if coupling is not None:
        proposed=float((coupling['matrix']@velocity)[0])*dt;fq=coupling['qpos'];joint=coupling['joint']
        predicted[fq]=np.clip(predicted[fq]+proposed,*m.jnt_range[joint])
        coupling_record=dict(joint=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,joint),pads=list(coupling['pads'].values()),predicted_fixture_delta=float(predicted[fq]-d.qpos[fq]),note=coupling['note'])
    swept=check.path([d.qpos.copy(),predicted],['manipulate'])
    if not swept['valid']:raise QPProtectionStop('whole-body predicted swept clearance failure')
    # Express the arm part of QP motion through the unchanged native OSC.
    arm_twist=J[:,3:10]@velocity[3:10]
    point=dict(pos=d.site_xpos[site]+arm_twist[:3]*dt,rot=Rotation.from_rotvec(arm_twist[3:]*dt).as_matrix()@d.site_xmat[site].reshape(3,3),grasp=intent['grasp'])
    action,pe,re,be=mapped_action(ref,point,d.qpos[base.qpos_index]+velocity[:3]/1.5,arm_enabled=True)
    # QP output is velocity, not a tiny position-servo error. Preserve the
    # inherited friction/actuator/frame map without applying its position deadband.
    ids=np.asarray(ref.robot._ref_actuators_indexes_dict['base'],int)
    friction=m.dof_frictionloss[base.qvel_index]/m.actuator_gainprm[ids,0]
    requested=velocity[:3]+np.where(abs(velocity[:3])>1e-6,np.sign(velocity[:3])*friction,0.)
    goal=requested/(.5*(base.actuator_max-base.actuator_min));_,ori=base.get_base_pose()
    theta=np.arctan2(ori[1,0],ori[0,0])-np.arctan2(base.init_ori[1,0],base.init_ori[0,0])
    mapping=np.array([[-np.sin(theta),np.cos(theta),0],[np.cos(theta),np.sin(theta),0],[0,0,1]])
    action[7:10]=np.linalg.solve(mapping,goal)
    grip_record=None
    if grip is not None:
        goals=predicted[grip['qids']]
        normalized=2*(goals-gl[:,0])/(gl[:,1]-gl[:,0])-1
        grip_record=dict(target_qpos=goals.tolist(),normalized_position_goal=normalized.tolist(),velocity_m_s=velocity[10:].tolist(),
            control='native gripper position goal; action0 preserves the goal; never assign live qpos',solver_buffer_m=.001)
    receipt=dict(receipt,velocity=velocity.tolist(),predicted_sweep=swept,nominal_eef_pos=np.asarray(intent['pos']).tolist(),qp_eef_pos=point['pos'].tolist(),base_goal=np.asarray(base_goal).tolist(),arm_nullspace_goal=(d.qpos[qids]+velocity[3:10]*dt).tolist(),observed_contact_coupling=coupling_record,actuated_gripper=grip_record)
    return action,receipt,velocity
