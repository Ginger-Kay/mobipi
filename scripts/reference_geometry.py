"""Pre-outcome geometry helpers for reference-conditioned development control."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def pose_ik(model, data, site, qids, dofs, target, seed, limits):
    data.qpos[qids]=seed
    jp=np.zeros((3,model.nv));jr=jp.copy()
    for _ in range(100):
        mujoco.mj_forward(model,data)
        ep=target['pos']-data.site_xpos[site]
        er=Rotation.from_matrix(target['rot']@data.site_xmat[site].reshape(3,3).T).as_rotvec()
        if np.linalg.norm(ep)<.004 and np.linalg.norm(er)<.04:break
        mujoco.mj_jacSite(model,data,jp,jr,site)
        J=np.vstack([jp[:,dofs],.2*jr[:,dofs]])
        error=np.r_[ep,.2*er]
        dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*.0002,error)
        data.qpos[qids]=np.clip(data.qpos[qids]+np.clip(dq,-.12,.12),limits[:,0]+.015,limits[:,1]-.015)
    mujoco.mj_forward(model,data)
    ep=float(np.linalg.norm(target['pos']-data.site_xpos[site]))
    er=float(np.linalg.norm(Rotation.from_matrix(target['rot']@data.site_xmat[site].reshape(3,3).T).as_rotvec()))
    return data.qpos[qids].copy(),ep,er


def plan_dock(ref,points):
    """Rank 9 predeclared geometric candidates by IK reachability, no rollouts.

    Only robot/fixture geometry is read. This checks sampled manipulation poses,
    not the swept navigation path or formal planner Gate.
    """
    m,d=ref.model_data();arm=ref.robot.part_controllers['right'];base=ref.robot.part_controllers['base']
    qids=np.asarray(arm.qpos_index,int);dofs=np.asarray(arm.qvel_index,int)
    joint_ids=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in qids]
    limits=m.jnt_range[joint_ids]
    site=ref.robot.eef_site_id['right'];scratch=mujoco.MjData(m)
    opening=float(ref.trace()['target']['door'])
    first=next((i for i,p in enumerate(points) if p['opening']<opening-.01),0)
    indices=np.unique(np.linspace(first,len(points)-1,8).astype(int))
    start=points[0]['base'];end=points[-1]['base'];delta=end[:2]-start[:2]
    lateral=np.array([-delta[1],delta[0]])/max(np.linalg.norm(delta),1e-6)
    body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ref.base_body)
    from reference_stow import build_stow, load_stow
    stow_record=build_stow(ref)
    local,localrot,stow_q=load_stow(ref,stow_record)
    R=d.xmat[body].reshape(3,3)
    stow=dict(pos=d.xpos[body]+R@local,rot=R@localrot)
    scratch.qpos[:]=d.qpos
    scratch.qpos[qids]=stow_q
    mujoco.mj_forward(m,scratch)
    stow_pe=float(np.linalg.norm(scratch.site_xpos[site]-stow['pos']))
    stow_re=float(Rotation.from_matrix(stow['rot']@scratch.site_xmat[site].reshape(3,3).T).magnitude())
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
    def robot(name):return name.startswith(('robot0_','gripper0_','mobilebase0_'))
    rows=[]
    for fraction in [.25,.5,.75]:
        for side in [-.12,0.,.12]:
            dock=(1-fraction)*start+fraction*end;dock=dock.copy();dock[:2]+=side*lateral
            scratch.qpos[:]=d.qpos;scratch.qpos[base.qpos_index]=dock
            seed=points[first]['arm_qpos'].copy();samples=[];first_q=None
            for i in indices:
                point=points[int(i)]
                if point.get('fixture_qpos_address') is not None:scratch.qpos[point['fixture_qpos_address']]=point['fixture_qpos']
                q,pe,re=pose_ik(m,scratch,site,qids,dofs,point,seed,limits)
                if first_q is None:first_q=q.copy()
                seed=q
                samples.append(dict(waypoint=int(i),pos_error_m=pe,rot_error_rad=re,joint_margin_rad=float(np.min(np.minimum(q-limits[:,0],limits[:,1]-q)))))
            valid=all(s['pos_error_m']<.012 and s['rot_error_rad']<.10 for s in samples)
            navigation_contacts=[]
            for f in np.linspace(0,1,17):
                scratch.qpos[:]=d.qpos;scratch.qpos[qids]=stow_q
                # Navigation commands an open gripper; plan the same aperture.
                for j in range(m.njnt):
                    name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or ''
                    if name.startswith('gripper0_') and m.jnt_limited[j]:
                        scratch.qpos[m.jnt_qposadr[j]]=m.jnt_range[j,np.argmax(np.abs(m.jnt_range[j]))]
                scratch.qpos[base.qpos_index]=(1-f)*d.qpos[base.qpos_index]+f*dock
                mujoco.mj_forward(m,scratch)
                for c in scratch.contact:
                    n1,n2=names[c.geom1],names[c.geom2]
                    if robot(n1)!=robot(n2) and 'floor' not in (n1+n2):
                        navigation_contacts.append(dict(fraction=float(f),pair=[n1,n2],distance=float(c.dist)))
            nav_valid=not navigation_contacts and stow_pe<.012 and stow_re<.10
            rows.append(dict(navigation_valid=nav_valid,navigation_contacts=navigation_contacts,sampled_plan_valid=valid and nav_valid,id=len(rows),fraction=fraction,lateral_m=side,dock=dock.tolist(),arm_nullspace_goal=first_q.tolist(),samples=samples,sampled_ik_valid=valid,score=max(s['pos_error_m']+.2*s['rot_error_rad'] for s in samples)))
    for row in rows:row['stow_target']=stow_record
    ranked=sorted(rows,key=lambda r:(not r['sampled_plan_valid'],r['score'],r['id']))
    return dict(candidates=rows,selected=ranked[0],scope='8 manipulation IK poses and 17 stowed navigation collision samples; not continuous swept-path certification',reference_conditioned=True)


def translation_limit(ref):
    """Same contact-conditioned effort limit for E/D/A, without changing physics."""
    if ref.args.task!='CloseSingleDoor':return .10
    # Coordinated base motion retains the v15 cap exactly; stationary-base
    # manipulation uses the contact-conditioned extra effort from v16.
    if not getattr(ref,'base_locked',False):return .20
    m,d=ref.model_data();pairs={}
    for c in d.contact:
        ids=[int(c.geom1),int(c.geom2)]
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in ids]
        for i,n in enumerate(names):
            if n.startswith(ref.env.door_fxtr.name) and 'handle' in n:
                other=names[1-i]
                for finger in (1,2):
                    if 'gripper0_right_finger'+str(finger) in other:pairs.setdefault(ids[i],set()).add(finger)
    return .40 if any(len(x)==2 for x in pairs.values()) else .20


class PalmClearance:
    """Project the next OSC translation away from nearby hand/fixture geometry.

    Local linear distance guard; does not replace post-step collision detection.
    No model parameters or collision masks are changed.
    """
    def __init__(self,ref,margin=.004):
        self.ref=ref;self.margin=margin
        self.action_limit=.20 if ref.args.task=='CloseSingleDoor' else .10
        m,_=ref.model_data()
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
        self.palms=[i for i,n in enumerate(names) if 'gripper0_' in n and 'hand_collision' in n]
        fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
        self.fixture_geoms=[i for i,n in enumerate(names) if n.startswith(fixture.name) and (m.geom_contype[i] or m.geom_conaffinity[i])]
    def apply(self,action):
        m,d=self.ref.model_data();arm=self.ref.robot.part_controllers['right']
        site=self.ref.robot.eef_site_id['right'];R=np.asarray(arm.origin_ori)
        translation=R@(action[:3]*np.asarray(arm.output_max)[:3])
        rotation=R@(action[3:6]*np.asarray(arm.output_max)[3:6])
        receipt=[]
        fixture_dofs=[int(m.jnt_dofadr[j]) for j in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or '').endswith('_slidejoint')]
        base=self.ref.robot.part_controllers['base']
        jp=np.zeros((3,m.nv));jr=jp.copy()
        for palm in self.palms:
            for fixture_geom in self.fixture_geoms:
                segment=np.zeros(6);distance=float(mujoco.mj_geomDistance(m,d,palm,fixture_geom,.03,segment))
                if distance>=.03:continue
                normal=segment[:3]-segment[3:];length=np.linalg.norm(normal)
                if length<1e-9:continue
                normal/=length
                if distance<0:normal=-normal
                rotational_motion=np.cross(rotation,segment[:3]-d.site_xpos[site])
                mujoco.mj_jac(m,d,jp,jr,segment[:3],int(m.geom_bodyid[palm]))
                base_motion=jp[:,base.qvel_index]@d.qvel[base.qvel_index]/self.ref.env.control_freq
                motion=translation+rotational_motion+base_motion
                # With both pads on this handle, predict co-motion only along
                # its physical slide joint. Actual post-step contacts still guard failure.
                pads=set()
                for c in d.contact:
                    other=int(c.geom2) if c.geom1==fixture_geom else (int(c.geom1) if c.geom2==fixture_geom else -1)
                    if other>=0:
                        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,other) or ''
                        if 'gripper0_' in name and 'pad_collision' in name:pads.add(other)
                co_motion=np.zeros(3)
                if len(pads)>=2 and fixture_dofs:
                    hj=np.zeros((3,m.nv));hr=hj.copy()
                    mujoco.mj_jac(m,d,hj,hr,segment[3:],int(m.geom_bodyid[fixture_geom]))
                    J=hj[:,fixture_dofs]
                    co_motion=J@np.linalg.pinv(J)@motion
                correction=max(0.,self.margin-distance-float(normal@(motion-co_motion)))
                if correction:
                    translation+=normal*min(correction,.005)
                receipt.append(dict(palm=palm,fixture_geom=fixture_geom,distance_m=distance,correction_m=correction,pad_contacts=len(pads),predicted_fixture_motion=co_motion.tolist()))
        action[:3]=np.clip((R.T@translation)/np.asarray(arm.output_max)[:3],-translation_limit(self.ref),translation_limit(self.ref))
        return action,receipt


def adjust_reference(ref,scratch,point,margin=.004):
    """Offset unsafe reference palm geometry before the tracking-error gate.

    Signed-distance normal is reversed inside penetration; visual duplicate
    geoms are excluded. This only changes target geometry, never simulator state.
    """
    guard=PalmClearance(ref,margin)
    m,_=ref.model_data();shift=np.zeros(3)
    for _ in range(3):
        for palm in guard.palms:
            for fixture_geom in guard.fixture_geoms:
                segment=np.zeros(6)
                distance=float(mujoco.mj_geomDistance(m,scratch,palm,fixture_geom,.05,segment))
                if distance>=.05:continue
                normal=segment[:3]-segment[3:];length=np.linalg.norm(normal)
                if length<1e-9:continue
                normal/=length
                if distance<0:normal=-normal
                shift+=normal*max(0.,margin-distance-float(normal@shift))
    if np.linalg.norm(shift)>.03:
        raise ValueError('Reference needs more than 3cm palm correction; requires geometry review')
    point['palm_reference_offset']=shift.copy()
    point['pos']=point['pos']+shift
    return point
