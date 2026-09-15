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
            rows.append(dict(id=len(rows),fraction=fraction,lateral_m=side,dock=dock.tolist(),arm_nullspace_goal=first_q.tolist(),samples=samples,sampled_ik_valid=valid,score=max(s['pos_error_m']+.2*s['rot_error_rad'] for s in samples)))
    ranked=sorted(rows,key=lambda r:(not r['sampled_ik_valid'],r['score'],r['id']))
    return dict(candidates=rows,selected=ranked[0],scope='sampled IK only; no full collision/swept-path certification',reference_conditioned=True)


class PalmClearance:
    """Project the next OSC translation away from a nearby palm/handle pair.

    Local linear distance guard; does not replace post-step collision detection.
    No model parameters or collision masks are changed.
    """
    def __init__(self,ref,margin=.004):
        self.ref=ref;self.margin=margin
        m,_=ref.model_data()
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
        self.palms=[i for i,n in enumerate(names) if 'gripper0_' in n and 'hand_collision' in n]
        fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
        self.handles=[i for i,n in enumerate(names) if n.startswith(fixture.name) and 'handle' in n]
    def apply(self,action):
        m,d=self.ref.model_data();arm=self.ref.robot.part_controllers['right']
        site=self.ref.robot.eef_site_id['right'];R=np.asarray(arm.origin_ori)
        translation=R@(action[:3]*np.asarray(arm.output_max)[:3])
        rotation=R@(action[3:6]*np.asarray(arm.output_max)[3:6])
        receipt=[]
        for palm in self.palms:
            for handle in self.handles:
                segment=np.zeros(6);distance=float(mujoco.mj_geomDistance(m,d,palm,handle,.03,segment))
                if distance>=.03 or distance<=0:continue
                normal=segment[:3]-segment[3:];length=np.linalg.norm(normal)
                if length<1e-9:continue
                normal/=length
                rotational_motion=np.cross(rotation,segment[:3]-d.site_xpos[site])
                correction=max(0.,self.margin-distance-float(normal@(translation+rotational_motion)))
                if correction:
                    translation+=normal*min(correction,.005)
                    # Remove base-induced drift while close to the palm margin.
                    action[7:10]=0.
                receipt.append(dict(palm=palm,handle=handle,distance_m=distance,correction_m=correction))
        action[:3]=np.clip((R.T@translation)/np.asarray(arm.output_max)[:3],-.10,.10)
        return action,receipt
