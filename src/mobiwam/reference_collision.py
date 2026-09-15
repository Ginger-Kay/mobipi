"""Conservative joint-linear swept geometry checks on a scratch MuJoCo state.

No environment step, controller update, outcome or future recording is read.
Certification applies to the supplied geometric path, not an OSC rollout.
"""
from __future__ import annotations
import numpy as np
import mujoco


class SweptGeometry:
    def __init__(self, model, *, target_prefix='', margin=0.0005, max_depth=12):
        self.m=model; self.d=mujoco.MjData(model); self.margin=margin
        if model.npair:
            raise ValueError('explicit geom-pair overrides need an adapter; fail closed')
        self.max_depth=max_depth; self.target_prefix=target_prefix
        self.names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(model.ngeom)]
        self.robot=np.array([n.startswith(('robot0_','gripper0_','mobilebase0_')) for n in self.names])
        ids=np.flatnonzero((model.geom_contype!=0)|(model.geom_conaffinity!=0))
        pairs=[]; excluded=set(int(x) for x in model.exclude_signature)
        for offset,a in enumerate(ids):
            for b in ids[offset+1:]:
                if not (self.robot[a] or self.robot[b]):continue
                if not ((model.geom_contype[a]&model.geom_conaffinity[b]) or
                        (model.geom_contype[b]&model.geom_conaffinity[a])):continue
                ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
                wa,wb=int(model.body_weldid[ba]),int(model.body_weldid[bb])
                if wa==wb:continue
                if (min(ba,bb)<<16)+max(ba,bb) in excluded:continue
                if not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)):
                    pa=int(model.body_weldid[model.body_parentid[wa]])
                    pb=int(model.body_weldid[model.body_parentid[wb]])
                    if wa and wb and (pa==wb or pb==wa):continue
                # Mobile base floor support is intentional. Arm/finger-floor
                # and all other robot/environment pairs are still checked.
                if ('floor' in self.names[a] and self.names[b].startswith('mobilebase0_')) or ('floor' in self.names[b] and self.names[a].startswith('mobilebase0_')):continue
                pairs.append((a,b))
        self.pairs=np.asarray(pairs,dtype=int).reshape(-1,2)
        self.allowed_manipulation=np.array([
            ('finger' in self.names[a] and self.names[b].startswith(target_prefix) or
             'finger' in self.names[b] and self.names[a].startswith(target_prefix))
            if target_prefix else False for a,b in pairs])
        self.chains=[]
        for geom in range(model.ngeom):
            chain=[];body=int(model.geom_bodyid[geom])
            while body:
                chain.append(body);body=int(model.body_parentid[body])
            self.chains.append(chain)
        self.hinges=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
        self.slides=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_SLIDE)
        hmap={int(j):i for i,j in enumerate(self.hinges)}
        smap={int(j):i for i,j in enumerate(self.slides)}
        self.rigid_radius=np.zeros((model.ngeom,len(self.hinges)))
        self.slide_radius=np.zeros((model.ngeom,len(self.hinges),len(self.slides)))
        self.slide_motion=np.zeros((model.ngeom,len(self.slides)))
        for geom,chain in enumerate(self.chains):
            reach=float(np.linalg.norm(model.geom_pos[geom])+model.geom_rbound[geom])
            excursions=np.zeros(len(self.slides))
            for body in chain:
                joints=range(int(model.body_jntadr[body]),int(model.body_jntadr[body]+model.body_jntnum[body]))
                extra=sum(float(np.linalg.norm(model.jnt_pos[j]))*2 for j in joints)
                for j in joints:
                    if j in smap:excursions[smap[j]]+=1
                for j in joints:
                    if j in hmap:
                        self.rigid_radius[geom,hmap[j]]=reach+extra
                        self.slide_radius[geom,hmap[j]]=excursions
                    elif j in smap:self.slide_motion[geom,smap[j]]=1
                reach+=float(np.linalg.norm(model.body_pos[body]))+extra
        self.evaluations=0

    def motion_bounds(self,q0,q1):
        """Per-geometry point displacement upper bounds along q-linear paths.

        Revolute radius is bounded by summed descendant link/joint offsets,
        geometry radius, and maximal descendant slider excursions. Triangle
        inequalities make the bound conservative even when axes move.
        """
        m=self.m
        for j in np.flatnonzero(m.jnt_type<mujoco.mjtJoint.mjJNT_SLIDE):
            adr=int(m.jnt_qposadr[j]);width=7 if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 4
            if not np.array_equal(q0[adr:adr+width],q1[adr:adr+width]):
                raise ValueError('changed free/ball joint unsupported; fail closed')
        h=m.jnt_qposadr[self.hinges];s=m.jnt_qposadr[self.slides]
        excursions=np.maximum(abs(q0[s]),abs(q1[s]))
        radii=self.rigid_radius+np.einsum('ghs,s->gh',self.slide_radius,excursions)
        return radii@abs(q1[h]-q0[h])+self.slide_motion@abs(q1[s]-q0[s])

    def distances(self,q,phase):
        m,d=self.m,self.d;d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
        self.evaluations+=1
        active=np.ones(len(self.pairs),dtype=bool)
        if phase=='manipulate':active &= ~self.allowed_manipulation
        pairs=self.pairs[active]
        if not len(pairs):return pairs,np.zeros(0)
        a,b=pairs.T
        lower=np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b],axis=1)-m.geom_rbound[a]-m.geom_rbound[b]
        plane=(m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE)|(m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE)
        lower[plane]=-np.inf
        near=np.flatnonzero(lower<.10)
        values=np.minimum(lower,.10)
        for i in near:values[i]=mujoco.mj_geomDistance(m,d,int(a[i]),int(b[i]),.10,None)
        return pairs,values

    def segment(self,q0,q1,phase,depth=0):
        bounds=self.motion_bounds(q0,q1)
        pairs,dist=self.distances((q0+q1)/2,phase)
        if not len(pairs):return dict(valid=True,lower_bound_m=.10,leaves=1)
        lower=dist-.5*(bounds[pairs[:,0]]+bounds[pairs[:,1]])
        worst=int(np.argmin(lower)); minimum=float(lower[worst])
        if np.min(dist)<0:
            i=int(np.argmin(dist));return dict(valid=False,kind='collision',distance_m=float(dist[i]),
                lower_bound_m=minimum,pair=[self.names[x] for x in pairs[i]],depth=depth,leaves=1)
        if minimum>=self.margin:return dict(valid=True,lower_bound_m=minimum,leaves=1)
        if depth>=self.max_depth:
            return dict(valid=False,kind='clearance_unresolved',lower_bound_m=minimum,
                        pair=[self.names[x] for x in pairs[worst]],depth=depth,leaves=1)
        mid=(q0+q1)/2
        left=self.segment(q0,mid,phase,depth+1)
        if not left['valid']:return left
        right=self.segment(mid,q1,phase,depth+1)
        if not right['valid']:return right
        return dict(valid=True,lower_bound_m=min(left['lower_bound_m'],right['lower_bound_m']),
                    leaves=left['leaves']+right['leaves'])

    def path(self,states,phases):
        states=np.asarray(states,float)
        if len(states)<2 or len(phases)!=len(states)-1 or not np.isfinite(states).all():
            raise ValueError('invalid geometric path')
        receipts=[]; minimum=.10; initial=self.evaluations
        # Endpoints are explicit: midpoint recursion must never miss a
        # collision already present at the Source or at an option transition.
        for index,(q0,q1,phase) in enumerate(zip(states,states[1:],phases)):
            for q in (q0,q1):
                pairs,dist=self.distances(q,phase)
                if len(dist) and min(dist)<self.margin:
                    i=int(np.argmin(dist));return dict(valid=False,kind='endpoint_clearance',
                        segment=index,phase=phase,distance_m=float(dist[i]),
                        lower_bound_m=float(dist[i]),pair=[self.names[x] for x in pairs[i]],
                        evaluations=self.evaluations-initial,completed_segments=index)
            r=self.segment(q0,q1,phase);r.update(segment=index,phase=phase);receipts.append(r)
            minimum=min(minimum,r['lower_bound_m'])
            if not r['valid']:return dict(**r,evaluations=self.evaluations-initial,completed_segments=index)
        return dict(valid=True,lower_bound_m=minimum,segments=len(receipts),
                    leaf_intervals=sum(r['leaves'] for r in receipts),evaluations=self.evaluations-initial,
                    scope='conservative q-linear swept bound; configured contact rules; not actual rollout certification')
