"""Conservative joint-linear / rigid quaternion-geodesic swept geometry checks.

No environment step, controller update, outcome or future recording is read.
Certification applies to the supplied geometric path, not an OSC rollout.
"""
from __future__ import annotations
import copy
import numpy as np
import mujoco


class SweptGeometry:
    def __init__(self, model, *, target_prefix='', margin=0.0005, max_depth=12):
        # 3.2.6's legacy libccd distance path uses distmax as an inflation
        # margin and can report cap-dependent signed distances for mesh pairs.
        # Use native GJK/EPA only on an independent geometry model: never
        # change live simulation/contact solver flags or physical parameters.
        self.m=copy.copy(model)
        self.m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)
        self.m.opt.ccd_tolerance=1e-9
        self.m.opt.ccd_iterations=1000
        self.d=mujoco.MjData(self.m); self.margin=margin
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
        self.floating=np.flatnonzero(model.jnt_type<int(mujoco.mjtJoint.mjJNT_SLIDE))
        self.floating_radius=np.zeros((model.ngeom,len(self.floating)))
        self.floating_motion=np.zeros_like(self.floating_radius)
        self.floating_articulated=np.zeros(len(self.floating),dtype=bool)
        for column,j in enumerate(self.floating):
            body=int(model.jnt_bodyid[j])
            for geom,chain in enumerate(self.chains):
                if body not in chain:continue
                below=chain[:chain.index(body)]
                self.floating_articulated[column] |= bool(model.body_jntnum[body]>1 or any(model.body_jntnum[b] for b in below))
                self.floating_motion[geom,column]=1.
                self.floating_radius[geom,column]=(np.linalg.norm(model.geom_pos[geom])+model.geom_rbound[geom]
                    +sum(np.linalg.norm(model.body_pos[b]) for b in below)+2*np.linalg.norm(model.jnt_pos[j]))

    def geom_distance(self,a,b,ceiling=.10,segment=None):
        # NativeCCD in 3.2.6 returns before mj_geomDistance restores the input
        # geom order. Call with canonical type order and restore witnesses
        # ourselves, so signed-distance Jacobians use the correct two bodies.
        flip=self.m.geom_type[a]>self.m.geom_type[b]
        first,second=(b,a) if flip else (a,b)
        value=mujoco.mj_geomDistance(self.m,self.d,int(first),int(second),ceiling,segment)
        if flip and segment is not None:
            segment[:]=np.r_[segment[3:].copy(),segment[:3].copy()]
        if not np.isfinite(value):raise ValueError('non-finite geometry distance; fail closed')
        # NativeCCD returns mjMAXVAL when separation exceeds dist_cutoff.
        # Keep the advertised finite ceiling as a lower bound, not 1e10.
        return min(float(value),ceiling)

    def motion_bounds(self,q0,q1):
        """Per-geometry displacement bounds on scalar-linear / quaternion-geodesic paths.

        Revolute radius is bounded by summed descendant link/joint offsets,
        geometry radius, and maximal descendant slider excursions. Triangle
        inequalities make the bound conservative even when axes move.
        Moving rigid floating subtrees add translation plus radius times angle;
        articulated floating subtrees remain unsupported and fail closed.
        """
        m=self.m
        floating_bound=np.zeros(m.ngeom)
        velocity=np.zeros(m.nv)
        if len(self.floating):
            self._validate_quaternions(q0);self._validate_quaternions(q1)
            mujoco.mj_differentiatePos(m,velocity,1.,np.asarray(q0,float),np.asarray(q1,float))
        for column,j in enumerate(self.floating):
            adr=int(m.jnt_qposadr[j]);width=7 if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 4
            if not np.array_equal(q0[adr:adr+width],q1[adr:adr+width]):
                if self.floating_articulated[column]:
                    raise ValueError('changed articulated free/ball subtree unsupported; fail closed')
                dof=int(m.jnt_dofadr[j]);free=width==7
                angle=float(np.linalg.norm(velocity[dof+3:dof+6] if free else velocity[dof:dof+3]))
                translation=float(np.linalg.norm(np.asarray(q1)[adr:adr+3]-np.asarray(q0)[adr:adr+3])) if free else 0.
                # Rigid-point path length <= translation + radius * angle.
                floating_bound+=self.floating_motion[:,column]*translation+self.floating_radius[:,column]*angle
        h=m.jnt_qposadr[self.hinges];s=m.jnt_qposadr[self.slides]
        excursions=np.maximum(abs(q0[s]),abs(q1[s]))
        radii=self.rigid_radius+np.einsum('ghs,s->gh',self.slide_radius,excursions)
        return radii@abs(q1[h]-q0[h])+self.slide_motion@abs(q1[s]-q0[s])+floating_bound

    def _validate_quaternions(self,q):
        for j in self.floating:
            adr=int(self.m.jnt_qposadr[j])+(3 if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 0)
            quat=np.asarray(q)[adr:adr+4]
            if not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1.)>1e-10:
                raise ValueError('invalid free/ball quaternion')

    def midpoint(self,q0,q1):
        if not len(self.floating):return (q0+q1)/2
        self._validate_quaternions(q0);self._validate_quaternions(q1)
        velocity=np.zeros(self.m.nv)
        mujoco.mj_differentiatePos(self.m,velocity,1.,q0,q1)
        mid=np.asarray(q0,float).copy()
        mujoco.mj_integratePos(self.m,mid,velocity,.5)
        return mid

    def distances(self,q,phase):
        self._validate_quaternions(q)
        m,d=self.m,self.d;d.qpos[:]=q
        mujoco.mj_normalizeQuat(m,d.qpos)
        d.qvel[:]=0;mujoco.mj_forward(m,d)
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
        for i in near:values[i]=self.geom_distance(a[i],b[i])
        return pairs,values

    def segment(self,q0,q1,phase,depth=0):
        bounds=self.motion_bounds(q0,q1)
        mid=self.midpoint(q0,q1)
        pairs,dist=self.distances(mid,phase)
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
                    scope='conservative joint-linear and rigid free/ball geodesic swept bound; configured contact rules; not actual rollout certification')
