"""A-private contact-trigger repair; frozen E/D controller files stay unchanged."""
import mujoco
import numpy as np
from mobiwam.contact_rules import allowed_contact
from mobiwam.pi05_route import RouteDriver
from mobiwam.pi05_motion import arm_indices
from mobiwam.pi05_adapter import execute_projected
import mobiwam.pi05_motion as motion

def permitted_coupling(ref,dofs):
    m,d=ref.model_data();fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
    positions=[];fingers=[]
    for contact in d.contact[:d.ncon]:
        for target,hand in ((int(contact.geom1),int(contact.geom2)),(int(contact.geom2),int(contact.geom1))):
            tn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,target) or ''
            hn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,hand) or ''
            if tn.startswith(fixture.name) and hn.startswith('gripper0_') and 'finger' in hn and allowed_contact(tn,hn,'manipulate',fixture.name):
                positions.append(contact.pos.copy());fingers.append(hand)
    if not positions:return None
    joints=[j for j in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or '').startswith(fixture.name) and int(m.jnt_type[j]) in (2,3)]
    if len(joints)!=1:return None
    joint=joints[0];fd=int(m.jnt_dofadr[joint]);fq=int(m.jnt_qposadr[joint]);point=np.mean(positions,axis=0)
    jp=np.zeros((3,m.nv));jr=jp.copy()
    mujoco.mj_jac(m,d,jp,jr,point,int(m.geom_bodyid[fingers[0]]));robot=jp[:,dofs].copy()
    mujoco.mj_jac(m,d,jp,jr,point,int(m.jnt_bodyid[joint]));fixture_jac=jp[:,[fd]]
    return dict(joint=joint,dof=fd,qpos=fq,matrix=np.linalg.pinv(fixture_jac)@robot,pads={i:g for i,g in enumerate(sorted(set(fingers)))},point=point,
        note='A2 private current permitted finger/target contact; pushing contact is not described as bilateral grasp or force closure; live fixture state never assigned')

def execute_A2(ref,action,anchor,base_target,**kwargs):
    # This is a single-threaded A-only process. The temporary dependency binding
    # affects only this A call; E/D use the unchanged original function/code.
    original=motion.observed_articulation_coupling
    motion.observed_articulation_coupling=permitted_coupling
    try:return execute_projected(ref,action,anchor,base_target,**kwargs)
    finally:motion.observed_articulation_coupling=original

class A2Driver(RouteDriver):
    def __init__(self,ref,route):
        assert route=='A';super().__init__(ref,route);self.native_rows=[];self.native_request=np.zeros(3)
    def boundary(self,request):self.native_request=np.asarray(request).copy()
    def __enter__(self):
        ref=self.ref;sim=ref.env.sim;self.native_method='step2' if ref.env.lite_physics else 'step'
        self.native_original=getattr(sim,self.native_method);self.native_had_instance=self.native_method in vars(sim);self.native_instance=vars(sim).get(self.native_method)
        def tracked(*args,**kwargs):
            answer=self.native_original(*args,**kwargs)
            m,d=ref.model_data();_,ad,_=arm_indices(ref);base=ref.robot.part_controllers['base']
            contact=permitted_coupling(ref,np.r_[base.qvel_index,ad]) is not None
            self.native_rows.append(dict(time=float(d.time),contact=contact,base=d.qpos[base.qpos_index].copy(),
                arm=d.qpos[ref.robot.part_controllers['right'].qpos_index].copy(),fixture=float(d.qpos[self.fixture_qid]),request=self.native_request.copy()))
            return answer
        setattr(sim,self.native_method,tracked);return self
    def __exit__(self,*args):
        if self.native_had_instance:setattr(self.ref.env.sim,self.native_method,self.native_instance)
        else:delattr(self.ref.env.sim,self.native_method)
    def save_native(self,path):
        if not self.native_rows:return
        np.savez_compressed(path/'A2-native-contact.npz',time=[x['time'] for x in self.native_rows],contact=[x['contact'] for x in self.native_rows],
            base=[x['base'] for x in self.native_rows],arm=[x['arm'] for x in self.native_rows],fixture=[x['fixture'] for x in self.native_rows],request=[x['request'] for x in self.native_rows])
    def base_control(self):
        ref=self.ref;m,d=ref.model_data();_,ad,_=arm_indices(ref);base=ref.robot.part_controllers['base']
        contact=permitted_coupling(ref,np.r_[base.qvel_index,ad])
        articulated=abs(float(d.qpos[self.fixture_qid])-self.initial_articulation)>1e-5
        if contact is not None and articulated:
            if not self.a_started:self.events.append(dict(event='A2_mobility_triggered_by_actual_permitted_contact_and_articulation',sim_time=float(d.time)))
            self.a_started=True
        moving=self.a_started and contact is not None
        return (np.asarray(self.plan['primary']['base_goal']) if moving else d.qpos[base.qpos_index].copy()),not moving
    def observe(self,requested_base_velocity=None):
        ref=self.ref;m,d=ref.model_data();_,ad,_=arm_indices(ref);base=ref.robot.part_controllers['base']
        snapshot=np.r_[d.qpos[base.qpos_index],d.qpos[ref.robot.part_controllers['right'].qpos_index],d.qpos[self.fixture_qid]]
        contact=permitted_coupling(ref,np.r_[base.qvel_index,ad]) is not None
        if self.previous is not None:
            delta=abs(snapshot-self.previous);requested=np.asarray(requested_base_velocity if requested_base_velocity is not None else np.zeros(3))
            overlap=bool(self.a_started and contact and np.max(abs(requested))>=.0002 and np.max(delta[:3])>1e-5 and np.max(delta[3:10])>1e-5 and delta[-1]>1e-6)
            self.current_overlap=self.current_overlap+1 if overlap else 0;self.maximum_overlap=max(self.maximum_overlap,self.current_overlap)
            if overlap:
                self.maximum_contact_base_translation=max(self.maximum_contact_base_translation,float(np.linalg.norm(snapshot[:2]-self.initial_base[:2])))
                self.maximum_contact_base_yaw=max(self.maximum_contact_base_yaw,float(abs(snapshot[2]-self.initial_base[2])))
        self.previous=snapshot
        return dict(phase=self.phase,actual_permitted_contact=contact,A_mobility_triggered=self.a_started,maximum_continuous_articulation_base_arm_overlap_controls=self.maximum_overlap,contact_mode='permitted finger/target contact, not bilateral-grasp claim')
    def receipt(self,queries):
        q=super().receipt(queries);q['A_definition']='A2: existing-mask permitted finger/target contact, intentional QP base request,5 continuous articulation/base/arm overlap controls, net5mm or10mrad,2 fresh chunks'
        continuous=maximum=0.
        for before,after in zip(self.native_rows[:-1],self.native_rows[1:]):
            moving=(before['contact'] and after['contact'] and np.max(abs(after['request']))>=.0002
                and np.max(abs(after['base']-before['base']))>1e-7 and np.max(abs(after['arm']-before['arm']))>1e-7 and abs(after['fixture']-before['fixture'])>1e-7)
            continuous=continuous+after['time']-before['time'] if moving else 0.;maximum=max(maximum,continuous)
        q['A_native_contact_overlap_maximum_seconds']=maximum;q['A_native_samples']=len(self.native_rows)
        q['A_semantics_observed']=bool(q['A_semantics_observed'] and maximum>=.25)
        q['A_native_requirement']='at least0.25s consecutive recorded native intervals with permitted contact and actual base/arm/fixture motion; sampled native contact is not a force-closure certificate'
        q['A_private_version']='A2';return q
