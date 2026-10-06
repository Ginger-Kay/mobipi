"""Geometric D prefix and contact-triggered A mobility around policy intent."""
import numpy as np
from scipy.spatial.transform import Rotation
from reference_executor import mapped_action
from mobiwam.pi05_motion import docks,collaborative_paths,prefix_action,arm_indices,observed_articulation_coupling,QPProtectionStop

class RouteDriver:
    def __init__(self,ref,route):
        self.ref=ref;self.route=route;self.phase='stow' if route=='D' else 'manipulate'
        try:self.plan=docks(ref) if route=='D' else collaborative_paths(ref)
        except ValueError as exc:self.plan=dict(primary=None,candidates=[],reason=str(exc),environment_step_calls=0)
        m,d=ref.model_data();base=ref.robot.part_controllers['base'];site=ref.robot.eef_site_id['right']
        self.initial_base=d.qpos[base.qpos_index].copy();self.lock_target=self.initial_base.copy()
        self.initial_eef=dict(pos=d.site_xpos[site].copy(),rot=d.site_xmat[site].reshape(3,3).copy(),grasp=-1.)
        fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
        import mujoco
        joints=[j for j in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or '').startswith(fixture.name) and int(m.jnt_type[j]) in (2,3)]
        assert len(joints)==1
        self.fixture_qid=int(m.jnt_qposadr[joints[0]]);self.initial_articulation=float(d.qpos[self.fixture_qid])
        self.waypoint=0;self.settled=0;self.prefix_steps=0;self.a_started=False;self.maximum_overlap=0;self.current_overlap=0
        self.previous=None;self.events=[]
        self.maximum_contact_base_translation=0.;self.maximum_contact_base_yaw=0.

    def prefix(self):
        ref=self.ref;m,d=ref.model_data();base=ref.robot.part_controllers['base'];arm=ref.robot.part_controllers['right']
        ref.base_locked=self.phase in ('stow','reach')
        if self.phase in ('stow','navigate','settle'):
            phase='stow' if self.phase=='stow' else 'navigate'
            action,proof=prefix_action(ref,self.plan,phase,self.waypoint)
            arm.initial_joint=np.asarray(self.plan['stow']['arm_qpos'])
            if self.phase=='stow' and proof['geometric_stow_position_error_m']<.01 and proof['geometric_stow_rotation_error_rad']<.10:
                self.phase='navigate';self.events.append(dict(event='stow_complete',prefix_step=self.prefix_steps))
            elif self.phase=='navigate':
                path=self.plan['primary']['base_path'];error=d.qpos[base.qpos_index]-np.asarray(path[self.waypoint])
                if np.max(abs(error))<.005:
                    if self.waypoint<len(path)-1:self.waypoint+=1
                    else:self.phase='settle'
            elif self.phase=='settle':
                self.settled=self.settled+1 if np.max(abs(d.qvel[base.qvel_index]))<.003 else 0
                if self.settled>=5:
                    self.phase='reach';self.lock_target=d.qpos[base.qpos_index].copy()
                    self.events.append(dict(event='navigation_settled',prefix_step=self.prefix_steps,actual_lock_target=self.lock_target.tolist()))
        else:
            ref.base_locked=True;arm.initial_joint=np.asarray(self.plan['primary']['endpoint_arm_qpos'])
            action,pe,re,be=mapped_action(ref,self.initial_eef,self.lock_target,arm_enabled=True)
            proof=dict(phase='reach',position_error_m=pe,rotation_error_rad=re,base_error=be)
            if pe<.01 and re<.10:
                self.phase='manipulate';self.events.append(dict(event='fresh_policy_query_required',prefix_step=self.prefix_steps))
        action[10]=0.;action[11]=-1.;self.prefix_steps+=1
        return action,proof

    def base_control(self):
        if self.route=='D':return self.lock_target,True
        ref=self.ref;m,d=ref.model_data();_,ad,_=arm_indices(ref);base=ref.robot.part_controllers['base']
        contact=observed_articulation_coupling(ref,np.r_[base.qvel_index,ad])
        articulated=abs(float(d.qpos[self.fixture_qid])-self.initial_articulation)>1e-5
        if contact is not None and articulated:
            if not self.a_started:self.events.append(dict(event='A_mobility_triggered_by_actual_contact_and_articulation',sim_time=float(d.time)))
            self.a_started=True
        # Pause mobility if bilateral contact is lost; no fixture state injection.
        moving=self.a_started and contact is not None
        return (np.asarray(self.plan['primary']['base_goal']) if moving else d.qpos[base.qpos_index].copy()),not moving

    def observe(self,requested_base_velocity=None):
        ref=self.ref;m,d=ref.model_data();_,ad,_=arm_indices(ref);base=ref.robot.part_controllers['base']
        snapshot=np.r_[d.qpos[base.qpos_index],d.qpos[ref.robot.part_controllers['right'].qpos_index],d.qpos[self.fixture_qid]]
        contact=observed_articulation_coupling(ref,np.r_[base.qvel_index,ad]) is not None
        if self.previous is not None:
            delta=abs(snapshot-self.previous)
            requested=np.asarray(requested_base_velocity if requested_base_velocity is not None else np.zeros(3))
            intentional=bool(self.a_started and np.max(abs(requested))>=.0002)
            overlap=bool(self.route=='A' and intentional and contact and np.max(delta[:3])>1e-5 and np.max(delta[3:10])>1e-5 and delta[-1]>1e-6)
            self.current_overlap=self.current_overlap+1 if overlap else 0
            self.maximum_overlap=max(self.maximum_overlap,self.current_overlap)
            if overlap:
                self.maximum_contact_base_translation=max(self.maximum_contact_base_translation,float(np.linalg.norm(snapshot[:2]-self.initial_base[:2])))
                self.maximum_contact_base_yaw=max(self.maximum_contact_base_yaw,float(abs(snapshot[2]-self.initial_base[2])))
        self.previous=snapshot
        return dict(phase=self.phase,contact_bilateral=contact,A_mobility_triggered=self.a_started,maximum_continuous_articulation_base_arm_overlap_controls=self.maximum_overlap)

    def receipt(self,queries):
        return dict(route=self.route,prefix_steps=self.prefix_steps,events=self.events,final_phase=self.phase,queries=queries,
            D_fresh_query_after_settle=bool(self.route=='D' and self.phase=='manipulate' and queries>0),
            A_semantics_observed=bool(self.route=='A' and queries>=2 and self.maximum_overlap>=5 and (self.maximum_contact_base_translation>=.005 or self.maximum_contact_base_yaw>=.01)),
            A_maximum_continuous_overlap_controls=self.maximum_overlap,teacher_manipulation_points_used=False,
            A_contact_base_translation_m=self.maximum_contact_base_translation,A_contact_base_yaw_rad=self.maximum_contact_base_yaw,
            A_definition='intentional nonzero QP base request, retained bilateral contact, articulation/arm/base overlap5 controls, net5mm translation or10mrad yaw; at least2 fresh chunks',
            note='mobility trigger and overlap use actual native contact/joint motion; geometry is simulator oracle')
