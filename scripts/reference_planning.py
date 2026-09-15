"""Pre-execution geometric route compilation and all 21 candidate features.

Development schema binding v1: serial damped IK, q-linear segments, conservative
swept collision bounds, robot velocity/acceleration time scaling, camera FOV.
These are planned quantities, never measurements from a route rollout.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from reference_geometry_v16 import pose_ik
from mobiwam.reference_collision import SweptGeometry
from mobiwam.reference_ik import constrained_pose_ik
from mobiwam.scene004 import candidate_feature_vector
from reference_prefix_preview import preview_prefix


SPEC=dict(version='reference-geometric-candidate-v2',collision_margin_m=.0005,
          manipulation_solver='bounded sequential collision-constrained pose IK',solver_clearance_buffer_m=.001,
          swept_max_depth=12,pose_spacing_m=.015,rotation_spacing_rad=.08,
          arm_velocity_rad_s=1.,arm_acceleration_rad_s2=2.,base_acceleration_m_s2=.2,
          time_horizon_s=120.,clearance_ceiling_m=.10,
          view='minimum across path of fraction of three policy cameras with target handle centroid in frustum; no occlusion claim',
          scope='development binding; not a retrospective formal SCENE-004 freeze')


class Compiler:
    def __init__(self,ref):
        self.ref=ref;self.m,self.live=ref.model_data();self.d=mujoco.MjData(self.m)
        self.arm=ref.robot.part_controllers['right'];self.base=ref.robot.part_controllers['base']
        self.qids=np.asarray(self.arm.qpos_index,int);self.dofs=np.asarray(self.arm.qvel_index,int)
        self.bids=np.asarray(self.base.qpos_index,int)
        joint_ids=[int(np.flatnonzero(self.m.jnt_qposadr==i)[0]) for i in self.qids]
        self.limits=self.m.jnt_range[joint_ids];self.site=ref.robot.eef_site_id['right']
        fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
        self.check=SweptGeometry(self.m,target_prefix=fixture.name,margin=SPEC['collision_margin_m'],max_depth=SPEC['swept_max_depth'])
        self.handles=[i for i,n in enumerate(self.check.names) if n.startswith(fixture.name) and 'handle' in n and (self.m.geom_contype[i] or self.m.geom_conaffinity[i])]
        self.cameras=[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_CAMERA,n) for n in
                      ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand')]
        if not self.handles or min(self.cameras)<0:raise ValueError('target/policy cameras missing')
        self.initial=self.live.qpos.copy()
        self.solver_receipts=[]
        self.gripper=[]
        for j in range(self.m.njnt):
            name=mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_JOINT,j) or ''
            if name.startswith('gripper0_') and self.m.jnt_limited[j]:
                self.gripper.append((int(self.m.jnt_qposadr[j]),float(self.m.jnt_range[j,np.argmax(abs(self.m.jnt_range[j]))])))

    def pose(self,q):
        self.d.qpos[:]=q;mujoco.mj_forward(self.m,self.d)
        return self.d.site_xpos[self.site].copy(),self.d.site_xmat[self.site].reshape(3,3).copy()

    def transition(self,states,phases,errors,target,base,phase,grasp=-1.,fixture=None):
        start=states[-1].copy();pos,rot=self.pose(start)
        delta=Rotation.from_matrix(target['rot']@rot.T).magnitude()
        count=max(1,int(np.ceil(np.linalg.norm(target['pos']-pos)/SPEC['pose_spacing_m'])),
                  int(np.ceil(delta/SPEC['rotation_spacing_rad'])),
                  int(np.ceil(np.linalg.norm(np.asarray(base)-start[self.bids])/.02)))
        slerp=Slerp([0,1],Rotation.from_matrix([rot,target['rot']]))
        for f in np.linspace(0,1,count+1)[1:]:
            self.d.qpos[:]=states[-1];self.d.qpos[self.bids]=(1-f)*start[self.bids]+f*np.asarray(base)
            if fixture is not None:
                address,value=fixture;self.d.qpos[address]=(1-f)*start[address]+f*value
            for adr,opened in self.gripper:
                goal=opened if grasp<0 else np.sign(opened)*.012
                self.d.qpos[adr]=(1-f)*start[adr]+f*goal
            goal=dict(pos=(1-f)*pos+f*target['pos'],rot=slerp(f).as_matrix())
            if phase=='manipulate':
                q,pe,re,receipt=constrained_pose_ik(self.m,self.d,self.site,self.qids,self.dofs,
                    goal,states[-1][self.qids],self.limits,self.check,
                    buffer=SPEC['solver_clearance_buffer_m'])
                self.solver_receipts.append(receipt)
            else:
                q,pe,re=pose_ik(self.m,self.d,self.site,self.qids,self.dofs,goal,states[-1][self.qids],self.limits)
            states.append(self.d.qpos.copy());phases.append(phase);errors.append([pe,re])

    def prefix(self,dock):
        states=[self.initial.copy()];phases=[];errors=[]
        pos,rot=self.pose(self.initial)
        body=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,self.ref.base_body)
        basepos=self.d.xpos[body].copy();offset=pos-basepos
        offset[:2]*=max(0.,1.-.25/max(np.linalg.norm(offset[:2]),1e-6))
        stow=dict(pos=basepos+offset,rot=rot)
        self.transition(states,phases,errors,stow,self.initial[self.bids],'stow')
        # Hold exactly the stowed arm while holonomic base moves; orientation
        # of the stowed EEF follows the base, matching executor semantics.
        q0=states[-1].copy();n=max(1,int(np.ceil(np.linalg.norm(np.asarray(dock)-q0[self.bids])/.02)))
        for f in np.linspace(0,1,n+1)[1:]:
            q=q0.copy();q[self.bids]=(1-f)*q0[self.bids]+f*np.asarray(dock)
            states.append(q);phases.append('navigate');errors.append([0.,0.])
        states.append(states[-1].copy());phases.append('settle');errors.append([0.,0.])
        return states,phases,errors

    def route(self,route,points,dock,prefix=None):
        if route=='D':states,phases,errors=prefix or self.prefix(dock)
        else:states,phases,errors=[self.initial.copy()],[],[]
        states=[q.copy() for q in states];phases=list(phases);errors=list(errors)
        for point in points:
            base=point['base'] if route=='A' else (dock if route=='D' else self.initial[self.bids])
            self.transition(states,phases,errors,point,base,'manipulate',grasp=point['grasp'],
                            fixture=(point['fixture_qpos_address'],point['fixture_qpos']))
        return np.array(states),phases,np.array(errors)

    def metrics(self,route,states,phases,errors,slot,collision):
        manipulability=[];margins=[];views=[];eef=[];basepos=[];coarse_lower=[]
        jp=np.zeros((3,self.m.nv));jr=jp.copy()
        for q in states:
            self.d.qpos[:]=q;mujoco.mj_forward(self.m,self.d)
            mujoco.mj_jacSite(self.m,self.d,jp,jr,self.site)
            manipulability.append(float(np.linalg.svd(np.vstack([jp[:,self.dofs],.2*jr[:,self.dofs]]),compute_uv=False)[-1]))
            margins.append(float(np.minimum(q[self.qids]-self.limits[:,0],self.limits[:,1]-q[self.qids]).min()))
            target=self.d.geom_xpos[self.handles].mean(axis=0);visible=0
            for cam in self.cameras:
                local=self.d.cam_xmat[cam].reshape(3,3).T@(target-self.d.cam_xpos[cam]);depth=-local[2]
                half=depth*np.tan(np.deg2rad(self.m.cam_fovy[cam])/2)
                visible+=bool(depth>0 and abs(local[0])<half and abs(local[1])<half)
            views.append(visible/len(self.cameras));eef.append(self.d.site_xpos[self.site].copy())
            basepos.append(q[self.bids].copy())
        # For invalid paths we still need a route-wide finite geometric lower
        # bound, not a minimum over only the successful prefix before failure.
        if collision['valid']:clearance=collision['lower_bound_m']
        else:
            for q0,q1,phase in zip(states,states[1:],phases):
                pairs,dist=self.check.distances((q0+q1)/2,phase);bound=self.check.motion_bounds(q0,q1)
                if len(pairs):coarse_lower.append(float(np.min(dist-.5*(bound[pairs[:,0]]+bound[pairs[:,1]]))))
            clearance=min(coarse_lower,default=.10)
        robot_ids=np.r_[self.bids,self.qids];deltas=np.diff(states[:,robot_ids],axis=0)
        speed=.015 if self.ref.args.task=='CloseDrawer' else .09
        vmax=np.r_[[speed,speed,.09],np.ones(7)*SPEC['arm_velocity_rad_s']]
        amax=np.r_[[.2,.2,.2],np.ones(7)*SPEC['arm_acceleration_rad_s2']]
        duration=np.maximum(.05,np.max(abs(deltas)/vmax,axis=1))
        duration=np.array([max(t,1.) if p=='settle' else t for t,p in zip(duration,phases)])
        vel=deltas/duration[:,None]
        # Include start/terminal rest when checking acceleration.
        widths=np.r_[duration[0]/2,(duration[:-1]+duration[1:])/2,duration[-1]/2]
        acc=np.diff(np.vstack([np.zeros((1,10)),vel,np.zeros((1,10))]),axis=0)/widths[:,None]
        scale=max(1.,float(np.sqrt(np.max(abs(acc)/amax))))
        duration*=scale;vel/=scale;acc/=scale**2
        ve=float(np.min(1-abs(vel)/vmax));ae=float(np.min(1-abs(acc)/amax))
        pe=float(errors[:,0].max());re=float(errors[:,1].max())
        valid=bool(collision['valid'] and pe<.012 and re<.10 and min(margins)>.015 and ve>=-1e-9 and ae>=-1e-9)
        b=np.asarray(basepos);e=np.asarray(eef)
        features=dict(route_E=float(route=='E'),route_D=float(route=='D'),route_A=float(route=='A'),
            task_CloseDrawer=float(self.ref.args.task=='CloseDrawer'),task_CloseSingleDoor=float(self.ref.args.task=='CloseSingleDoor'),
            stage_precontact=1.,hard_valid=float(valid),minimum_continuous_clearance_m=float(clearance),
            minimum_manipulability=min(manipulability),minimum_joint_margin_rad=min(margins),
            minimum_policy_view_compatibility=min(views),total_planned_base_path_m=float(np.linalg.norm(np.diff(b[:,:2],axis=0),axis=1).sum()),
            planned_time_normalized=float(duration.sum()/SPEC['time_horizon_s']),planned_base_net_m=float(np.linalg.norm(b[-1,:2]-b[0,:2])),
            planned_eef_path_m=float(np.linalg.norm(np.diff(e,axis=0),axis=1).sum()),maximum_eef_tracking_error_m=pe,
            minimum_velocity_margin=ve,minimum_acceleration_margin=ae,solver_residual=float(np.max(errors[:,0]+.2*errors[:,1])),
            slot_index_normalized=float(slot/8 if route=='D' else 0.),simulator_oracle_pre_outcome=1.)
        candidate_feature_vector(features)
        return dict(features=features,hard_valid=valid,collision=collision,planned_duration_s=float(duration.sum()),
                    retiming_scale=scale,maximum_rotation_error_rad=re,segments=len(phases),
                    metrics_scope='entire planned path; no execution feedback/outcome data',duration_s=duration.tolist())


def compile_candidates(ref,points,dock_plan,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    compiler=Compiler(ref);snapshot=ref.integration().copy();prefixes={}
    # Check every candidate including the full Source -> stow transition.
    for candidate in dock_plan['candidates']:
        states,phases,errors=compiler.prefix(candidate['dock'])
        collision=compiler.check.path(states,phases)
        pe=max(e[0] for e in errors);re=max(e[1] for e in errors)
        candidate['prefix_validation']=dict(collision=collision,max_position_error_m=pe,max_rotation_error_rad=re,
            valid=bool(collision['valid'] and pe<.012 and re<.10))
        prefixes[candidate['id']]=(states,phases,errors)
        print('D prefix',candidate['id'],candidate['prefix_validation']['valid'],collision.get('kind','certified'),flush=True)
    ranked=sorted(dock_plan['candidates'],key=lambda c:(not(c['sampled_ik_valid'] and c['prefix_validation']['valid']),c['score'],c['id']))
    selected=ranked[0]
    for candidate in ranked:
        candidate['prefix_executable']=False
        if not (candidate['sampled_ik_valid'] and candidate['prefix_validation']['valid']):continue
        prediction=preview_prefix(ref,candidate,output/f"D-prefix-{candidate['id']}")
        candidate['controller_prefix_preview']=prediction
        candidate['prefix_executable']=prediction['valid']
        print('D controller preview',candidate['id'],prediction['valid'],prediction.get('failure'),flush=True)
        if prediction['valid']:
            selected=candidate;break
        if prediction['failure'].get('phase')=='stow':
            # All nine candidates share exactly the same controller commands
            # before stow ends, regardless of the destination dock.
            for other in ranked:
                other['prefix_executable']=False
                other['shared_stow_rejection_from_candidate']=candidate['id']
            break
    dock_plan['selected']=selected
    records=[]
    for route in ('E','D','A'):
        compiler.solver_receipts=[]
        states,phases,errors=compiler.route(route,points,np.asarray(selected['dock']),prefixes[selected['id']] if route=='D' else None)
        collision=compiler.check.path(states,phases)
        row=compiler.metrics(route,states,phases,errors,selected['id'],collision);row['route']=route
        row['manipulation_solver_receipts']=compiler.solver_receipts
        if route=='D' and not selected.get('prefix_executable',False):
            row['hard_valid']=False;row['features']['hard_valid']=0.
            row['prefix_prediction_rejected']=True
        row['source_qpos_unchanged']=bool(np.array_equal(snapshot,ref.integration()))
        np.savez_compressed(output/f'{route}-path.npz',qpos=states,phases=np.asarray(phases),ik_errors=errors)
        records.append(row);print('candidate',route,'features21','hard_valid',row['hard_valid'],flush=True)
    if not np.array_equal(snapshot,ref.integration()):raise RuntimeError('preflight mutated live state')
    payload=dict(spec=SPEC,records=records,source=str(ref.source),selected_dock_id=selected['id'],
        source_integration_unchanged=True,environment_step_calls=0,formal_train_ready=False,
        planning_physics_substeps=sum(c.get('controller_prefix_preview',{}).get('physics_substeps',0) for c in ranked),
        D_prefix_executable=bool(selected.get('prefix_executable',False)))
    (output/'candidate-features.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    return payload
