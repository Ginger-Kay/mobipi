"""Passive simulator diagnostics; never edits actions, states, or constraints."""
from pathlib import Path
import json
import mujoco
import numpy as np
from mobiwam.contact_rules import allowed_contact


class SimDiagnostics:
    def __init__(self, ref, path):
        self.ref=ref;self.path=Path(path);self.m,self.d=ref.model_data();self.step=-1
        self.native=[];self.contacts=[];self.last_qp=None;self.last_meta=None;self.current_policy=None
        self.names=[mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(self.m.ngeom)]
        fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
        self.target=fixture.name
        self.hands=[i for i,n in enumerate(self.names) if n.startswith('gripper0_') and
                    ('pad_collision' in n or 'hand_collision' in n)]
        self.targets=[i for i,n in enumerate(self.names) if n.startswith(self.target) and
                      (self.m.geom_contype[i] or self.m.geom_conaffinity[i])]
        self.fixture_joint=next(j for j in range(self.m.njnt) if (mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_JOINT,j) or '').startswith(self.target) and self.m.jnt_type[j] in (2,3))
        self.fixture_qid=int(self.m.jnt_qposadr[self.fixture_joint])
        self.handle=next((i for i in self.targets if 'handle' in self.names[i]),None)
        self.qp_log=(self.path/'qp-diagnostics.jsonl').open('x')
        self.native_method='step2' if ref.env.lite_physics else 'step'
        self.qp_velocity=np.zeros(11)

    def boundary(self, step, policy_action=None):
        self.step=int(step);self.current_policy=None if policy_action is None else np.asarray(policy_action).copy()

    def record_qp(self, J, target, lower, upper, inequalities, velocity, receipt, origins, controlled):
        A,b=inequalities if inequalities is not None else (np.empty((0,len(velocity))),np.empty(0))
        violation=np.maximum(0.,np.asarray(b)-np.asarray(A)@velocity)
        self.last_qp=dict(jacobian=np.asarray(J).copy(),target=np.asarray(target).copy(),A=np.asarray(A).copy(),b=np.asarray(b).copy(),
                         lower=np.asarray(lower).copy(),upper=np.asarray(upper).copy(),velocity=np.asarray(velocity).copy(),
                         qpos=self.d.qpos.copy(),qvel=self.d.qvel.copy(),sim_time=np.array(float(self.d.time)))
        self.qp_velocity=np.asarray(velocity).copy()
        self.last_meta=dict(step=self.step,sim_time=float(self.d.time),solver=dict(receipt),constraint_rows=origins,
                           variable_names=controlled,inequality_violation=violation.tolist(),
                           max_inequality_violation=float(np.max(violation,initial=0)),
                           lower_violation=np.maximum(0,lower-velocity).tolist(),upper_violation=np.maximum(0,velocity-upper).tolist(),
                           projection_iterations_limit=32,actual_iterations=receipt.get('projection_iterations','missing'),
                           policy_grasp=float(self.current_policy[6]) if self.current_policy is not None else None,
                           scope='Exact original arrays passed to baseline solver; diagnostics do not change them')
        self.qp_log.write(json.dumps(self.last_meta)+'\n');self.qp_log.flush()
        if self.step==0 or self.step%500==0:self.save_qp('normal-step-'+str(self.step))

    def save_qp(self, label):
        if self.last_qp is None:return
        np.savez_compressed(self.path/('qp-system-'+label+'.npz'),**self.last_qp)
        (self.path/('qp-system-'+label+'.json')).write_text(json.dumps(self.last_meta,indent=2)+'\n')

    def sample(self, before_time, before_qpos):
        base=self.ref.robot.part_controllers['base'];arm=self.ref.robot.part_controllers['right']
        gaps=[];normals=[];closest=[]
        for hand in self.hands:
            best=.10;normal=np.full(3,np.nan);gid=-1
            for target in self.targets:
                segment=np.zeros(6);gap=float(mujoco.mj_geomDistance(self.m,self.d,hand,target,.10,segment))
                if gap<best:
                    best=gap;gid=target;direction=segment[3:]-segment[:3];length=np.linalg.norm(direction)
                    normal=direction/length if length>1e-12 else np.full(3,np.nan)
            gaps.append(best);normals.append(normal);closest.append(gid)
        observed=[]
        for c in self.d.contact[:self.d.ncon]:
            a,b=int(c.geom1),int(c.geom2);an,bn=self.names[a],self.names[b]
            if (an.startswith('gripper0_') and bn.startswith(self.target)) or (bn.startswith('gripper0_') and an.startswith(self.target)):
                observed.append(dict(geom1=a,geom2=b,gap_m=float(c.dist),normal=c.frame[:3].tolist(),position_m=c.pos.tolist(),
                                     permitted=allowed_contact(an,bn,'manipulate',self.target)))
        if observed:self.contacts.append(dict(step=self.step,cache_time=before_time,after_integration_time=float(self.d.time),contacts=observed))
        site=self.ref.robot.eef_site_id['right'];handle=self.d.geom_xpos[self.handle].copy() if self.handle is not None else np.full(3,np.nan)
        self.native.append(dict(before_time=before_time,time=float(self.d.time),step=self.step,base=self.d.qpos[base.qpos_index].copy(),
            arm=self.d.qpos[arm.qpos_index].copy(),base_velocity=self.d.qvel[base.qvel_index].copy(),arm_velocity=self.d.qvel[arm.qvel_index].copy(),
            fixture=float(self.d.qpos[self.fixture_qid]),gap=np.asarray(gaps),normal=np.asarray(normals),closest=np.asarray(closest),
            eef_cached=self.d.site_xpos[site].copy(),handle_cached=handle,qp_base=self.qp_velocity[:3].copy(),
            gripper_goal=self.ref.robot.gripper['right'].current_action.copy(),gripper_qpos=self.d.qpos[11:13].copy(),
            any_permitted_contact=any(c['permitted'] for c in observed)))

    def __enter__(self):
        sim=self.ref.env.sim;self.original=getattr(sim,self.native_method);self.had=self.native_method in vars(sim);self.instance=vars(sim).get(self.native_method)
        def tracked(*args,**kwargs):
            before_time=float(self.d.time);before_qpos=self.d.qpos.copy();result=self.original(*args,**kwargs)
            self.sample(before_time,before_qpos);return result
        setattr(sim,self.native_method,tracked);return self

    def __exit__(self,*args):
        if self.had:setattr(self.ref.env.sim,self.native_method,self.instance)
        else:delattr(self.ref.env.sim,self.native_method)

    def save(self):
        self.qp_log.close()
        if self.native:
            arrays={k:np.asarray([r[k] for r in self.native]) for k in self.native[0]}
            np.savez_compressed(self.path/'diagnostic-native-motion.npz',**arrays)
        with (self.path/'diagnostic-native-contacts.jsonl').open('w') as f:
            for row in self.contacts:f.write(json.dumps(row)+'\n')
        (self.path/'diagnostic-native-binding.json').write_text(json.dumps(dict(hands=[dict(id=i,name=self.names[i]) for i in self.hands],
            targets=[dict(id=i,name=self.names[i]) for i in self.targets],native_method=self.native_method,
            contact_cache_time='before integration for mj_step2, recorded separately from after-integration qpos/time; no live forward call',
            geometry_gap_scope='Read-only live distance using cached poses; distance cap0.10m is a diagnostic measurement cap, never action/qualification threshold',
            normal_scope='nearest-point direction; nan when undefined; contact normals separately from native contact table',
            samples=len(self.native),action_changes_from_logger=False,state_changes_from_logger=False),indent=2)+'\n')
