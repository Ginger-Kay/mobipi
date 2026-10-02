"""Observe native physical substeps on a formal candidate without changing physics.

This is a guard and an evidence recorder. It is not a continuous swept
clearance certificate; that must be checked on the recorded native states.
"""
from __future__ import annotations
import numpy as np
import mujoco


class FormalSafetyStop(RuntimeError):
    def __init__(self, failure):
        super().__init__('DR-v0.4 native substep unsafe contact')
        self.failure = failure


class FormalSubstepMonitor:
    def __init__(self, ref, target_name):
        self.ref = ref
        self.model, self.data = ref.model_data()
        self.sim = ref.env.sim
        self.name = 'step2' if ref.env.lite_physics else 'step'
        self.names = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
                      for i in range(self.model.ngeom)]
        self.target = str(target_name)
        self.states = [self.data.qpos.copy()]
        self.phases = []
        self.steps = []
        self.times = [float(self.data.time)]
        self.step = -1
        self.phase = 'precontact'
        self.first_forbidden = None

    def set_boundary(self, step, phase):
        self.step,self.phase=int(step),str(phase)

    def forbidden_contacts(self):
        for c in self.data.contact[:self.data.ncon]:
            a,b=self.names[c.geom1],self.names[c.geom2]
            robot_a=a.startswith(('robot0_','gripper0_','mobilebase0_'))
            robot_b=b.startswith(('robot0_','gripper0_','mobilebase0_'))
            if not (robot_a or robot_b):continue
            if ('floor' in a and b.startswith('mobilebase0_')) or ('floor' in b and a.startswith('mobilebase0_')):
                continue
            if self.phase=='manipulate' and ((robot_a and 'finger' in a and b.startswith(self.target)) or
                                             (robot_b and 'finger' in b and a.startswith(self.target))):
                continue
            yield dict(geom1=a,geom2=b,distance_m=float(c.dist))

    def __enter__(self):
        self.original=getattr(self.sim,self.name)
        self.had_instance_value=self.name in vars(self.sim)
        self.instance_value=vars(self.sim).get(self.name)
        def guarded(*args,**kwargs):
            outcome=self.original(*args,**kwargs)
            self.states.append(self.data.qpos.copy())
            self.times.append(float(self.data.time))
            self.phases.append(self.phase)
            self.steps.append(self.step)
            contacts=list(self.forbidden_contacts())
            if contacts:
                self.first_forbidden=dict(kind='native_forbidden_contact',step=self.step,
                    substep=len(self.phases)-1,phase=self.phase,contacts=contacts)
                raise FormalSafetyStop(self.first_forbidden)
            return outcome
        setattr(self.sim,self.name,guarded)
        return self

    def __exit__(self,*exc):
        if self.had_instance_value:setattr(self.sim,self.name,self.instance_value)
        else:delattr(self.sim,self.name)

    def save(self,path):
        if len(self.states)!=len(self.times) or len(self.phases)!=len(self.steps) or len(self.states)!=len(self.phases)+1:
            raise ValueError('native substep trace misaligned')
        np.savez_compressed(path/'formal-native-substeps.npz',
            qpos=np.asarray(self.states,dtype=np.float64),
            sim_time=np.asarray(self.times,dtype=np.float64),
            step_index=np.asarray(self.steps,dtype=np.int64),
            phases=np.asarray(self.phases,dtype='<U12'))
        return dict(substeps=len(self.phases),states=len(self.states),
            forbidden_contact=self.first_forbidden,
            scope='native integration qpos plus contact guard; actual continuous swept 0.5mm clearance still requires separate audit')
