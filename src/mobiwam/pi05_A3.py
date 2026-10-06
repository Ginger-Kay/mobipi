"""A-private measured-normal contact preload; no fixture trajectory teacher."""
import mujoco
import numpy as np
from mobiwam.contact_rules import allowed_contact
from mobiwam.pi05_adapter import world_intent
from mobiwam.pi05_A2 import execute_A2,A2Driver

def contact_preload(ref):
    m,d=ref.model_data();fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr;found=[]
    for c in d.contact[:d.ncon]:
        for target,hand,sign in ((int(c.geom2),int(c.geom1),1.),(int(c.geom1),int(c.geom2),-1.)):
            tn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,target) or '';hn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,hand) or ''
            if tn.startswith(fixture.name) and hn.startswith('gripper0_') and 'finger' in hn and allowed_contact(tn,hn,'manipulate',fixture.name):
                found.append((float(c.dist),sign*np.asarray(c.frame[:3]),hand,target))
    if found:
        gap,normal,hand,target=min(found,key=lambda x:x[0])
        ref.pi05_A3_contact=dict(normal=normal.copy(),hand=hand,target=target)
    else:
        previous=getattr(ref,'pi05_A3_contact',None)
        if previous is None:return np.zeros(3),dict(applied=False,reason='no observed permitted contact')
        # Recovery of a momentarily separated existing contact is geometric,
        # not a future target pose or a qualification threshold relaxation.
        check=ref.pi05_qp_geometry;check.distances(d.qpos.copy(),'manipulate');segment=np.zeros(6)
        gap=check.geom_distance(previous['hand'],previous['target'],.10,segment)
        if gap>.002:return np.zeros(3),dict(applied=False,reason='last actual contact separated beyond recovery range',gap_m=float(gap))
        normal=segment[3:]-segment[:3]
        if np.linalg.norm(normal)<1e-12:normal=previous['normal']
    normal=np.asarray(normal)/max(np.linalg.norm(normal),1e-12)
    return normal*.0005,dict(applied=True,world_delta_m=(normal*.0005).tolist(),contact_gap_m=float(gap),
        preload_m=.0005,role='measured permitted-contact normal constraint added through same QP; no tangential fixture path or human goal',qualification_thresholds_unchanged=True)

def execute_A3(ref,action,anchor,base_target,**kwargs):
    original=world_intent(action,anchor);delta,record=contact_preload(ref)
    adjusted=np.asarray(action).copy();adjusted[:3]+=anchor['base_world_R'].T@delta
    actual,point=execute_A2(ref,adjusted,anchor,base_target,**kwargs)
    point['projection']['A3_contact_preload']=record
    point['projection']['pi05_world_intent_before_contact_constraint']=dict(pos=original['pos'].tolist(),rot=original['rot'].tolist(),grasp=original['grasp'])
    return actual,point

class A3Driver(A2Driver):
    def receipt(self,queries):
        q=super().receipt(queries);q['A_private_version']='A3';q['A_contact_preload_m']=.0005;return q
