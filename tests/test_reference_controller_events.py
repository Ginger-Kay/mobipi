import copy
from types import SimpleNamespace
import numpy as np,pytest
from mobiwam.reference_controller_events import validate_events,apply_event

def event(kind='dock_settled_reobserve_feedback_reset',step=4):
    return dict(event=kind,step=step,timing='before_action',arm_nullspace_goal=list(np.arange(7,dtype=float)))

class Arm:
    def __init__(self,calls):self.calls=calls;self.initial_joint=np.zeros(7);self.goal_pos=np.zeros(3)
    def set_goal_update_mode(self,x):self.calls.append(('mode',x))
    def set_goal(self,x):self.calls.append(('goal',x.tolist()))

def fixture():
    calls=[];arm=Arm(calls);controller=SimpleNamespace(update_state=lambda:calls.append('update'))
    env=SimpleNamespace(_get_observations=lambda **k:calls.append('observe'),rng=np.random.default_rng(7))
    return SimpleNamespace(env=env,robot=SimpleNamespace(part_controllers={'right':arm},composite_controller=controller)),calls

def test_shared_reset_before_action_order():
    ref,calls=fixture();receipt=apply_event(ref,event());calls.append('native_action')
    assert calls==['observe','update',('mode','achieved'),('goal',[0.]*6),'native_action']
    assert receipt['after']['arm']['initial_joint']==list(range(7))

def test_step0_stow_changes_only_nullspace():
    ref,calls=fixture();apply_event(ref,event('initial_stow_nullspace',0));assert calls==[]
    assert ref.robot.part_controllers['right'].initial_joint.tolist()==list(range(7))

def test_missing_unknown_duplicate_rejected():
    with pytest.raises(ValueError,match='missing initial stow'):validate_events([], 'D',10,True)
    with pytest.raises(ValueError,match='unknown'):validate_events([event('unknown')],'D',10)
    with pytest.raises(ValueError,match='duplicate'):validate_events([event(),event()],'D',10)
    with pytest.raises(ValueError,match='missing controller event payload'):validate_events([dict(event(),arm_nullspace_goal=None)],'D',10)

def test_wrong_event_timing_and_initial_step_rejected():
    with pytest.raises(ValueError,match='timing'):validate_events([dict(event(),timing='after_action')],'D',10)
    with pytest.raises(ValueError,match='action0'):validate_events([event('initial_stow_nullspace',1)],'D',10)
    with pytest.raises(ValueError,match='step/order'):validate_events([event(step=11)],'D',10)

def test_e_a_have_no_reset():
    assert validate_events([],'E',10)==validate_events([],'A',10)==[]
    with pytest.raises(ValueError,match='E/A'):validate_events([event()],'E',10)
