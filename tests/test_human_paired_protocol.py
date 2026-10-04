import numpy as np
import pytest
from mobiwam.human_paired_protocol import route_inputs,validate_dock

def parts():return dict(right=np.ones(6)*.06,right_gripper=np.array([1.]),base=np.array([.35,.35,.2]),base_mode=1.)

def test_d_navigation_preserves_base_and_blocks_arm_grasp_without_mutating_input():
    x=parts();y=route_inputs(x,'D',False)
    assert np.all(y['right']==0) and y['right_gripper'][0]==-1
    assert np.array_equal(y['base'],x['base']) and y['base_mode']==1
    assert np.all(x['right']==.06) and x['right_gripper'][0]==1

@pytest.mark.parametrize('route,docked',[('E',False),('D',True)])
def test_locked_base_keeps_original_arm_speed(route,docked):
    x=parts();y=route_inputs(x,route,docked)
    assert np.array_equal(x['right'],y['right']) and np.all(y['base']==0) and y['base_mode']==-1

def test_a_unchanged():
    x=parts();y=route_inputs(x,'A',False)
    assert all(np.array_equal(x[k],y[k]) for k in x)

@pytest.mark.parametrize('changes',[
    {'route':'A'},{'docked':True},{'base_qvel':[.02,0,0]},{'base_qvel':[0,0,.03]},
    {'arm_qpos':[.06]*7},{'closed':True},{'target_contact':True},{'base_qvel':[float('nan'),0,0]}])
def test_invalid_dock_rejected(changes):
    args=dict(route='D',docked=False,base_qvel=[0,0,0],arm_qpos=[0]*7,stow_qpos=[0]*7,closed=False,target_contact=False)
    args.update(changes)
    with pytest.raises(ValueError):validate_dock(**args)

def test_valid_stopped_open_stowed_dock():
    assert validate_dock('D',False,[0,0,0],[0]*7,[0]*7,False,False)
