import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from mobiwam.pi05_adapter import world_intent
from unittest.mock import patch
from mobiwam.pi05_adapter import execute_static


def test_nominal_world_intent_is_invariant_when_query_base_changes():
    point=np.array([1.3,-.6,.8]);orientation=Rotation.from_euler('xyz',[.2,-.3,.8]).as_matrix()
    for yaw in (0.,.5,-1.7):
        R=Rotation.from_euler('z',yaw).as_matrix();p=np.array([yaw*.1,.2,0.])
        raw=np.zeros(32);raw[:3]=R.T@(point-p);raw[3:6]=Rotation.from_matrix(R.T@orientation).as_rotvec();raw[6]=1.
        result=world_intent(raw,dict(base_world_p=p,base_world_R=R))
        np.testing.assert_allclose(result['pos'],point,atol=1e-12)
        np.testing.assert_allclose(result['rot'],orientation,atol=1e-12)
        assert result['grasp']==1.


@pytest.mark.parametrize('bad',[np.zeros(7),np.full(32,np.nan)])
def test_invalid_policy_actions_fail_before_native_control(bad):
    with pytest.raises(ValueError,match='finite32'):
        world_intent(bad,dict(base_world_p=np.zeros(3),base_world_R=np.eye(3)))


def test_static_execution_consumes_real_native_mapper_tuple():
    native=np.arange(12,dtype=float)
    with patch('mobiwam.pi05_adapter.mapped_action',return_value=(native,.01,.02,.003)):
        actual,intent=execute_static(object(),np.zeros(32),dict(base_world_p=np.zeros(3),base_world_R=np.eye(3)),np.zeros(3))
    assert actual.shape==(12,) and actual[10]==0 and actual[11]==-1
    assert intent['tracking_error']['position_m']==.01


def test_query_relative_intent_reconstructs_same_absolute_world_goal():
    R=Rotation.from_euler('z',.7).as_matrix();ep=np.array([1.1,-.2,.9]);er=Rotation.from_euler('xyz',[.2,.1,-.3]).as_matrix()
    raw=np.zeros(32);raw[:3]=[.01,-.02,.003];raw[3:6]=[.03,0,-.01]
    out=world_intent(raw,dict(action_representation='query_relative_eef',base_world_R=R,eef_world_p=ep,eef_world_R=er))
    np.testing.assert_allclose(out['pos'],ep+R@raw[:3],atol=1e-12)
    np.testing.assert_allclose(out['rot'],R@Rotation.from_rotvec(raw[3:6]).as_matrix()@R.T@er,atol=1e-12)
