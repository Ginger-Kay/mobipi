import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from mobiwam.pi05_adapter import world_intent


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
