import numpy as np
import pytest
from mobiwam.reference_transfer import _map_drawer_joint_by_closed_progress

def test_shorter_microwave_opening_preserves_reference_fraction_and_closed_endpoint():
    old=-1.4782294331140298;new=-.8891660852145085;limits=np.array([-1.57,0.])
    assert _map_drawer_joint_by_closed_progress(old,old,new,limits,limits)==new
    assert _map_drawer_joint_by_closed_progress(0.,old,new,limits,limits)==0.
    value=_map_drawer_joint_by_closed_progress(-.0741449995710391,old,new,limits,limits)
    assert value<0 and np.isclose(value/new,(-.0741449995710391)/old)
    with pytest.raises(ValueError):_map_drawer_joint_by_closed_progress(.01,old,new,limits,limits)
