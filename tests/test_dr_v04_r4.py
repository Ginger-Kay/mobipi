import numpy as np
import pytest
from mobiwam.dr_v04_r4 import explain_selection
from mobiwam.dr_v04_r3_learning import select

def test_windows_units_and_constant_collision_are_identical_to_r3():
    p=np.array([[.9,0,.5,.2,30],[.88,0,.7,.3,10],[.2,0,.9,.1,1]])
    result=explain_selection(p,[0,2,3,4])
    assert result["route"]=="D" and select(p,[0,2,3,4])==1
    assert result["stages"][0]["remaining"]==["E","D"]
    assert result["stages"][1]["skipped"] and result["stages"][2]["remaining"]==["D"]
    assert explain_selection(p,[0,2,3,4],[False,False,True])["route"]=="A"

def test_no_supported_head_or_candidate_is_unavailable():
    p=np.zeros((3,5))
    assert explain_selection(p,[])["route"] is None
    assert explain_selection(p,[0],[False,False,False])["route"] is None
    assert explain_selection(p,[0,2,3,4])["route"]=="E"

def test_nonfinite_supported_head_is_not_silently_ranked():
    p=np.zeros((3,5));p[1,0]=np.nan
    with pytest.raises(ValueError):explain_selection(p,[0])
