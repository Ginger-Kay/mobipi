import numpy as np
import pytest
from mobiwam.teleop_feedback import signed_planar_angle_deg


def direction(degrees):
    x=np.radians(degrees)
    return np.array([np.cos(x),np.sin(x),0.])


def test_world_up_sign_wrap_and_no_mutation():
    hand,door=direction(170),direction(-170)
    old_hand,old_door=hand.copy(),door.copy()
    assert signed_planar_angle_deg(hand,door)==pytest.approx(20.)
    assert signed_planar_angle_deg(door,hand)==pytest.approx(-20.)
    assert signed_planar_angle_deg(direction(0),direction(90))==pytest.approx(90.)
    assert np.array_equal(hand,old_hand) and np.array_equal(door,old_door)


def test_tilted_or_invalid_direction_does_not_offer_yaw_hint():
    assert signed_planar_angle_deg([0,0,1],[1,0,0]) is None
    assert signed_planar_angle_deg([1,0,0],[0,0,-1]) is None
    assert signed_planar_angle_deg([np.nan,0,0],[1,0,0]) is None
