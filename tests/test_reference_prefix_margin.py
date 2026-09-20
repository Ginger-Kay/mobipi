import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location(
    'prefix_geometry', Path(__file__).resolve().parents[1] / 'scripts/reference_geometry_v16.py')
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def fixture():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><compiler angle="radian"/><worldbody><body>
    <joint type="hinge" axis="0 0 1" range="-.2 .2" limited="true"/>
    <geom type="sphere" size=".05"/><site pos="1 0 0"/>
    </body></worldbody></mujoco>''')
    return model, mujoco.MjData(model)


def solve(angle, margin=None, seed=0.):
    model, data = fixture()
    rot = Rotation.from_euler('z', angle).as_matrix()
    return geometry.pose_ik(model, data, 0, np.array([0]), np.array([0]),
                            dict(pos=rot @ np.array([1., 0., 0.]), rot=rot),
                            np.array([seed]), model.jnt_range, joint_margin=margin)


@pytest.mark.parametrize('angle', [-.4, .4])
def test_buffer_keeps_strict_margin_without_hiding_pose_failure(angle):
    old, _, _ = solve(angle)
    new, pe, _ = solve(angle, .01501)
    assert .2 - abs(old[0]) == pytest.approx(.015)
    assert .2 - abs(new[0]) > .015
    assert pe > .012


def test_buffer_applies_before_early_convergence():
    q, _, _ = solve(.185, .01501, seed=.185)
    assert .2 - abs(q[0]) > .015


def test_interior_solution_unchanged():
    for left, right in zip(solve(0.), solve(0., .01501)):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('margin', [0., -.1, float('nan'), float('inf'), .2])
def test_invalid_or_relaxed_margin_rejected(margin):
    with pytest.raises(ValueError, match='invalid IK joint margin'):
        solve(.1, margin)
