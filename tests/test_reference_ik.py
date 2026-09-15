import mujoco
import numpy as np
from mobiwam.reference_collision import SweptGeometry
from mobiwam.reference_ik import constrained_pose_ik, distance_rows, swept_pose_segment


def model_and_data():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="wall" type="plane" size="1 1 .1"/>
      <body pos="0 0 0"><joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
        <geom name="robot0_hand" type="sphere" size=".05" mass="1"/>
        <site name="eef"/>
      </body></worldbody></mujoco>''')
    return model, mujoco.MjData(model)


def test_signed_distance_gradient_matches_finite_difference():
    model, data = model_and_data(); check = SweptGeometry(model)
    q = np.array([.1, .2, .049])  # include penetration normal convention
    jac, distances = distance_rows(check, q, 'manipulate', np.arange(3))
    numeric = []
    for k in range(3):
        delta = np.eye(3)[k] * 1e-6
        numeric.append((check.distances(q + delta, 'manipulate')[1][0] -
                        check.distances(q - delta, 'manipulate')[1][0]) / 2e-6)
    np.testing.assert_allclose(jac[0], numeric, atol=1e-8)


def test_ik_reaches_clear_target_without_mutating_live_state():
    model, data = model_and_data(); live = mujoco.MjData(model); live.qpos[:] = [.2,.1,.2]
    limits = np.array([[-1., 1.]] * 3)
    q, pe, re, receipt = constrained_pose_ik(model, data, 0, np.arange(3), np.arange(3),
        dict(pos=np.array([.1, .2, .15]), rot=np.eye(3)), live.qpos.copy(), limits, SweptGeometry(model))
    assert pe < .0002 and re < .002
    np.testing.assert_array_equal(live.qpos, [.2,.1,.2])


def test_conflicting_pose_does_not_relax_collision_constraint():
    model, data = model_and_data(); check = SweptGeometry(model)
    q, pe, re, receipt = constrained_pose_ik(model, data, 0, np.arange(3), np.arange(3),
        dict(pos=np.zeros(3), rot=np.eye(3)), np.array([0., 0., .06]),
        np.array([[-1., 1.]] * 3), check)
    assert pe > .05  # target is impossible without penetrating the wall
    assert receipt['minimum_endpoint_distance_m'] >= .000999
    assert not receipt['collision_gate_relaxed']


def test_refinement_removes_between_waypoint_collision_without_moving_obstacle():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="obstacle" type="sphere" size=".005"/>
      <body><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/><geom name="robot0_hand" size=".005" mass="1"/>
      <site name="eef"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model); check = SweptGeometry(model)
    start = np.array([-.03,.009,0.]); end = np.array([.03,.009,0.])
    assert not check.path([start, end], ['manipulate'])['valid']
    states, errors, receipts = swept_pose_segment(model, data, 0, np.arange(3), np.arange(3),
        dict(pos=end, rot=np.eye(3)), start, end, np.array([[-1.,1.]]*3), check)
    assert len(states) > 1
    assert check.path([start]+states, ['manipulate']*len(states))['valid']
    assert max(e[0] for e in errors) < .009
    np.testing.assert_array_equal(model.geom_pos[0], np.zeros(3))
