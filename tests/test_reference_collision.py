import mujoco
import numpy as np
import pytest
from mobiwam.reference_collision import SweptGeometry


def slider_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="obstacle" type="sphere" size=".05" pos="0 0 0"/>
      <body name="robot0_body" pos="0 0 0"><joint type="slide" axis="1 0 0"/>
        <geom name="robot0_collision" type="sphere" size=".05" mass="1"/>
      </body></worldbody></mujoco>''')


def test_endpoint_free_but_middle_collision_is_rejected():
    check=SweptGeometry(slider_model())
    result=check.path([[-.3],[.3]],['stow'])
    assert not result['valid'] and result['kind']=='collision'


def test_safe_interval_has_positive_conservative_bound():
    check=SweptGeometry(slider_model())
    result=check.path([[-.4],[-.2]],['navigate'])
    assert result['valid'] and result['lower_bound_m']>=check.margin


def test_source_collision_not_hidden_by_free_midpoint():
    check=SweptGeometry(slider_model())
    result=check.path([[0],[.4]],['stow'])
    assert not result['valid'] and result['kind']=='endpoint_clearance'


def test_scratch_check_does_not_mutate_model_or_live_data():
    model=slider_model();live=mujoco.MjData(model);live.qpos[:]=.7
    before=live.qpos.copy();mask=model.geom_contype.copy();flags=model.opt.enableflags
    check=SweptGeometry(model);check.path([[-.4],[-.2]],['navigate'])
    np.testing.assert_array_equal(before,live.qpos)
    np.testing.assert_array_equal(mask,model.geom_contype)
    assert model.opt.enableflags==flags
    assert not np.shares_memory(model.geom_pos,check.m.geom_pos)


def test_mesh_distance_is_cap_independent_and_witness_order_matches_input():
    vertices=' '.join(str(v) for x in (-.05,.05) for y in (-.05,.05) for z in (-.05,.05) for v in (x,y,z))
    model=mujoco.MjModel.from_xml_string('''<mujoco><asset><mesh name="cube" vertex="'''+vertices+'''"/></asset>
      <worldbody><geom name="obstacle" type="cylinder" size=".02 .05" pos=".08 0 0"/>
      <body><joint type="slide" axis="1 0 0"/><geom name="robot0_mesh" type="mesh" mesh="cube" mass="1"/>
      </body></worldbody></mujoco>''')
    check=SweptGeometry(model);check.distances([0.],'manipulate')
    distances=[]
    for ceiling in (.02,.03,.05,.1):
        segment=np.zeros(6);distances.append(check.geom_distance(1,0,ceiling,segment))
        assert segment[0]<segment[3]  # witnesses are mesh then cylinder
    np.testing.assert_allclose(distances,.01,atol=1e-7)
    assert check.geom_distance(1,0,.005)==.005
    from mobiwam.reference_ik import distance_rows
    jac,_=distance_rows(check,np.array([0.]),'manipulate',np.array([0]))
    np.testing.assert_allclose(jac[0],[-1.],atol=1e-6)


def test_thin_obstacle_between_initial_midpoint_and_endpoint_is_detected():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="obstacle" type="sphere" size=".005" pos=".15 0 0"/>
      <body><joint type="slide" axis="1 0 0"/>
        <geom name="robot0_collision" type="sphere" size=".01" mass="1"/>
      </body></worldbody></mujoco>''')
    result=SweptGeometry(model).path([[-.3],[.3]],['navigate'])
    assert not result['valid'] and result['kind']=='collision'


def test_unresolved_interval_never_becomes_safe_at_depth_limit():
    model=slider_model()
    result=SweptGeometry(model,max_depth=0).path([[-.5],[-.2]],['navigate'])
    assert not result['valid'] and result['kind']=='clearance_unresolved'


def test_hinge_and_slider_motion_bound_covers_geometry_surface():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body pos=".2 0 0"><joint type="hinge" axis="0 0 1" pos=".05 0 0"/>
      <geom name="robot0_link" size=".03" mass="1"/>
      <body pos=".4 0 0"><joint type="slide" axis="1 0 0"/>
      <geom name="robot0_tip" type="box" size=".08 .02 .03" pos=".1 0 0" mass="1"/>
      </body></body></worldbody></mujoco>''')
    check=SweptGeometry(model);data=mujoco.MjData(model)
    q0=np.array([-.7,-.1]);q1=np.array([.9,.2]);bound=check.motion_bounds(q0,q1)
    points=np.array([[1.,0,0],[-1.,0,0],[0,1.,0],[0,0,1.]])
    def surface(q):
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        return np.array([data.geom_xpos[g]+points@data.geom_xmat[g].reshape(3,3).T*model.geom_rbound[g]
                         for g in range(model.ngeom)])
    start=surface(q0)
    for t in np.linspace(.05,1,21):
        actual=np.linalg.norm(surface((1-t)*q0+t*q1)-start,axis=2).max(axis=1)
        assert np.all(actual<=t*bound+1e-12)


def floating_model(kind='free'):
    joint='<freejoint/>' if kind=='free' else '<joint type="ball"/>'
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="obstacle" type="sphere" size=".03" pos="0 .4 0"/>
      <body>'''+joint+'''<geom name="robot0_tip" type="sphere" size=".03" mass="1" pos=".4 0 0"/>
      </body></worldbody></mujoco>''')


@pytest.mark.parametrize('kind',['free','ball'])
def test_quaternion_arc_collision_between_clear_endpoints(kind):
    model=floating_model(kind);check=SweptGeometry(model)
    q0=model.qpos0.copy();q1=q0.copy();start=3 if kind=='free' else 0
    q1[start:start+4]=[0,0,0,1]
    result=check.path([q0,q1],['navigate'])
    assert not result['valid'] and result['kind']=='collision'


@pytest.mark.parametrize('kind',['free','ball'])
def test_rigid_motion_bound_covers_native_geodesic_surface(kind):
    model=floating_model(kind);check=SweptGeometry(model);data=mujoco.MjData(model)
    q0=model.qpos0.copy();q1=q0.copy();start=3 if kind=='free' else 0
    q1[start:start+4]=[np.cos(.6),0,0,np.sin(.6)]
    if kind=='free':q1[:3]=[.2,-.1,.05]
    bounds=check.motion_bounds(q0,q1)
    velocity=np.zeros(model.nv);mujoco.mj_differentiatePos(model,velocity,1.,q0,q1)
    def surface(q):
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        axes=np.r_[np.eye(3),-np.eye(3)]
        return np.array([data.geom_xpos[g]+axes@data.geom_xmat[g].reshape(3,3).T*model.geom_rbound[g] for g in range(model.ngeom)])
    initial=surface(q0)
    for t in np.linspace(0,1,21):
        qt=q0.copy();mujoco.mj_integratePos(model,qt,velocity,t)
        displacement=np.linalg.norm(surface(qt)-initial,axis=2).max(axis=1)
        assert np.all(displacement<=t*bounds+1e-12)


def test_antipodal_quaternions_are_same_rotation():
    model=floating_model();check=SweptGeometry(model)
    q0=model.qpos0.copy();q1=q0.copy();q1[3:]*=-1
    np.testing.assert_allclose(check.motion_bounds(q0,q1),0.,atol=1e-12)
    np.testing.assert_allclose(check.midpoint(q0,q1),q0,atol=1e-12)


def test_nonunit_quaternion_and_articulated_floating_motion_rejected():
    model=floating_model();check=SweptGeometry(model)
    q0=model.qpos0.copy();bad=q0.copy();bad[3:]=0
    with pytest.raises(ValueError,match='quaternion'):check.path([q0,bad],['stow'])
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
    <geom name="robot0_root" type="sphere" size=".03" mass="1"/>
    <body pos=".2 0 0"><joint/><geom name="robot0_child" type="sphere" size=".03" mass="1"/></body>
    </body></worldbody></mujoco>''')
    check=SweptGeometry(model);q0=model.qpos0.copy();q1=q0.copy();q1[0]=.1
    with pytest.raises(ValueError,match='articulated'):check.motion_bounds(q0,q1)
