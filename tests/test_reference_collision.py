import mujoco
import numpy as np
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
    before=live.qpos.copy();mask=model.geom_contype.copy()
    SweptGeometry(model).path([[-.4],[-.2]],['navigate'])
    np.testing.assert_array_equal(before,live.qpos)
    np.testing.assert_array_equal(mask,model.geom_contype)
